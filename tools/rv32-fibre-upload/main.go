// Command rv32-fibre-upload publishes the rv32 GKR prover's rsema1d input square
// to a Fibre DA network and records its commitment on-chain via MsgPayForFibre,
// so the prover can later open its GKR proof against that exact commitment.
//
// It reads the raw input-vals byte vector (one GF2x8 byte per hypercube coeff,
// length 2^numVars), reproduces the committed square byte-identically via
// rsema1d.EncodeGKRInputSquare, wraps it into a fibre Blob WITHOUT re-encoding or
// injecting the v0 header (fibre.NewBlobFromExtendedData), uploads the shards +
// broadcasts MsgPayForFibre, and writes the serialized extended matrix (the exact
// format the Rust prover's install_da_commitment_from_serialized consumes) for
// the prover to load.
package main

import (
	"context"
	"crypto/rand"
	"encoding/binary"
	"encoding/hex"
	"encoding/json"
	"flag"
	"fmt"
	"log"
	"math"
	"os"
	"time"

	"github.com/celestiaorg/celestia-app/v10/app"
	"github.com/celestiaorg/celestia-app/v10/app/encoding"
	"github.com/celestiaorg/celestia-app/v10/fibre"
	"github.com/celestiaorg/celestia-app/v10/pkg/rsema1d"
	"github.com/celestiaorg/celestia-app/v10/pkg/user"
	"github.com/celestiaorg/celestia-app/v10/x/fibre/types"
	"github.com/celestiaorg/go-square/v4/share"
	"github.com/cosmos/cosmos-sdk/crypto/keyring"
	sdk "github.com/cosmos/cosmos-sdk/types"
	"google.golang.org/grpc"
	"google.golang.org/grpc/credentials/insecure"
)

// output is the JSON printed to stdout on success.
type output struct {
	Commitment   string `json:"commitment"`
	DAHeight     int64  `json:"daHeight"`
	TxHash       string `json:"txHash"`
	Namespace    string `json:"namespace"`
	ExtendedFile string `json:"extendedFile"`
	BlobID       string `json:"blobId"`
}

func main() {
	if err := run(); err != nil {
		log.Fatal(err)
	}
}

func run() error {
	var (
		inputValsFile  = flag.String("input-vals-file", "", "path to raw input-vals bytes (length 2^num-vars, one GF2x8 byte per coeff)")
		numVars        = flag.Int("num-vars", 0, "number of multilinear variables; input-vals must be 2^num-vars bytes")
		extendedOut    = flag.String("extended-out", "", "path to write the serialized extended matrix for the prover")
		namespaceHex   = flag.String("namespace", "", "hex namespace ID (10 bytes for v0); random if empty")
		chainID        = flag.String("chain-id", "test", "chain ID of the network")
		keyName        = flag.String("key-name", "validator", "key name in keyring")
		keyringBackend = flag.String("keyring-backend", "test", "keyring backend (only 'test' supported)")
		home           = flag.String("home", "", "home directory (default: $HOME/.celestia-app)")
		grpcAddr       = flag.String("grpc-addr", "localhost:9090", "gRPC address of the node")
		timeout        = flag.Duration("timeout", 60*time.Second, "timeout for network operations")
	)
	flag.Parse()

	if *inputValsFile == "" {
		return fmt.Errorf("--input-vals-file is required")
	}
	if *numVars < 2 {
		return fmt.Errorf("--num-vars must be >= 2, got %d", *numVars)
	}
	if *extendedOut == "" {
		return fmt.Errorf("--extended-out is required")
	}
	if *home == "" {
		*home = os.Getenv("HOME") + "/.celestia-app"
	}
	if *keyringBackend != "test" {
		return fmt.Errorf("unsupported keyring backend: %s (only 'test' is supported)", *keyringBackend)
	}

	// 1) read input-vals and reproduce the committed rsema1d square.
	inputVals, err := os.ReadFile(*inputValsFile)
	if err != nil {
		return fmt.Errorf("reading input-vals file: %w", err)
	}
	wantLen := 1 << uint(*numVars)
	if len(inputVals) != wantLen {
		return fmt.Errorf("input-vals length %d != 2^num-vars = %d", len(inputVals), wantLen)
	}
	ed, err := rsema1d.EncodeGKRInputSquare(inputVals, *numVars)
	if err != nil {
		return fmt.Errorf("encoding GKR input square: %w", err)
	}
	commitment := ed.Commitment()
	log.Printf("encoded rsema1d square: num-vars=%d commitment=%s", *numVars, hex.EncodeToString(commitment[:]))

	// 2) wrap the already-encoded square into a fibre Blob (no re-encode, no header).
	blob, err := fibre.NewBlobFromExtendedData(ed)
	if err != nil {
		return fmt.Errorf("wrapping extended data into blob: %w", err)
	}
	defer blob.Free()
	if blob.ID().Commitment() != fibre.Commitment(commitment) {
		return fmt.Errorf("internal error: blob commitment %s != square commitment %s",
			blob.ID().Commitment(), hex.EncodeToString(commitment[:]))
	}
	log.Printf("blob: K=%d N=%d rowLen=%d uploadSize=%d blobID=%s",
		blob.Config().OriginalRows, blob.Config().ParityRows, blob.RowSize(), blob.UploadSize(), blob.ID())

	// 3) namespace.
	ns, err := resolveNamespace(*namespaceHex)
	if err != nil {
		return err
	}
	log.Printf("namespace: %s", ns.String())

	// 4) write the serialized extended matrix for the prover BEFORE the network
	// round-trip, so the prover artifact exists even if settlement is retried.
	if err := writeExtended(*extendedOut, ed, blob.Config().OriginalRows, blob.Config().ParityRows, blob.RowSize()); err != nil {
		return fmt.Errorf("writing extended matrix: %w", err)
	}
	log.Printf("wrote extended matrix: %s", *extendedOut)

	// 5) connect + upload + settle.
	ctx, cancel := context.WithTimeout(context.Background(), *timeout)
	defer cancel()
	res, err := uploadAndSettle(ctx, uploadCfg{
		grpcAddr: *grpcAddr, chainID: *chainID, keyName: *keyName, home: *home,
	}, ns, blob)
	if err != nil {
		return fmt.Errorf("uploading/settling on fibre: %w", err)
	}

	out := output{
		Commitment:   hex.EncodeToString(commitment[:]),
		DAHeight:     res.height,
		TxHash:       res.txHash,
		Namespace:    hex.EncodeToString(ns.Bytes()),
		ExtendedFile: *extendedOut,
		BlobID:       blob.ID().String(),
	}
	enc := json.NewEncoder(os.Stdout)
	return enc.Encode(&out)
}

func resolveNamespace(nsHex string) (share.Namespace, error) {
	var nsID []byte
	if nsHex == "" {
		nsID = make([]byte, share.NamespaceVersionZeroIDSize)
		if _, err := rand.Read(nsID); err != nil {
			return share.Namespace{}, fmt.Errorf("generating random namespace: %w", err)
		}
	} else {
		b, err := hex.DecodeString(nsHex)
		if err != nil {
			return share.Namespace{}, fmt.Errorf("decoding namespace hex: %w", err)
		}
		if len(b) != share.NamespaceVersionZeroIDSize {
			return share.Namespace{}, fmt.Errorf("namespace ID must be %d bytes, got %d", share.NamespaceVersionZeroIDSize, len(b))
		}
		nsID = b
	}
	id := make([]byte, 0, share.NamespaceIDSize)
	id = append(id, share.NamespaceVersionZeroPrefix...)
	id = append(id, nsID...)
	return share.NewNamespace(share.NamespaceVersionZero, id)
}

// writeExtended serializes the full K+N extended row matrix in the EXACT format
// consumed by the Rust prover's install_da_commitment_from_serialized (and by
// pkg/rsema1d/cshim/cshim_da.go serializeExtended): little-endian u32 K, u32 N,
// u32 rowLen, then (K+N)*rowLen row-major bytes (originals then RS parity).
func writeExtended(path string, ed *rsema1d.ExtendedData, k, n, rowLen int) error {
	total := k + n
	buf := make([]byte, 12+total*rowLen)
	binary.LittleEndian.PutUint32(buf[0:4], uint32(k))
	binary.LittleEndian.PutUint32(buf[4:8], uint32(n))
	binary.LittleEndian.PutUint32(buf[8:12], uint32(rowLen))
	off := 12
	for i := 0; i < total; i++ {
		row := ed.Row(i)
		if len(row) != rowLen {
			return fmt.Errorf("row %d has length %d, expected %d", i, len(row), rowLen)
		}
		copy(buf[off:off+rowLen], row)
		off += rowLen
	}
	return os.WriteFile(path, buf, 0o644)
}

type uploadCfg struct {
	grpcAddr string
	chainID  string
	keyName  string
	home     string
}

type settleResult struct {
	txHash string
	height int64
}

// uploadAndSettle mirrors fibre.Put but uploads a pre-built Blob (from
// NewBlobFromExtendedData) instead of encoding raw bytes, and keeps
// single-validator behavior (MaxValidatorCount=1) like tools/submit-fibre-blob.
func uploadAndSettle(ctx context.Context, cfg uploadCfg, ns share.Namespace, blob *fibre.Blob) (settleResult, error) {
	encCfg := encoding.MakeConfig(app.ModuleEncodingRegisters...)

	kr, err := keyring.New(app.Name, keyring.BackendTest, cfg.home, nil, encCfg.Codec)
	if err != nil {
		return settleResult{}, fmt.Errorf("initializing keyring: %w", err)
	}

	grpcConn, err := grpc.NewClient(
		cfg.grpcAddr,
		grpc.WithTransportCredentials(insecure.NewCredentials()),
		grpc.WithDefaultCallOptions(
			grpc.MaxCallSendMsgSize(math.MaxInt32),
			grpc.MaxCallRecvMsgSize(math.MaxInt32),
		),
	)
	if err != nil {
		return settleResult{}, fmt.Errorf("creating gRPC connection: %w", err)
	}
	defer grpcConn.Close()

	params := fibre.DefaultProtocolParams
	params.MaxValidatorCount = 1 // single-node testnet
	clientCfg := fibre.NewClientConfigFromParams(params)
	clientCfg.StateAddress = cfg.grpcAddr
	clientCfg.DefaultKeyName = cfg.keyName

	txClient, err := user.SetupTxClient(ctx, kr, grpcConn, encCfg, user.WithDefaultAccount(cfg.keyName))
	if err != nil {
		return settleResult{}, fmt.Errorf("setting up tx client: %w", err)
	}

	fibreClient, err := fibre.NewClient(kr, clientCfg)
	if err != nil {
		return settleResult{}, fmt.Errorf("creating fibre client: %w", err)
	}
	defer func() {
		if err := fibreClient.Stop(ctx); err != nil {
			log.Printf("stopping fibre client: %v", err)
		}
	}()
	if err := fibreClient.Start(ctx); err != nil {
		return settleResult{}, fmt.Errorf("starting fibre client: %w", err)
	}

	// Upload shards + collect validator signatures over the payment promise.
	log.Printf("uploading blob to fibre (upload_size=%d)...", blob.UploadSize())
	signedPromise, err := fibreClient.Upload(ctx, ns, blob, fibre.WithKeyName(cfg.keyName))
	if err != nil {
		return settleResult{}, fmt.Errorf("fibre upload: %w", err)
	}
	log.Printf("upload succeeded: collected %d validator signature(s)", len(signedPromise.ValidatorSignatures))

	// Broadcast MsgPayForFibre recording the commitment on-chain.
	promiseProto, err := signedPromise.ToProto()
	if err != nil {
		return settleResult{}, fmt.Errorf("converting payment promise to proto: %w", err)
	}
	msg := &types.MsgPayForFibre{
		Signer:              txClient.DefaultAddress().String(),
		PaymentPromise:      *promiseProto,
		ValidatorSignatures: signedPromise.ValidatorSignatures,
	}
	broadcastResp, err := txClient.BroadcastTx(ctx, []sdk.Msg{msg})
	if err != nil {
		return settleResult{}, fmt.Errorf("broadcasting MsgPayForFibre: %w", err)
	}
	txResp, err := txClient.ConfirmTx(ctx, broadcastResp.TxHash)
	if err != nil {
		return settleResult{}, fmt.Errorf("confirming MsgPayForFibre: %w", err)
	}
	if txResp.Code != 0 {
		return settleResult{}, fmt.Errorf("MsgPayForFibre failed on-chain: code=%d codespace=%s", txResp.Code, txResp.Codespace)
	}
	return settleResult{txHash: txResp.TxHash, height: txResp.Height}, nil
}
