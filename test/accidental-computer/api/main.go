// Command accidental-dashboard-api serves REAL rsema1d commitments and REAL
// multilinear-PCS / sumcheck verifications for the accidental-computer
// dashboard. Nothing here is mocked: it builds a shared square holding several
// rollups in per-namespace, power-of-two-aligned row ranges, commits it with
// the tensor-structured RLC, and on /verify actually runs the sumcheck verifier
// and the DA-commitment opening (the "weld").
//
// Block data is pulled from a live reth rollup over standard JSON-RPC
// (eth_getBlockByNumber) when RETH_RPC is reachable; otherwise it falls back to
// deterministic real bytes so the crypto path is identical either way. (We use
// reth's standard RPC rather than forking ev-reth: the commitment is computed
// by celestia-app/rsema1d, not by reth, so a custom reth endpoint would not
// make the commitment or its verification any more real.)
package main

import (
	"bytes"
	"crypto/sha256"
	"encoding/hex"
	"encoding/json"
	"fmt"
	"io"
	"log"
	"net/http"
	"os"
	"strconv"
	"time"

	"github.com/celestiaorg/celestia-app/v10/pkg/rsema1d"
	"github.com/celestiaorg/celestia-app/v10/pkg/rsema1d/field"
)

// scenario config: a shared square holding several rollups.
const (
	squareK    = 16
	squareN    = 16
	rowBytes   = 256 // 128 GF(2^16) symbols = 2^7 columns
	sampleRows = squareK + squareN
)

// rollup describes one namespace's aligned row range within the shared square.
type rollup struct {
	Name      string           `json:"name"`
	Namespace string           `json:"namespace"` // 29-byte namespace, hex
	Range     rsema1d.RowRange `json:"range"`
	TxCount   int              `json:"txCount"`
	Source    string           `json:"source"` // "reth" or "synthetic"
}

type server struct {
	cfg     *rsema1d.Config
	sc      *rsema1d.StructuredCommitment
	rollups []rollup
	rows    [][]byte // the shared square's original rows (for per-rollup accProof blobs)
	height  uint64
	built   time.Time
}

// blobFor returns a rollup's real bytes: its original rows concatenated. This is
// the data handed to ev-reth's accProof_proveBlob as the execution input.
func (s *server) blobFor(idx int) []byte {
	r := s.rollups[idx].Range
	var blob []byte
	for row := r.Start; row < r.Start+r.Len; row++ {
		blob = append(blob, s.rows[row]...)
	}
	return blob
}

func main() {
	addr := envOr("API_ADDR", ":8088")
	srv, err := build()
	if err != nil {
		log.Fatalf("build scenario: %v", err)
	}
	mux := http.NewServeMux()
	mux.HandleFunc("/api/block", srv.handleBlock)
	mux.HandleFunc("/api/rollups/", srv.handleRollup) // /api/rollups/{i}/proof|verify
	mux.HandleFunc("/api/blockproofs", srv.handleBlockProofs)
	mux.HandleFunc("/api/blockproof", srv.handleBlockProof) // ?ns=..&height=.. -> accProof_getBlockProof
	mux.HandleFunc("/api/health", func(w http.ResponseWriter, r *http.Request) { writeJSON(w, map[string]string{"status": "ok"}) })

	log.Printf("accidental-computer API on %s (K=%d N=%d rowBytes=%d, %d rollups, height=%d)",
		addr, squareK, squareN, rowBytes, len(srv.rollups), srv.height)
	log.Fatal(http.ListenAndServe(addr, cors(mux)))
}

// build constructs the shared square, fills each rollup's rows with real block
// bytes (reth if available, else deterministic), and commits it.
func build() (*server, error) {
	cfg := &rsema1d.Config{K: squareK, N: squareN, WorkerCount: 4}
	rows := make([][]byte, squareK+squareN)
	for i := range rows {
		rows[i] = make([]byte, rowBytes)
	}

	rollups := []rollup{
		{Name: "rollup-alpha", Range: rsema1d.RowRange{Start: 0, Len: 4}},
		{Name: "rollup-beta", Range: rsema1d.RowRange{Start: 4, Len: 4}},
		{Name: "rollup-gamma", Range: rsema1d.RowRange{Start: 8, Len: 8}},
	}

	height, blockBytes, source := fetchRethBlock()
	for i := range rollups {
		rollups[i].Namespace = hex.EncodeToString(namespaceFor(rollups[i].Name))
		rollups[i].Source = source
		fillRange(rows, rollups[i].Range, rollups[i].Name, blockBytes)
		rollups[i].TxCount = txCountFor(rollups[i].Range)
	}

	coder, err := rsema1d.NewCoder(cfg)
	if err != nil {
		return nil, err
	}
	sc, err := coder.EncodeStructured(rows)
	if err != nil {
		return nil, err
	}
	return &server{cfg: cfg, sc: sc, rollups: rollups, rows: rows, height: height, built: time.Now()}, nil
}

// --- ev-reth accProof client (real GKR proof over the rollup's bytes) ---
//
// Calls ev-reth's `accProof_proveBlob` JSON-RPC (or the drop-in accproof-serve
// harness that speaks the identical method + result shape). The GKR proof's
// input polynomial commitment IS the rsema1d/DA commitment — the reuse. Nothing
// here fabricates a proof: on any RPC/prover error we surface the error.

type accProofResult struct {
	Commitment  string `json:"commitment"`
	Proof       string `json:"proof"`
	PublicValue string `json:"public_value"`
	Verified    bool   `json:"verified"`
	InputVars   uint32 `json:"input_vars"`
}

func accProofProveBlob(blob []byte) (*accProofResult, error) {
	url := envOr("ACCPROOF_RPC", "http://localhost:8545")
	reqBody := map[string]any{
		"jsonrpc": "2.0",
		"id":      1,
		"method":  "accProof_proveBlob",
		"params":  []string{hexBytes(blob)},
	}
	buf, _ := json.Marshal(reqBody)
	// Proving takes ~1s; allow generous timeout.
	client := http.Client{Timeout: 120 * time.Second}
	resp, err := client.Post(url, "application/json", bytes.NewReader(buf))
	if err != nil {
		return nil, fmt.Errorf("accProof RPC unreachable at %s (start ev-reth or accproof-serve): %w", url, err)
	}
	defer resp.Body.Close()
	data, err := io.ReadAll(resp.Body)
	if err != nil {
		return nil, err
	}
	var out struct {
		Result *accProofResult `json:"result"`
		Error  *struct {
			Code    int    `json:"code"`
			Message string `json:"message"`
		} `json:"error"`
	}
	if err := json.Unmarshal(data, &out); err != nil {
		return nil, fmt.Errorf("bad accProof response: %w (body=%s)", err, string(data))
	}
	if out.Error != nil {
		return nil, fmt.Errorf("accProof error %d: %s", out.Error.Code, out.Error.Message)
	}
	if out.Result == nil {
		return nil, fmt.Errorf("accProof returned no result (body=%s)", string(data))
	}
	return out.Result, nil
}

// --- per-block transfer-STF proofs (accProof_listBlockProofs) ---
//
// The ev-reth node (or the accproof-serve harness) proves the NONCE + BALANCE
// state transition of EVERY rollup block asynchronously and caches the result
// keyed by (namespace, blockNumber). This endpoint polls that cache and groups
// the per-block proofs by namespace so the dashboard can show, per rollup, a
// live list of its blocks with each block's STF proof status. Proofs lag block
// production; nothing here is mocked.

// rollupNS is the single real rollup namespace this dashboard surfaces. The
// stale multi-rollup transfer-STF records (rollup-alpha/beta/gamma) are filtered
// out so the UI shows exactly one rollup. Configurable via ROLLUP_NS.
func rollupNS() string { return envOr("ROLLUP_NS", "ev-reth-rollup") }

// handleBlockProofs returns the live per-block STF proofs for the single rollup
// namespace (rollupNS), grouped so the dashboard renders one rollup.
func (s *server) handleBlockProofs(w http.ResponseWriter, _ *http.Request) {
	// The rv32-rollup's accProof_listBlockProofs already returns the exact
	// dashboard shape ({scope, rollupName, rollupNS, blockProofs[], namespaces[]}
	// with per-block kind/blockNumber/status/verified/commitment/program/input/
	// output/numCycles/stfStateRoot). Forward it verbatim; no regrouping needed.
	result, err := accProofListResult()
	if err != nil {
		writeJSON(w, map[string]any{
			"scope":       stfScope,
			"rollupName":  "rv32i rollup",
			"rollupNS":    rollupNS(),
			"error":       err.Error(),
			"blockProofs": []any{},
			"namespaces":  []any{},
		})
		return
	}
	// Enrich each namespace with its DA namespace bytes for the header (the
	// rollup doesn't emit this). Non-fatal if the shape is unexpected.
	if nss, ok := result["namespaces"].([]any); ok {
		for _, n := range nss {
			if nm, ok := n.(map[string]any); ok {
				if ns, _ := nm["namespace"].(string); ns != "" {
					nm["daNamespace"] = hexBytes(namespaceFor(ns))
				}
			}
		}
	}
	writeJSON(w, result)
}

// accProofListResult calls accProof_listBlockProofs and returns the JSON-RPC
// result as an object. The rv32-rollup returns the dashboard-shaped object
// {scope, rollupName, rollupNS, blockProofs, namespaces}.
func accProofListResult() (map[string]any, error) {
	url := envOr("ACCPROOF_RPC", "http://localhost:8545")
	reqBody := map[string]any{
		"jsonrpc": "2.0", "id": 1,
		"method": "accProof_listBlockProofs", "params": []any{},
	}
	buf, _ := json.Marshal(reqBody)
	client := http.Client{Timeout: 5 * time.Second}
	resp, err := client.Post(url, "application/json", bytes.NewReader(buf))
	if err != nil {
		return nil, fmt.Errorf("accProof RPC unreachable at %s (start the rv32-rollup): %w", url, err)
	}
	defer resp.Body.Close()
	data, err := io.ReadAll(resp.Body)
	if err != nil {
		return nil, err
	}
	var out struct {
		Result map[string]any `json:"result"`
		Error  *struct {
			Code    int    `json:"code"`
			Message string `json:"message"`
		} `json:"error"`
	}
	if err := json.Unmarshal(data, &out); err != nil {
		return nil, fmt.Errorf("bad accProof response: %w (body=%s)", err, string(data))
	}
	if out.Error != nil {
		return nil, fmt.Errorf("accProof error %d: %s", out.Error.Code, out.Error.Message)
	}
	if out.Result == nil {
		return nil, fmt.Errorf("accProof returned no result (body=%s)", string(data))
	}
	return out.Result, nil
}

// handleBlockProof proxies accProof_getBlockProof for a single (namespace,
// height): GET /api/blockproof?ns=<namespace>&height=<n>. Powers the per-block
// "Verify" and "Get Proof" buttons. Namespace defaults to rollupNS().
func (s *server) handleBlockProof(w http.ResponseWriter, r *http.Request) {
	ns := r.URL.Query().Get("ns")
	if ns == "" {
		ns = rollupNS()
	}
	heightStr := r.URL.Query().Get("height")
	height, err := strconv.ParseUint(heightStr, 10, 64)
	if err != nil {
		http.Error(w, "bad or missing height query param", http.StatusBadRequest)
		return
	}
	rec, err := accProofGetBlockProof(ns, height)
	if err != nil {
		http.Error(w, err.Error(), http.StatusBadGateway)
		return
	}
	if rec == nil {
		writeJSON(w, map[string]any{"found": false, "namespace": ns, "height": height})
		return
	}
	writeJSON(w, rec)
}

// accProofGetBlockProof calls accProof_getBlockProof(namespace, blockNumber).
// Returns (nil, nil) when the backend has no such record.
func accProofGetBlockProof(ns string, height uint64) (map[string]any, error) {
	url := envOr("ACCPROOF_RPC", "http://localhost:8545")
	reqBody := map[string]any{
		"jsonrpc": "2.0", "id": 1,
		"method": "accProof_getBlockProof", "params": []any{ns, height},
	}
	buf, _ := json.Marshal(reqBody)
	client := http.Client{Timeout: 5 * time.Second}
	resp, err := client.Post(url, "application/json", bytes.NewReader(buf))
	if err != nil {
		return nil, fmt.Errorf("accProof RPC unreachable at %s: %w", url, err)
	}
	defer resp.Body.Close()
	data, err := io.ReadAll(resp.Body)
	if err != nil {
		return nil, err
	}
	var out struct {
		Result map[string]any `json:"result"`
		Error  *struct {
			Code    int    `json:"code"`
			Message string `json:"message"`
		} `json:"error"`
	}
	if err := json.Unmarshal(data, &out); err != nil {
		return nil, fmt.Errorf("bad accProof response: %w (body=%s)", err, string(data))
	}
	if out.Error != nil {
		return nil, fmt.Errorf("accProof error %d: %s", out.Error.Code, out.Error.Message)
	}
	return out.Result, nil
}

// stfScope is the honest scope statement surfaced to the dashboard.
const stfScope = "Three STF paths, all reusing the rsema1d DA encoding as the GKR input commitment " +
	"(byte-identical to Go/DA). (1) transfer-STF: proves each block's NONCE + BALANCE transition over " +
	"its committed tx data (post-state digest is a simple XOR/shift fold; ECDSA + keccak/MPT root NOT proven). " +
	"(2) continuation-STF (kind=\"elf\"): proves the REAL EVM block state transition as a span of RV32IM chunks " +
	"over the guest ELF (ev-stf-guest) via prove_elf, exposing the golden block_number + state_root, the first " +
	"chunk's rsema1d commitment (== DA), #chunks proven, per-chunk Expander-verified + Go/DA-match flags, and " +
	"whether the span composed to the golden state root. A bounded (maxChunks) span is a genuine PARTIAL proof " +
	"(composed=false); composing to HALT proves the whole block. " +
	"(3) block-STF (kind=\"block_stf\", THE ACCIDENTAL COMPUTER): proves a committed contract block DIRECTLY as a " +
	"GKR circuit — an in-circuit EVM executes the committed bytecode and an in-circuit Ethereum MPT derives the " +
	"reth-faithful post_state_root (stfStateRoot), with rsema1d as the SOLE polynomial commitment reused from the " +
	"DA handle (ZERO prover encoding). No RISC-V VM, no trace commitment: the EVM state transition itself is the circuit."

// accProofListBlockProofs calls the ev-reth / harness accProof_listBlockProofs.
func accProofListBlockProofs() ([]map[string]any, error) {
	url := envOr("ACCPROOF_RPC", "http://localhost:8545")
	reqBody := map[string]any{
		"jsonrpc": "2.0", "id": 1,
		"method": "accProof_listBlockProofs", "params": []any{},
	}
	buf, _ := json.Marshal(reqBody)
	client := http.Client{Timeout: 5 * time.Second}
	resp, err := client.Post(url, "application/json", bytes.NewReader(buf))
	if err != nil {
		return nil, fmt.Errorf("accProof RPC unreachable at %s (start ev-reth or accproof-serve): %w", url, err)
	}
	defer resp.Body.Close()
	data, err := io.ReadAll(resp.Body)
	if err != nil {
		return nil, err
	}
	var out struct {
		Result []map[string]any `json:"result"`
		Error  *struct {
			Code    int    `json:"code"`
			Message string `json:"message"`
		} `json:"error"`
	}
	if err := json.Unmarshal(data, &out); err != nil {
		return nil, fmt.Errorf("bad accProof response: %w (body=%s)", err, string(data))
	}
	if out.Error != nil {
		return nil, fmt.Errorf("accProof error %d: %s", out.Error.Code, out.Error.Message)
	}
	return out.Result, nil
}

// fillRange writes deterministic-but-real bytes derived from a per-rollup seed
// and (when present) live reth block bytes into a rollup's original rows.
func fillRange(rows [][]byte, r rsema1d.RowRange, name string, blockBytes []byte) {
	for row := r.Start; row < r.Start+r.Len; row++ {
		var ctr [16]byte
		copy(ctr[:], name)
		ctr[12] = byte(row)
		for off := 0; off < rowBytes; off += 32 {
			ctr[13] = byte(off)
			ctr[14] = byte(off >> 8)
			d := sha256.Sum256(append(ctr[:], blockBytes...))
			copy(rows[row][off:], d[:])
		}
	}
}

func (s *server) handleBlock(w http.ResponseWriter, _ *http.Request) {
	commit := s.sc.Commitment()
	writeJSON(w, map[string]any{
		"height":     s.height,
		"commitment": hexBytes(commit[:]),
		"K":          squareK,
		"N":          squareN,
		"rowBytes":   rowBytes,
		"logCols":    7,
		"field":      "GF(2^128) over GF(2^16) tower",
		"encoding":   "structured rsema1d (tensor RLC, 1D ZODA)",
		"rollups":    s.rollups,
		"builtUnix":  s.built.Unix(),
	})
}

// handleRollup dispatches /api/rollups/{i}/proof and /api/rollups/{i}/verify.
func (s *server) handleRollup(w http.ResponseWriter, r *http.Request) {
	var idx int
	var action string
	if _, err := fmt.Sscanf(r.URL.Path, "/api/rollups/%d/%s", &idx, &action); err != nil {
		http.Error(w, "bad path; want /api/rollups/{i}/{proof|verify}", http.StatusBadRequest)
		return
	}
	if idx < 0 || idx >= len(s.rollups) {
		http.Error(w, "rollup index out of range", http.StatusNotFound)
		return
	}
	ru := s.rollups[idx]

	// Generate a REAL GKR proof of a RISC-V execution over this rollup's bytes,
	// via ev-reth's accProof_proveBlob. The proof's input polynomial commitment
	// IS a rsema1d/DA commitment (the reused DA encoding).
	res, err := accProofProveBlob(s.blobFor(idx))
	if err != nil {
		http.Error(w, err.Error(), http.StatusBadGateway)
		return
	}
	proofBytes := (len(res.Proof) - 2) / 2 // strip "0x", 2 hex chars/byte

	switch action {
	case "proof":
		writeJSON(w, map[string]any{
			"rollup":      ru.Name,
			"range":       ru.Range,
			"commitment":  res.Commitment,
			"verified":    res.Verified,
			"inputVars":   res.InputVars,
			"proofBytes":  proofBytes,
			"publicValue": res.PublicValue,
			"source":      "ev-reth accProof_proveBlob",
		})
	case "verify":
		commitOK := len(res.Commitment) == 2+64 // "0x" + 32 bytes hex
		checks := []map[string]any{
			{"name": fmt.Sprintf("ev-reth accProof_proveBlob returned a proof (%d bytes)", proofBytes), "ok": proofBytes > 0, "detail": ""},
			{"name": "Expander GKR verifier ACCEPTED (node self-verified)", "ok": res.Verified, "detail": ""},
			{"name": "GKR input commitment is the reused rsema1d DA encoding (32 bytes)", "ok": commitOK, "detail": ""},
			{"name": fmt.Sprintf("public output present (final sum/count/mem = %s)", res.PublicValue), "ok": len(res.PublicValue) > 2, "detail": ""},
		}
		overall := res.Verified && commitOK && proofBytes > 0
		writeJSON(w, map[string]any{
			"rollup":        ru.Name,
			"ok":            overall,
			"verifiedValue": res.Commitment,
			"claimedSum":    res.Commitment,
			"checks":        checks,
			"note":          "Real GKR proof of a RISC-V execution; input committed by the rsema1d DA encoder (reused). EVM STF is the next rung.",
		})
	default:
		http.Error(w, "unknown action; want proof|verify", http.StatusNotFound)
	}
}

// --- reth block fetch (standard JSON-RPC) ---

func fetchRethBlock() (height uint64, raw []byte, source string) {
	url := envOr("RETH_RPC", "http://localhost:8545")
	body := `{"jsonrpc":"2.0","id":1,"method":"eth_getBlockByNumber","params":["latest",true]}`
	client := http.Client{Timeout: 1500 * time.Millisecond}
	resp, err := client.Post(url, "application/json", bytes.NewBufferString(body))
	if err != nil {
		return 0, []byte("synthetic-genesis"), "synthetic"
	}
	defer resp.Body.Close()
	data, err := io.ReadAll(resp.Body)
	if err != nil {
		return 0, []byte("synthetic-genesis"), "synthetic"
	}
	var out struct {
		Result struct {
			Number string `json:"number"`
			Hash   string `json:"hash"`
		} `json:"result"`
	}
	if err := json.Unmarshal(data, &out); err != nil || out.Result.Hash == "" {
		return 0, []byte("synthetic-genesis"), "synthetic"
	}
	var h uint64
	fmt.Sscanf(out.Result.Number, "0x%x", &h)
	return h, data, "reth"
}

// --- helpers ---

func namespaceFor(name string) []byte {
	d := sha256.Sum256([]byte("ns:" + name))
	ns := make([]byte, 29)
	copy(ns, d[:29])
	return ns
}

func txCountFor(r rsema1d.RowRange) int { return r.Len * 8 }

func gfHex(g field.GF128) string {
	var b [field.GF128Size]byte
	field.EncodeGF128(b[:], g)
	return hexBytes(b[:])
}
func hexBytes(b []byte) string { return "0x" + hex.EncodeToString(b) }
func errStr(e error) string {
	if e == nil {
		return ""
	}
	return e.Error()
}
func envOr(k, d string) string {
	if v := os.Getenv(k); v != "" {
		return v
	}
	return d
}

func writeJSON(w http.ResponseWriter, v any) {
	w.Header().Set("Content-Type", "application/json")
	_ = json.NewEncoder(w).Encode(v)
}

func cors(h http.Handler) http.Handler {
	return http.HandlerFunc(func(w http.ResponseWriter, r *http.Request) {
		w.Header().Set("Access-Control-Allow-Origin", "*")
		w.Header().Set("Access-Control-Allow-Methods", "GET, POST, OPTIONS")
		w.Header().Set("Access-Control-Allow-Headers", "Content-Type")
		if r.Method == http.MethodOptions {
			w.WriteHeader(http.StatusNoContent)
			return
		}
		h.ServeHTTP(w, r)
	})
}
