// Command accidental-dashboard-api is the read-only HTTP backend for the
// accidental-computer dashboard. It is a thin, honest proxy: every value it
// serves comes from the rv32-rollup's accProof_* JSON-RPC (the node that
// actually produces the GKR proofs). Nothing here fabricates a proof or a
// commitment — on any RPC error the error is surfaced to the UI.
//
// Routes (all consumed by test/accidental-computer/dashboard):
//
//	GET /api/health                        liveness
//	GET /api/blockproofs                   live per-block STF proofs, grouped
//	GET /api/blockproof?ns=..&height=..    one block's proof record
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
)

// server carries no state: every handler is a proxy onto the rollup's RPC.
type server struct{}

func main() {
	addr := envOr("API_ADDR", ":8088")
	srv := &server{}
	mux := http.NewServeMux()
	mux.HandleFunc("/api/blockproofs", srv.handleBlockProofs)
	mux.HandleFunc("/api/blockproof", srv.handleBlockProof) // ?ns=..&height=.. -> accProof_getBlockProof
	mux.HandleFunc("/api/health", func(w http.ResponseWriter, r *http.Request) { writeJSON(w, map[string]string{"status": "ok"}) })

	log.Printf("accidental-computer API on %s (proxying accProof_* at %s, namespace %s)",
		addr, envOr("ACCPROOF_RPC", "http://localhost:8545"), rollupNS())
	log.Fatal(http.ListenAndServe(addr, cors(mux)))
}

// --- per-block STF proofs (accProof_listBlockProofs) ---
//
// The rv32-rollup proves the state transition of EVERY rollup block
// asynchronously and caches the result keyed by (namespace, blockNumber). This
// endpoint polls that cache so the dashboard can show a live list of blocks
// with each block's STF proof status. Proofs lag block production; nothing here
// is mocked.

// rollupNS is the rollup namespace this dashboard surfaces. Configurable via
// ROLLUP_NS.
func rollupNS() string { return envOr("ROLLUP_NS", "rv32-rollup") }

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

// stfScope is the honest scope statement surfaced to the dashboard when the
// rollup RPC is unreachable (the rollup supplies its own scope otherwise).
const stfScope = "rv32i block-STF: every rollup block is proved DIRECTLY as a GKR circuit over an " +
	"in-circuit RV32I machine, with rsema1d reused as the SOLE polynomial commitment from the DA " +
	"handle (ZERO prover re-encoding, byte-identical to Go/DA). The circuit binds the computed final " +
	"registers/memory to the committed public post-state and derives pre_root/post_root with in-circuit " +
	"keccak. Out of scope: RV32I base only (no M extension), bounded committed memory slots, fixed " +
	"unroll depth, and a toy keyed-mixing transaction signature rather than a real signature scheme."

// --- helpers ---

func namespaceFor(name string) []byte {
	d := sha256.Sum256([]byte("ns:" + name))
	ns := make([]byte, 29)
	copy(ns, d[:29])
	return ns
}

func hexBytes(b []byte) string { return "0x" + hex.EncodeToString(b) }
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
