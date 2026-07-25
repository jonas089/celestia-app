#!/bin/bash
# Device-ceiling benchmark: the GKR prover is single-core per proof, so run K
# concurrent workers (= K cores) each proving an 8-lane batch (reps=2, so each
# reports a WARM, compile-amortized block). Aggregate steady-state TPS = sum of
# each worker's warm tx/s; peak RSS = summed across workers (compile is the
# memory-heavy phase). Sweep K until memory-bound.
set -u
BIN=/Users/jonas/Desktop/celestia-app/rust/crates/rv32-rollup/target/release/tx_bench_simd
OUT=/private/tmp/claude-501/-Users-jonas-Desktop-celestia-app/ce3ffb86-e06e-46e9-acec-815702606135/scratchpad
N=${N:-4}            # tx per lane  (8 lanes -> 8N tx per block per worker)
KS="$@"; [ -z "$KS" ] && KS="1 2 3 4"
printf "%-3s %-9s %-9s %-11s %-9s %s\n" K workers total_tx warm_ms/wk peakGB agg_tps
for K in $KS; do
  pids=(); logs=()
  for i in $(seq 1 "$K"); do
    log="$OUT/par_k${K}_w${i}.log"; logs+=("$log")
    TX_ACCOUNTS=8 "$BIN" "$N" 8 2 >"$log" 2>&1 &
    pids+=($!)
  done
  # sample summed RSS of all tx_bench_simd workers while they run
  peak=0
  while :; do
    alive=0; for p in "${pids[@]}"; do kill -0 "$p" 2>/dev/null && alive=1; done
    [ "$alive" = 0 ] && break
    rss=$(ps -A -o rss,comm | awk '/tx_bench_simd/{s+=$1} END{print s+0}')
    [ "$rss" -gt "$peak" ] && peak=$rss
    sleep 2
  done
  fail=0; for p in "${pids[@]}"; do wait "$p" || fail=1; done
  # warm (rep=1) prove_ms per worker; aggregate tps = sum of per-worker warm tps
  warm=$(grep -h "rep=1" "${logs[@]}" 2>/dev/null | grep -oE "prove_ms=[0-9]+" | grep -oE "[0-9]+")
  txper=$(( 8 * N ))
  # average warm ms, aggregate tps = K workers each doing txper in ~warm ms
  avg=$(echo "$warm" | awk '{s+=$1;n++} END{if(n)printf "%d", s/n; else print 0}')
  aggtps=$(awk "BEGIN{ if($avg>0) printf \"%.2f\", $K*$txper*1000.0/$avg; else print 0 }")
  peakgb=$(awk "BEGIN{printf \"%.1f\", $peak/1048576}")
  total=$(( K * txper ))
  if [ "$fail" != 0 ] || [ -z "$warm" ]; then
    printf "%-3s %-9s %-9s %-11s %-9s FAILED/OOM (peak %sGB)\n" "$K" "$K" "$total" "$avg" "$peakgb" "$peakgb"
    break
  fi
  printf "%-3s %-9s %-9s %-11s %-9s %s\n" "$K" "$K" "$total" "$avg" "$peakgb" "$aggtps"
done
echo "done."
