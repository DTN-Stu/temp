#!/usr/bin/env bash
# =============================================================================
# run.sh - Build + run the snax_llm_top_clean testbenches with FST traces.
#
# Targets : clean (FlashAttention) | gemm (dense GEMM) | llama (LLM shape) | all
# Outputs : sim_out/build/<t>/        Verilator build tree
#           sim_out/waves/<t>.fst     waveform (FST)
#           sim_out/logs/<t>.log      simulation log
#
# Usage   : ./run.sh [target] [options] [-- <sim plusargs>]
# =============================================================================
set -euo pipefail

ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
RTL_DIR="$ROOT/rtl/01_snax_llm_fp4_subsystem"
TB_GEN_DIR="$ROOT/tb/tb"     # python stimulus generators
SIM_DATA_DIR="$ROOT/tb/sim"  # TB reads sim_snax_llm_top_*/hex relative to cwd
FL_DIR="$ROOT/filelists"
OUT="$ROOT/sim_out"
JOBS="$(nproc 2>/dev/null || echo 4)"

usage() {
  cat <<EOF
Usage: ./run.sh [target] [options] [-- <sim plusargs>]

  target: all | clean | gemm | llama | layer   (default: all)
          (layer = CPU+DMA+GEMM system smoke test, reuses gemm stimulus)

Options:
  --no-gen      Skip regenerating stimulus/golden (reuse tb/sim data)
  --no-build    Reuse an existing Verilator build (skip rebuild)
  --no-trace    Run without FST waveform (much faster)
  --deep        Dump the whole TB hierarchy (+WAVE_DEEP); default dumps DUT only
  -j N          Build parallelism (default: $JOBS)
  -h, --help    Show this help

Examples:
  ./run.sh clean                 # build + run FlashAttention TB, dump dut
  ./run.sh gemm --deep           # full-hierarchy FST for the GEMM TB
  ./run.sh llama --no-gen        # reuse existing stimulus
  ./run.sh clean -- +NO_WAVE     # run without waveform (fast)

Waveforms land in: $OUT/waves/
EOF
}

TARGET="all"; DO_GEN=1; DO_BUILD=1; DO_TRACE=1; DEEP=0
SIM_PLUS=()

while [[ $# -gt 0 ]]; do
  case "$1" in
    all|clean|gemm|llama|layer) TARGET="$1"; shift ;;
    --no-gen)   DO_GEN=0; shift ;;
    --no-build) DO_BUILD=0; shift ;;
    --no-trace) DO_TRACE=0; shift ;;
    --deep)     DEEP=1; shift ;;
    -j)         JOBS="$2"; shift 2 ;;
    -h|--help)  usage; exit 0 ;;
    --)         shift; SIM_PLUS+=("$@"); break ;;
    *) echo "ERROR: unknown option '$1'"; echo; usage; exit 1 ;;
  esac
done

mkdir -p "$OUT/build" "$OUT/waves" "$OUT/logs"

gen_one() {
  case "$1" in
    clean) ( cd "$TB_GEN_DIR" && python3 generate_snax_llm_top_clean_data.py ) ;;
    gemm)  ( cd "$TB_GEN_DIR" && python3 generate_snax_llm_top_gemm_data.py ) ;;
    llama) ( cd "$TB_GEN_DIR" && python3 generate_snax_llm_top_llama_data.py ) ;;
    layer) ( cd "$TB_GEN_DIR" && python3 generate_snax_llm_top_gemm_data.py ) ;;
  esac
}

# Map a short target name to its top module.
top_of() {
  case "$1" in
    layer) echo "tb_snax_llm_layer_smoke" ;;
    *)     echo "tb_snax_llm_top_$1" ;;
  esac
}

run_one() {
  local name="$1"
  local top
  top="$(top_of "$name")"
  local bdir="$OUT/build/$name"
  local wave="$OUT/waves/$top.fst"
  local log="$OUT/logs/$top.log"
  local fl="$FL_DIR/tb_$name.f"

  [[ -f "$fl" ]] || { echo "ERROR: missing filelist $fl"; return 1; }

  if [[ "$DO_BUILD" == 1 ]]; then
    echo ">>> [build] $top"
    mkdir -p "$bdir"
    ( cd "$ROOT" && verilator --binary --timing -j "$JOBS" -Wno-fatal \
        --trace-fst -DFST_TRACE --top-module "$top" \
        -f "$fl" --Mdir "$bdir" -o "sim_$name" ) \
      2>&1 | tee "$OUT/logs/build_$name.log" | tail -n 3
  fi

  local args=()
  [[ "$DO_TRACE" == 1 ]] && args+=(--trace)
  [[ "$DEEP" == 1 ]] && args+=(+WAVE_DEEP)
  [[ ${#SIM_PLUS[@]} -gt 0 ]] && args+=("${SIM_PLUS[@]}")

  echo ">>> [run] $top ${args[*]:-<no args>}"
  set +e
  ( cd "$SIM_DATA_DIR" && "$bdir/sim_$name" "${args[@]}" ) 2>&1 | tee "$log"
  local rc=${PIPESTATUS[0]}
  set -e

  # The TB writes the FST in its cwd (tb/sim); move it next to the other outputs.
  if [[ -f "$SIM_DATA_DIR/$top.fst" ]]; then
    mv -f "$SIM_DATA_DIR/$top.fst" "$wave"
    echo ">>> waveform: $wave ($(du -h "$wave" | cut -f1))"
  fi

  echo ">>> result : $top rc=$rc   (log: $log)"
  return "$rc"
}

case "$TARGET" in
  all) TARGETS=(clean gemm llama) ;;
  *)   TARGETS=("$TARGET") ;;
esac

if [[ "$DO_GEN" == 1 ]]; then
  echo ">>> [gen] stimulus + golden"
  for t in "${TARGETS[@]}"; do gen_one "$t"; done
fi

fail=0
for t in "${TARGETS[@]}"; do
  run_one "$t" || fail=1
done

[[ "$fail" == 0 ]] && echo ">>> ALL DONE" || echo ">>> SOME TARGETS FAILED"
exit "$fail"
