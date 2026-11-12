#!/usr/bin/env bash
# =============================================================================
# lef2tf.sh - generate a Synopsys technology file (.tf) from the ASAP7 tech LEF.
#
# ASAP7 does not ship a .tf.  rmbrum/lef2tf converts the LEF into one that
# icc2_lm_shell / fc_shell accept via `create_workspace -technology`.
#
# The macro LEF is passed as a second input so its SITE definition
# (asap7sc7p5t, 0.216 x 1.08) is emitted as a Tile.
# =============================================================================
set -euo pipefail

ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
PDK="${PDK_DIR:-$ROOT/pdk_data}"
TOOLS="$PDK/tools"
LEF2TF="$TOOLS/lef2tf"

TECH_LEF="$PDK/lef/asap7_tech_4x_201209.lef"
CELL_LEF="$PDK/lef/asap7sc7p5t_27_R_4x_201211.lef"
OUT_TF="$PDK/asap7_4x.tf"

for f in "$TECH_LEF" "$CELL_LEF"; do
  [[ -f "$f" ]] || { echo "missing $f (run pdk/fetch_pdk.sh first)" >&2; exit 1; }
done

if [[ ! -f "$LEF2TF/lef2tf.py" ]]; then
  mkdir -p "$TOOLS"
  echo "[git ] cloning lef2tf"
  git clone --depth 1 https://github.com/rmbrum/lef2tf "$LEF2TF"
fi

echo "[tf  ] $TECH_LEF + $CELL_LEF -> $OUT_TF"
python3 "$LEF2TF/lef2tf.py" "$TECH_LEF" "$CELL_LEF" -o "$OUT_TF" --name asap7

# Verify the tool accepted the generated technology file (if available).
if command -v "${LM_SHELL:-icc2_lm_shell}" >/dev/null 2>&1; then
  cat > "$PDK/.check_tf.tcl" <<EOF
create_workspace -technology $OUT_TF tfcheck
check_workspace
remove_workspace
exit
EOF
  echo "[chk ] validating .tf with ${LM_SHELL:-icc2_lm_shell}"
  ( cd "$PDK" && "${LM_SHELL:-icc2_lm_shell}" -f .check_tf.tcl ) || \
    echo "WARN: .tf validation failed; inspect $OUT_TF before building NDM" >&2
fi

echo "wrote $OUT_TF"
