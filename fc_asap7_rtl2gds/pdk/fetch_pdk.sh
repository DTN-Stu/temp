#!/usr/bin/env bash
# =============================================================================
# fetch_pdk.sh - download the parts of the ASAP7 7nm predictive PDK that this
# flow needs.  Only the RVT 4x-scaled views are fetched by default.
#
#   bash pdk/fetch_pdk.sh                # tech LEF, RVT cell LEF, RVT NLDM libs
#   bash pdk/fetch_pdk.sh --all-vt       # also LVT and SLVT cell LEF + libs
#   bash pdk/fetch_pdk.sh --with-qrc     # also the single QRC techfile (~65 MB)
#
# Requires: curl, python3, and py7zr for .lib.7z extraction
#   python3 -m pip install py7zr
# =============================================================================
set -euo pipefail

ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
PDK="${PDK_DIR:-$ROOT/pdk_data}"
BASE="https://raw.githubusercontent.com/The-OpenROAD-Project/asap7sc7p5t_27/main"
ALL_VT=0
WITH_QRC=0
for a in "$@"; do
  case "$a" in
    --all-vt)   ALL_VT=1 ;;
    --with-qrc) WITH_QRC=1 ;;
    *) echo "unknown option: $a" >&2; exit 2 ;;
  esac
done

mkdir -p "$PDK/lef" "$PDK/lib" "$PDK/lib7z" "$PDK/ndm" "$PDK/tluplus" "$PDK/qrc"

dl() {
  local url="$1" out="$2"
  if [[ -s "$out" ]]; then echo "[skip] $(basename "$out")"; return; fi
  echo "[get ] $url"
  curl -fsSL "$url" -o "$out"
}

# --- Technology + physical LEF (4x scaled; tech LEF is 4x) -------------------
dl "$BASE/techlef_misc/asap7_tech_4x_201209.lef" \
   "$PDK/lef/asap7_tech_4x_201209.lef"

VTS="R"
[[ $ALL_VT -eq 1 ]] && VTS="R L SL"
for vt in $VTS; do
  dl "$BASE/LEF/scaled/asap7sc7p5t_27_${vt}_4x_201211.lef" \
     "$PDK/lef/asap7sc7p5t_27_${vt}_4x_201211.lef"
done

# --- NLDM timing libraries ---------------------------------------------------
# Cell set is partitioned by function; all groups must be linked together.
FUNCS="AO INVBUF OA SEQ SIMPLE"
CORNERS="TT FF SS"
for vt in $VTS; do
  case "$vt" in
    R)  vtl=RVT  ;;
    L)  vtl=LVT  ;;
    SL) vtl=SLVT ;;
    *)  vtl=$vt  ;;
  esac
  for f in $FUNCS; do
    for c in $CORNERS; do
      name="asap7sc7p5t_${f}_${vtl}_${c}_nldm_201020"
      dl "$BASE/LIB/NLDM/${name}.lib.7z" "$PDK/lib7z/${name}.lib.7z"
    done
  done
done

if [[ $WITH_QRC -eq 1 ]]; then
  dl "$BASE/qrc/qrcTechFile_typ03_scaled4xV06" \
     "$PDK/qrc/qrcTechFile_typ03_scaled4xV06"
fi

# --- Extract .lib.7z -> .lib -------------------------------------------------
python3 - "$PDK" <<'PY'
import glob, os, sys
pdk = sys.argv[1]
try:
    import py7zr
except ImportError:
    sys.exit("py7zr is required: python3 -m pip install py7zr")
out = os.path.join(pdk, "lib")
archives = sorted(glob.glob(os.path.join(pdk, "lib7z", "*.lib.7z")))
if not archives:
    sys.exit("no .lib.7z archives found")
with py7zr.SevenZipFile(archives[0], "r") as z:
    pass  # sanity check
for f in archives:
    with py7zr.SevenZipFile(f, "r") as z:
        z.extractall(out)
print(f"extracted {len(archives)} libraries -> {out}")
PY

echo
echo "PDK ready under: $PDK"
echo "Next: bash pdk/lef2tf.sh   (needs git; builds asap7_4x.tf with lef2tf)"
