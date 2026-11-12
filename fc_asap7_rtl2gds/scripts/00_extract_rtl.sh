#!/usr/bin/env bash
# Extract RTL.zip into work/rtl (keeping the vendor directory layout).
set -euo pipefail
ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
WORK="${WORK_DIR:-$ROOT/work}"

ZIP=""
for cand in "$ROOT/RTL.zip" "$ROOT/../RTL.zip" "$ROOT"/../RTL*.zip "$ROOT"/RTL*.zip; do
  if [[ -f "$cand" ]]; then ZIP="$cand"; break; fi
done
[[ -n "$ZIP" ]] || { echo "RTL.zip not found next to the project or one level up" >&2; exit 1; }

mkdir -p "$WORK/rtl"
unzip -o "$ZIP" -d "$WORK/rtl" >/dev/null
echo "RTL extracted to $WORK/rtl"
find "$WORK/rtl" -maxdepth 2 -type f | sed 's/^/  /' | head -40
