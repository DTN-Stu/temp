#!/usr/bin/env python3
"""Coarse floorplan estimator for the ASAP7 / Fusion Compiler flow.

It does NOT synthesise.  It produces a defensible first-pass die/core size from
either

  1. an exact post-synthesis area report  (--area-rpt), or
  2. an explicit gate count               (--gate-count), or
  3. a coarse RTL model                   (default):
        - flip-flop bits are estimated from non-blocking assignments,
        - inferred memory bits are estimated from array declarations,
        - logic area = logic_gates_per_ff * FF bits * median gate area,
     with cell areas read from the ASAP7 LEF.

The output is a Tcl snippet consumed by scripts/40_floorplan.tcl:

    initialize_floorplan -boundary {{0 0} {W H}} -core_offset {x y} ...

Always replace the estimate with a real report_area number once you have run
compile_fusion (use --area-rpt reports/<design>.area.rpt).
"""

import argparse
import glob
import json
import math
import os
import re
import statistics
import sys

GRID = 0.004          # manufacturing grid (um) from the ASAP7 tech LEF
SITE_W = 0.216        # asap7sc7p5t site width  (um)
SITE_H = 1.08         # asap7sc7p5t site height (um)

SIMPLE_PAT = re.compile(
    r"(INV|BUF|NAND|NOR|AND|OR|XOR|XNOR|AOI|OAI|MAJ|HA|FA|TIE|DECAP)",
    re.IGNORECASE,
)
DFF_PAT = re.compile(r"DFF", re.IGNORECASE)


# --------------------------------------------------------------------------- #
# LEF parsing
# --------------------------------------------------------------------------- #
def parse_lef_areas(path):
    """Return {macro_name: area_um2} for every MACRO with a SIZE statement."""
    areas = {}
    name, size = None, None
    with open(path, errors="ignore") as fh:
        for raw in fh:
            s = raw.strip()
            if s.startswith("MACRO "):
                parts = s.split()
                name = parts[1] if len(parts) > 1 else None
                size = None
                continue
            if name is None:
                continue
            if size is None:
                m = re.match(r"SIZE\s+([0-9.]+)\s+BY\s+([0-9.]+)", s)
                if m:
                    size = float(m.group(1)) * float(m.group(2))
            if s.startswith("END "):
                tok = s.split()
                if len(tok) > 1 and tok[1] == name:
                    if size:
                        areas[name] = size
                    name, size = None, None
    return areas


def median_area(areas, pattern):
    vals = [a for n, a in areas.items() if pattern.search(n)]
    return statistics.median(vals) if vals else None


# --------------------------------------------------------------------------- #
# RTL coarse model
# --------------------------------------------------------------------------- #
DECL_RE = re.compile(
    r"\b(?:logic|reg|bit)\b\s*(?:signed\s*)?"
    r"((?:\[[^\]]*\]\s*)*)"          # packed dimensions
    r"\b([A-Za-z_]\w*)\b\s*"
    r"(\[[^\]]*\])?"                 # optional unpacked (array) dimension
)
NBA_RE = re.compile(r"\b([A-Za-z_]\w*)\b\s*(?:\[[^\]]*\])?\s*<=")
PARAM_RE = re.compile(
    r"\b(?:parameter|localparam)\b[^;=\n]*?\b([A-Za-z_]\w*)\s*=\s*([0-9]+)\b"
)


def eval_dim_expr(expr, symbols):
    """Evaluate a constant dimension expression, substituting known params."""
    expr = str(expr).strip()
    if not expr:
        return None

    def repl(m):
        n = m.group(1)
        return str(symbols[n]) if n in symbols else n

    expr = re.sub(r"\b([A-Za-z_]\w*)\b", repl, expr)
    if not re.fullmatch(r"[0-9\s+\-*/()]+", expr):
        return None
    try:
        return int(eval(expr, {"__builtins__": {}}, {}))  # noqa: S307 - sanitized
    except Exception:
        return None


def packed_width(dims, symbols):
    if not dims or not dims.strip():
        return 1
    width = 1
    for d in re.findall(r"\[([^\]]*)\]", dims):
        d = d.strip()
        if ":" in d:
            hi, lo = d.split(":", 1)
            h, l = eval_dim_expr(hi, symbols), eval_dim_expr(lo, symbols)
            if h is None or l is None:
                return None
            width *= abs(h - l) + 1
        else:
            v = eval_dim_expr(d, symbols)
            if v is None:
                return None
            width *= v
    return width


def unpacked_depth(dim, symbols):
    if not dim:
        return 1
    d = dim.strip()[1:-1].strip()
    if ":" in d:
        hi, lo = d.split(":", 1)
        h, l = eval_dim_expr(hi, symbols), eval_dim_expr(lo, symbols)
        if h is None or l is None:
            return None
        return abs(h - l) + 1
    return eval_dim_expr(d, symbols)


def rtl_scan(rtl_dir):
    files = []
    for pat in ("**/*.sv", "**/*.v", "**/*.svh"):
        files.extend(glob.glob(os.path.join(rtl_dir, pat), recursive=True))
    files = [f for f in files if "/sim/" not in f.replace("\\", "/")]

    texts = {}
    for path in files:
        with open(path, errors="ignore") as fh:
            texts[path] = fh.read()

    # Build a coarse parameter symbol table (last definition wins).  This
    # resolves most widths such as [PKT_W-1:0] and [NUM_LANES-1:0].
    symbols = {}
    for text in texts.values():
        for m in PARAM_RE.finditer(text):
            symbols[m.group(1)] = int(m.group(2))

    decl_width, decl_depth, symbolic = {}, {}, set()
    nba_names = set()
    always_ff = 0

    for path, text in texts.items():
        always_ff += len(re.findall(r"\balways_ff\b", text))
        for m in DECL_RE.finditer(text):
            packed, name, unpacked = m.group(1), m.group(2), m.group(3)
            w = packed_width(packed, symbols)
            depth = unpacked_depth(unpacked, symbols)
            if w is None or depth is None:
                symbolic.add(name)
                continue
            decl_width[name] = w
            decl_depth[name] = depth
        for m in NBA_RE.finditer(text):
            nba_names.add(m.group(1))

    ff_bits = 0
    mem_bits = 0
    for name in nba_names:
        if name in symbolic or name not in decl_width:
            continue
        w, depth = decl_width[name], decl_depth.get(name, 1)
        if depth > 1:
            mem_bits += w * depth
        else:
            ff_bits += w

    # Symbolic arrays assigned in an always block are the most likely large
    # memories; report them so the user can pass --mem-bits explicitly.
    unknown_mem = {n: decl_width.get(n, "?")
                   for n in symbolic if n in nba_names}

    return {
        "files": len(files),
        "ff_bits": ff_bits,
        "mem_bits": mem_bits,
        "unknown_arrays": unknown_mem,
        "nba_names": len(nba_names),
        "always_ff": always_ff,
        "symbols": len(symbols),
    }


def parse_area_report(path):
    with open(path, errors="ignore") as fh:
        text = fh.read()
    for pat in (
        r"Total\s+cell\s+area\s*[:=]?\s*([0-9]+\.?[0-9]*)",
        r"Cell\s+area\s*[:=]?\s*([0-9]+\.?[0-9]*)",
        r"Total\s+area\s*[:=]?\s*([0-9]+\.?[0-9]*)",
    ):
        m = re.search(pat, text, re.IGNORECASE)
        if m:
            return float(m.group(1))
    raise ValueError(f"could not find 'Total cell area' in {path}")


# --------------------------------------------------------------------------- #
# Geometry
# --------------------------------------------------------------------------- #
def snap_up(value, step):
    return math.ceil(value / step - 1e-9) * step


def compute_die(total_area, util, aspect, margin, site_w, site_h, grid):
    core_area = max(total_area / max(util, 1e-3), 1.0)
    w = math.sqrt(core_area * aspect)
    h = math.sqrt(core_area / aspect)

    die_w = snap_up(w + 2.0 * margin, site_w)
    die_h = snap_up(h + 2.0 * margin, site_h)
    die_w = snap_up(die_w, grid)
    die_h = snap_up(die_h, grid)

    off = snap_up(margin, grid)
    if 2.0 * off >= min(die_w, die_h):
        off = 0.0

    core_w = max(die_w - 2.0 * off, site_w)
    core_h = max(die_h - 2.0 * off, site_h)
    rows = int(core_h // site_h)
    return {
        "core_area_um2": core_area,
        "die_w": round(die_w, 3),
        "die_h": round(die_h, 3),
        "core_w": round(core_w, 3),
        "core_h": round(core_h, 3),
        "core_offset": round(off, 3),
        "die_area_mm2": round(die_w * die_h / 1e6, 5),
        "rows": rows,
    }


def emit_tcl(path, geom, site_name, total_area, util):
    core_x1 = geom["core_offset"]
    core_y1 = geom["core_offset"]
    core_x2 = round(geom["die_w"] - geom["core_offset"], 3)
    core_y2 = round(geom["die_h"] - geom["core_offset"], 3)
    lines = [
        "# AUTO-GENERATED by floorplan/estimate_floorplan.py - re-run after synthesis.",
        f"# total cell area = {total_area:.3f} um^2, target utilization = {util}",
        f"# die = {geom['die_w']} x {geom['die_h']} um "
        f"({geom['die_area_mm2']} mm^2), {geom['rows']} site rows",
        f"set fp_die_w {geom['die_w']}",
        f"set fp_die_h {geom['die_h']}",
        f"set fp_core_offset [list {geom['core_offset']} {geom['core_offset']}]",
        "initialize_floorplan \\",
        "  -boundary [list [list 0 0] [list $fp_die_w $fp_die_h]] \\",
        f"  -core_offset $fp_core_offset -site_def {site_name} -flip_first_row false",
        "",
    ]
    with open(path, "w") as fh:
        fh.write("\n".join(lines))


def main():
    here = os.path.dirname(os.path.abspath(__file__))
    proj = os.path.dirname(here)

    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--lef", default=os.path.join(
        proj, "pdk_data/lef/asap7sc7p5t_27_R_4x_201211.lef"))
    ap.add_argument("--rtl", default=os.path.join(proj, "work/rtl"))
    ap.add_argument("--area-rpt", help="Fusion Compiler report_area output (most accurate)")
    ap.add_argument("--gate-count", type=float, help="override gate count")
    ap.add_argument("--ff-bits", type=float, help="override estimated FF bits")
    ap.add_argument("--mem-bits", type=float, help="override inferred memory bits")
    ap.add_argument("--exclude-mem", action="store_true",
                    help="ignore inferred memory in the area model")
    ap.add_argument("--logic-gates-per-ff", type=float, default=4.0,
                    help="logic gates per FF bit (default 4)")
    ap.add_argument("--mem-area-per-bit", type=float, default=None,
                    help="area per inferred memory bit (default: DFF area)")
    ap.add_argument("--util", type=float, default=0.65)
    ap.add_argument("--aspect", type=float, default=1.0)
    ap.add_argument("--margin", type=float, default=5.0,
                    help="die-to-core margin in um")
    ap.add_argument("--site-w", type=float, default=SITE_W)
    ap.add_argument("--site-h", type=float, default=SITE_H)
    ap.add_argument("--grid", type=float, default=GRID)
    ap.add_argument("--site-name", default="asap7sc7p5t")
    ap.add_argument("--out", default=os.path.join(proj, "work/floorplan.tcl"))
    ap.add_argument("--json", default=None)
    args = ap.parse_args()

    if not os.path.exists(args.lef):
        sys.exit(f"LEF not found: {args.lef} (run pdk/fetch_pdk.sh)")

    areas = parse_lef_areas(args.lef)
    gate_area = median_area(areas, SIMPLE_PAT)
    dff_area = median_area(areas, DFF_PAT)
    if gate_area is None:
        gate_area = 0.5
    if dff_area is None:
        dff_area = 0.3

    model = {}
    if args.area_rpt:
        total_area = parse_area_report(args.area_rpt)
        source = f"area report {args.area_rpt}"
        model = {"mode": "area_report", "total_area_um2": total_area}
    elif args.gate_count:
        total_area = args.gate_count * gate_area
        source = f"gate count {args.gate_count:g}"
        model = {"mode": "gate_count", "gate_count": args.gate_count,
                 "gate_area_um2": gate_area, "total_area_um2": total_area}
    else:
        scan = rtl_scan(args.rtl) if os.path.isdir(args.rtl) else {
            "files": 0, "ff_bits": 0, "mem_bits": 0,
            "unknown_arrays": {}, "nba_names": 0, "always_ff": 0,
        }
        ff_bits = args.ff_bits if args.ff_bits is not None else scan["ff_bits"]
        mem_bits = args.mem_bits if args.mem_bits is not None else scan["mem_bits"]
        mem_area_bit = args.mem_area_per_bit if args.mem_area_per_bit is not None else dff_area

        ff_area = ff_bits * dff_area
        logic_area = ff_bits * args.logic_gates_per_ff * gate_area
        mem_area = 0.0 if args.exclude_mem else mem_bits * mem_area_bit
        total_area = ff_area + logic_area + mem_area
        source = "RTL coarse model"
        model = {
            "mode": "rtl_model",
            "rtl_files": scan["files"],
            "always_ff_blocks": scan["always_ff"],
            "nba_names": scan["nba_names"],
            "ff_bits": ff_bits,
            "mem_bits": mem_bits,
            "unknown_arrays": scan["unknown_arrays"],
            "gate_area_um2": gate_area,
            "dff_area_um2": dff_area,
            "ff_area_um2": ff_area,
            "logic_area_um2": logic_area,
            "mem_area_um2": mem_area,
            "total_area_um2": total_area,
        }

    if total_area <= 0:
        sys.exit("estimated area is zero; pass --gate-count or --area-rpt")

    geom = compute_die(total_area, args.util, args.aspect, args.margin,
                       args.site_w, args.site_h, args.grid)

    os.makedirs(os.path.dirname(args.out), exist_ok=True)
    emit_tcl(args.out, geom, args.site_name, total_area, args.util)

    summary = {
        "source": source,
        "units": "um / um^2",
        "lef": args.lef,
        "model": model,
        "floorplan": geom,
        "tcl": args.out,
    }

    if args.json:
        with open(args.json, "w") as fh:
            json.dump(summary, fh, indent=2)

    print("=" * 68)
    print("ASAP7 floorplan estimate")
    print("=" * 68)
    print(f"  source            : {source}")
    if model.get("mode") == "rtl_model":
        print(f"  RTL files         : {model['rtl_files']}")
        print(f"  always_ff blocks  : {model['always_ff_blocks']}")
        print(f"  estimated FF bits : {model['ff_bits']:.0f}")
        print(f"  estimated mem bits: {model['mem_bits']:.0f} "
              f"({'included' if not args.exclude_mem else 'excluded'})")
        if model["unknown_arrays"]:
            print(f"  symbolic arrays   : {model['unknown_arrays']}")
        print(f"  median gate area  : {model['gate_area_um2']:.4f} um^2")
        print(f"  median DFF area   : {model['dff_area_um2']:.4f} um^2")
    print(f"  total cell area   : {total_area:.2f} um^2 "
          f"({total_area / 1e6:.4f} mm^2)")
    print(f"  die               : {geom['die_w']} x {geom['die_h']} um "
          f"({geom['die_area_mm2']} mm^2)")
    print(f"  core              : {geom['core_w']} x {geom['core_h']} um")
    print(f"  site rows         : {geom['rows']}")
    print(f"  floorplan tcl     : {args.out}")
    print("=" * 68)

    if model.get("mode") == "rtl_model" and model["mem_bits"] > 2.0e6:
        print("WARNING: >2 Mbit of inferred storage.  That is the full 32x1024x72")
        print("         TCDM treated as flops.  Use the small-TCDM wrapper")
        print("         (default) or pass --mem-bits/--exclude-mem.")
    if geom["die_area_mm2"] > 500:
        print("WARNING: estimated die > 500 mm^2 - almost certainly an")
        print("         inferred-memory artefact.  Re-run with a real")
        print("         --area-rpt after synthesis.")


if __name__ == "__main__":
    main()
