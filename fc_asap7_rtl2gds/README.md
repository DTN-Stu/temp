# ASAP7 + Fusion Compiler RTL-to-GDSII flow

Complete, runnable-by-hand flow for the **SNAX FP4 LLM** RTL (`RTL.zip`) on the
**ASAP7 7 nm predictive PDK** using **Synopsys Fusion Compiler**.

The machine this was authored on has no EDA tools, so nothing here has been run
through `fc_shell`. Every Tcl file passes a brace/syntax check, the Python and
shell scripts are linted, and the command syntax was cross-checked against
public ICC2/FC flow scripts (citations at the bottom). **Expect to adjust a few
release-specific options on your installation** — the risky calls are wrapped
in `catch` and called out in the troubleshooting table.

---

## 1. What is here

```
fc_asap7_rtl2gds/
├── env.sh                     # tool paths / env vars
├── config.tcl                 # all knobs (design, timing, PDK, scenarios)
├── Makefile                   # convenience driver
├── RTL.zip                    # (you place the RTL zip here)
├── pdk/
│   ├── fetch_pdk.sh           # download ASAP7 RVT LEF + NLDM libs
│   ├── lef2tf.sh              # tech LEF -> Synopsys .tf (via lef2tf)
│   ├── 10_lib2db.tcl          # .lib -> .db (lc_shell)
│   ├── 20_build_ndm.tcl       # LEF + .db -> NDM (icc2_lm_shell)
│   └── 25_techlib_parasitic.tcl  # optional tech NDM + TLU+
├── rtl/
│   └── snax_llm_top_clean_synth.sv   # small-TCDM wrapper (see §3)
├── constraints/
│   ├── snax_llm_top.sdc
│   └── mcmm_setup.tcl
├── floorplan/
│   └── estimate_floorplan.py  # Python die/core estimator -> work/floorplan.tcl
└── scripts/
    ├── 00_extract_rtl.sh
    ├── run_fc.tcl             # master driver (-from / -to)
    ├── 00_setup.tcl
    ├── 10_create_lib.tcl
    ├── 20_read_design.tcl
    ├── 30_scenarios.tcl
    ├── 40_floorplan.tcl
    ├── 50_compile_fusion.tcl
    ├── 60_clock_opt.tcl
    ├── 70_route.tcl
    ├── 80_chip_finish.tcl
    └── 90_write_outputs.tcl
```

Generated data lives under `pdk_data/` and `work/` (not committed).

---

## 2. Prerequisites

* Fusion Compiler (`fc_shell`), Library Compiler (`lc_shell`), and a Library
  Manager (`icc2_lm_shell`, or `fc_lm_shell` on newer installs).
* Python 3.8+ with `py7zr` (`python3 -m pip install py7zr`) for `.lib.7z`.
* `curl`, `git`, `unzip`.
* Licenses for FC and the relevant features.

Set the executable names if they are not on `PATH`:

```bash
source ./env.sh
export FC_SHELL=/path/to/fc_shell
export LM_SHELL=/path/to/icc2_lm_shell
export LC_SHELL=/path/to/lc_shell
```

---

## 3. About the RTL top and the TCDM

The real top is **`snax_llm_top_clean`** (21-master crossbar, 32-bank TCDM,
streamers, FP4 core, softmax, log-quantizer). Its default parameters infer:

```
NUM_BANKS x BANK_DEPTH x 72 = 32 x 1024 x 72 = 2,359,296 storage bits
```

ASAP7 ships **no SRAM macros**, so synthesising that as flops is impractical.
The flow therefore uses the wrapper `rtl/snax_llm_top_clean_synth.sv`, which
instantiates the *unmodified* `snax_llm_top_clean` with:

```
NUM_BANKS=8, BANK_DEPTH=64, NUM_LANES=16, ACC_DEPTH=16
=> 36,864 storage bits
```

Set `USE_WRAPPER 0` in `config.tcl` to synthesise the full top instead (you
must then provide SRAM macros, or accept a very large flop-based design).

Two extra defines are passed to RTL: `IGC_CELL` and `LATCH_CELL` (use the real
ASAP7 `ICGx4` / `DHLx2` cells instead of behavioural clock gates/latches) and
`ACC_DEPTH_64`.

---

## 4. Quick start

```bash
cd fc_asap7_rtl2gds
source ./env.sh

# 1) PDK: download tech LEF, RVT cell LEF, RVT NLDM libs; extract .lib.7z
bash pdk/fetch_pdk.sh            # add --all-vt / --with-qrc if needed

# 2) Technology file (.tf) from the tech LEF
bash pdk/lef2tf.sh

# 3) .lib -> .db
$LC_SHELL -f pdk/10_lib2db.tcl

# 4) Build NDM reference libraries (RVT by default)
$LM_SHELL -f pdk/20_build_ndm.tcl
#    other VTs:  NDM_VTS="R L SL" $LM_SHELL -f pdk/20_build_ndm.tcl

# 5) Extract RTL and estimate a floorplan
bash scripts/00_extract_rtl.sh
python3 floorplan/estimate_floorplan.py \
  --lef pdk_data/lef/asap7sc7p5t_27_R_4x_201211.lef \
  --rtl work/rtl --mem-bits 36864 \
  --util 0.65 --aspect 1.0 --margin 5.0 \
  --out work/floorplan.tcl

# 6) Run the flow
$FC_SHELL -f scripts/run_fc.tcl
```

Or simply:

```bash
make pdk tf db ndm rtl floorplan
make fc
```

Run a subset (uses saved checkpoints):

```bash
$FC_SHELL -f scripts/run_fc.tcl -from 50_compile_fusion -to 90_write_outputs
```

---

## 5. Stage details

| Stage | What it does | Key commands |
|---|---|---|
| `10_create_lib` | create design library from `.tf` + NDM refs; read TLU+ if present | `create_lib -technology ... -ref_libs ...`, `read_parasitic_tech` |
| `20_read_design` | parse filelist, add wrapper, read SV with defines, link | `read_verilog -sv -define ...`, `link_block` |
| `30_scenarios` | TT/FF/SS corners, `func` mode, scenarios, read SDC | `create_mode/corner/scenario`, `set_process_label`, `read_sdc` |
| `40_floorplan` | source `work/floorplan.tcl`, restore layer directions, PG | `initialize_floorplan`, `connect_pg_net`, `create_pg_*` |
| `50_compile_fusion` | physically-aware synthesis + placement | `compile_fusion` |
| `60_clock_opt` | CTS + post-CTS opt | `clock_opt -from build_clock`, `-from final_opto` |
| `70_route` | global/detail route + post-route opt | `route_auto`, `route_opt` |
| `80_chip_finish` | fillers, PG connect | `create_stdcell_fillers` |
| `90_write_outputs` | reports + netlist/DEF/SDC/SPEF/GDS | `report_*`, `write_verilog/def/sdc/parasitics/gds` |

Outputs: `work/outputs/<design>.{qor,timing,area,power,clock_qor}.rpt`,
`.v.gz`, `.def.gz`, `.sdc`, `.spef`, `.gds`.

---

## 6. Floorplan estimation

`floorplan/estimate_floorplan.py` supports three modes, in order of accuracy:

1. **`--area-rpt reports/<design>.area.rpt`** — parses `Total cell area` from a
   Fusion Compiler `report_area`. **Use this after a first synthesis run.**
2. **`--gate-count N`** — area = N × median simple-gate area from the LEF.
3. **RTL coarse model** (default) — estimates FF bits from non-blocking
   assignments, resolves `[PKT_W-1:0]`-style widths from parameter defaults,
   estimates inferred memory bits from array declarations, and adds a logic
   factor (`--logic-gates-per-ff`, default 4).

It computes `core_area = cell_area / util`, applies the aspect ratio, adds a
die-to-core margin, snaps to the `asap7sc7p5t` site (0.216 × 1.08 µm) and the
0.004 µm manufacturing grid, then emits:

```tcl
initialize_floorplan -boundary {{0 0} {W H}} -core_offset {x y} \
  -site_def asap7sc7p5t -flip_first_row false
```

The RTL model is deliberately coarse: it counts *declarations*, not instances,
so multiply-instantiated memories are undercounted. For this design the wrapper
already fixes the TCDM size, hence `--mem-bits 36864` in the Makefile. Replace
the estimate with a real `--area-rpt` number once available.

Recommended loop:

```bash
# first pass with estimate -> run compile_fusion -> re-estimate with real area
python3 floorplan/estimate_floorplan.py --area-rpt work/outputs/snax_llm_top_clean_synth.area.rpt \
  --out work/floorplan.tcl
$FC_SHELL -f scripts/run_fc.tcl -from 40_floorplan
```

---

## 7. Known gaps / caveats (read this)

* **ASAP7 is a predictive, academic PDK** — not silicon-validated, and the
  "4x" LEF is scaled to make it usable. Keep **4x LEF + 4x tech LEF + .tf from
  the 4x tech LEF** consistent.
* **No `.tf` in the PDK.** It is generated by `lef2tf` from the tech LEF. The
  converter does not emit routing directions; `40_floorplan.tcl` restores them.
* **No TLU+ / ITF.** ASAP7 provides only a Cadence **QRC techfile** (typical
  corner, 4x). Until you convert it (or use StarRC), RC is estimated. Put a
  TLU+ and layer map at `pdk_data/tluplus/typical.{tluplus,map}` to enable it.
* **Only TT/FF/SS**, no separate RC corners, so MCMM is limited to three
  process-derived scenarios.
* **No DRC/LVS/antenna signoff decks** and no `streamout.map`. `write_gds`
  needs a layer map for real stream-out; the script warns if it is missing.
* **No SRAM macros.** The wrapper shrinks the TCDM; see §3.
* **Latches.** `latch_array` uses `DHL` latches. FC supports latch-based timing,
  but verify `time.enable_latch_based_timing_analysis` (default on in modern
  releases) and inspect `check_timing`.
* **Version differences.** `read_verilog -sv/-define`, `compile_fusion -check_only`,
  `clock_opt -from/-to`, `write_gds` flags, and `check_design -checks` vary by
  release. They are wrapped in `catch`; check the log warnings and adjust.

---

## 8. Troubleshooting

| Symptom | Likely cause / fix |
|---|---|
| `create_lib` fails: missing technology | `pdk_data/asap7_4x.tf` missing → run `pdk/lef2tf.sh` |
| `read_verilog` unknown option `-sv`/`-define` | Remove them in `scripts/20_read_design.tcl`; `.sv` extension usually enables SV |
| unresolved `pe_defines.svh` | add the incdir to `search_path` (already done) or `-incdir` |
| `link_block` reports unresolved modules | check `work/rtl` extraction and the filelist |
| `create_lib` cannot find ref NDM | run `icc2_lm_shell -f pdk/20_build_ndm.tcl` |
| every timing arc is `??` | wrong NDM/timing views or mixed 1x/4x scaling |
| `initialize_floorplan` rejects `-boundary` | use `-core_utilization` fallback (script does this) |
| `clock_opt -from build_clock` fails | fallback `clock_opt` runs; adjust stage names for your release |
| `write_gds` fails | supply `pdk_data/streamout.map` with `-layer_map` |
| huge area / million flops | TCDM inferred; use the wrapper or SRAM macros |

---

## 9. References

These public scripts/docs were used to confirm command syntax:

* Fusion Compiler / ICC2 flow: `create_lib -technology -ref_libs`,
  `read_verilog -library -design -top`, `link_block`, `read_sdc`,
  `initialize_floorplan`, `place_opt`, `clock_opt`, `route_auto`, `route_opt`,
  `write_gds` — <https://github.com/plusisland/ICC2/blob/main/FC_flow.md>,
  <https://github.com/Entropy-xcy/saed_demo/blob/main/icc2.tcl>,
  <https://github.com/SkillSurf/systemverilog/blob/master/asic_flow/scripts/icc2/icc2.tcl>
* NDM creation via Library Manager (`create_workspace`, `read_lef`, `read_db`,
  `commit_workspace`) — <https://github.com/abdelazeem201/ASIC-Design-Roadmap/blob/main/IC%20Compiler%20II/NDM_Creation.tcl>,
  <https://github.com/aadabathon/SiliconBadgers/blob/main/tutorials/synopsys_debugging/saed32_icc2_kit/build_ndm.tcl>
* tech LEF → `.tf`: <https://github.com/rmbrum/lef2tf>
* ASAP7 PDK: <https://github.com/The-OpenROAD-Project/asap7sc7p5t_27>
* Fusion Compiler command overview: <https://github.com/xp4t/synopsys-skill/blob/main/synopsys-skills/skills/synopsys-fusion-compiler/SKILL.md>
