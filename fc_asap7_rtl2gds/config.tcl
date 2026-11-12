# =============================================================================
# config.tcl - shared configuration for the ASAP7 + Fusion Compiler flow
# Sourced by scripts/run_fc.tcl.  Edit the values below for your site.
# =============================================================================

# Project root is the directory containing this file, so the flow can be
# launched from any working directory.
set ::PROJ_ROOT [file normalize [file dirname [file normalize [info script]]]]

# -----------------------------------------------------------------------------
# Design
# -----------------------------------------------------------------------------
set FULL_TOP        snax_llm_top_clean
set USE_WRAPPER     1                     ;# 1: small-TCDM wrapper, 0: full clean top
set DESIGN          snax_llm_top_clean_synth
set LIB_NAME        snax_llm_top
set CLK_PORT        clk
set RST_PORT        rst_n

# -----------------------------------------------------------------------------
# Timing targets
# CLK_PERIOD is in ns.  2.0 ns = 500 MHz; relax it if the first compile is
# timing-limited (this is a large datapath-heavy design on a predictive PDK).
# -----------------------------------------------------------------------------
set CLK_PERIOD       2.0
set CLK_UNC_SETUP    0.05
set CLK_UNC_HOLD     0.02
set CLK_TRANS        0.05
set INPUT_DELAY_PCT  0.30
set OUTPUT_DELAY_PCT 0.30
set DRIVE_CELL       BUFx2_ASAP7_75t_R
set LOAD_FF          5.0      ;# output load in fF
set MAX_TRANS        0.10     ;# ns
set MAX_CAP_FF       20.0     ;# fF

# -----------------------------------------------------------------------------
# PDK data (populated by pdk/fetch_pdk.sh + pdk/lef2tf.sh)
# -----------------------------------------------------------------------------
set PDK_DIR         $::PROJ_ROOT/pdk_data
set PDK_LEF_DIR     $PDK_DIR/lef
set PDK_LIB_DIR     $PDK_DIR/lib
set PDK_NDM_DIR     $PDK_DIR/ndm
set PDK_TF          $PDK_DIR/asap7_4x.tf
set PDK_TECH_NDM    $PDK_NDM_DIR/asap7_tech.ndm

# Technology source passed to `create_lib -technology`.
# Use the .tf for the first bring-up; switch to the tech NDM once you have
# built it and attached TLU+ (see pdk/25_techlib_parasitic.tcl).
set TECH_SOURCE     $PDK_TF
set REF_LIBS        [list $PDK_NDM_DIR/asap7_rvt.ndm]

# Extraction.  ASAP7 ships a single QRC techfile, not TLU+.  If you convert it
# (or use StarRC) put the TLU+ and layer map here; otherwise RC is estimated.
set TLUPLUS_FILE    $PDK_DIR/tluplus/typical.tluplus
set LAYERMAP_FILE   $PDK_DIR/tluplus/typical.map

# -----------------------------------------------------------------------------
# RTL
# -----------------------------------------------------------------------------
set WORK_DIR        $::PROJ_ROOT/work
set RTL_DIR         $WORK_DIR/rtl
set RTL_FILELIST    $RTL_DIR/filelist_rtl.f
set EXTRA_RTL       [list $::PROJ_ROOT/rtl/snax_llm_top_clean_synth.sv]
set VERILOG_DEFINES [list ACC_DEPTH_64 IGC_CELL LATCH_CELL]

# Synthesize the unmodified snax_llm_top_clean instead of the small-TCDM
# wrapper.  Requires SRAM macros (not shipped with ASAP7) or a very large die.
if {!$USE_WRAPPER} {
  set DESIGN    $FULL_TOP
  set EXTRA_RTL {}
}
set RTL_INCDIRS     [list $RTL_DIR/rtl/01_snax_llm_fp4_subsystem]

# -----------------------------------------------------------------------------
# Floorplan (used only if work/floorplan.tcl is absent)
# -----------------------------------------------------------------------------
set FP_UTIL         0.65
set FP_ASPECT       1.0
set FP_MARGIN       5.0
set FP_SITE         asap7sc7p5t
set FP_TCL          $WORK_DIR/floorplan.tcl

# -----------------------------------------------------------------------------
# MCMM scenarios: {scenario_name process_label voltage temperature}
# The process labels must match the -process_label used when building the NDM.
# -----------------------------------------------------------------------------
set SCENARIOS [list \
  [list func_tt tt 0.7 25] \
  [list func_ff ff 0.7 25] \
  [list func_ss ss 0.7 25] ]

set SDC_FILE        $::PROJ_ROOT/constraints/snax_llm_top.sdc

# -----------------------------------------------------------------------------
# Outputs / checkpoints
# -----------------------------------------------------------------------------
set OUT_DIR         $WORK_DIR/outputs
