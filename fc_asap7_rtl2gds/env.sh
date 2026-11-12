#!/usr/bin/env bash
# Source this file before running anything:
#   source ./env.sh
#
# Override any of these on your EDA machine if the executables are not on PATH.

export PROJ_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
export PDK_DIR="${PDK_DIR:-$PROJ_ROOT/pdk_data}"
export WORK_DIR="${WORK_DIR:-$PROJ_ROOT/work}"

# Synopsys executables (override if your site uses wrappers / different names)
export FC_SHELL="${FC_SHELL:-fc_shell}"
export LM_SHELL="${LM_SHELL:-icc2_lm_shell}"   # Fusion Compiler installs may ship fc_lm_shell
export LC_SHELL="${LC_SHELL:-lc_shell}"

# Python used for .7z extraction and the floorplan estimator
export PYTHON="${PYTHON:-python3}"

# Library / design names (must match config.tcl)
export DESIGN="${DESIGN:-snax_llm_top_synth}"
export LIB_NAME="${LIB_NAME:-snax_llm_top}"
