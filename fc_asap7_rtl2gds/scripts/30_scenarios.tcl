# =============================================================================
# 30_scenarios.tcl - MCMM modes/corners/scenarios and SDC loading.
# =============================================================================
if {![file exists $::SDC_FILE]} { error "SDC not found: $::SDC_FILE" }
if {![file exists $::PROJ_ROOT/constraints/mcmm_setup.tcl]} {
  error "mcmm_setup.tcl not found"
}

# Export the config variables needed by the sourced files.
set CLK_PERIOD       $::CLK_PERIOD
set CLK_UNC_SETUP    $::CLK_UNC_SETUP
set CLK_UNC_HOLD     $::CLK_UNC_HOLD
set CLK_TRANS        $::CLK_TRANS
set INPUT_DELAY_PCT  $::INPUT_DELAY_PCT
set OUTPUT_DELAY_PCT $::OUTPUT_DELAY_PCT
set DRIVE_CELL       $::DRIVE_CELL
set LOAD_FF          $::LOAD_FF
set MAX_TRANS        $::MAX_TRANS
set MAX_CAP_FF       $::MAX_CAP_FF

source $::PROJ_ROOT/constraints/mcmm_setup.tcl

catch { check_timing }
puts "INFO: scenario setup complete"
