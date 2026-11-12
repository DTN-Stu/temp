# =============================================================================
# snax_llm_top.sdc - timing constraints (SDC / Tcl)
#
# Defaults are provided so the file is self-contained; config.tcl variables
# override them when present.
# =============================================================================
if {![info exists CLK_PERIOD]}       { set CLK_PERIOD 2.0 }
if {![info exists CLK_UNC_SETUP]}    { set CLK_UNC_SETUP 0.05 }
if {![info exists CLK_UNC_HOLD]}     { set CLK_UNC_HOLD 0.02 }
if {![info exists CLK_TRANS]}        { set CLK_TRANS 0.05 }
if {![info exists INPUT_DELAY_PCT]}  { set INPUT_DELAY_PCT 0.30 }
if {![info exists OUTPUT_DELAY_PCT]} { set OUTPUT_DELAY_PCT 0.30 }
if {![info exists DRIVE_CELL]}       { set DRIVE_CELL BUFx2_ASAP7_75t_R }
if {![info exists LOAD_FF]}          { set LOAD_FF 5.0 }
if {![info exists MAX_TRANS]}        { set MAX_TRANS 0.10 }
if {![info exists MAX_CAP_FF]}       { set MAX_CAP_FF 20.0 }

# --- Clock -------------------------------------------------------------------
create_clock -name clk -period $CLK_PERIOD [get_ports clk]
set_clock_uncertainty -setup $CLK_UNC_SETUP [get_clocks clk]
set_clock_uncertainty -hold  $CLK_UNC_HOLD  [get_clocks clk]
set_clock_transition  $CLK_TRANS [get_clocks clk]

# --- Reset -------------------------------------------------------------------
# Asynchronous, active-low reset: not a timed data path.
set_false_path -from [get_ports rst_n]

# --- I/O ---------------------------------------------------------------------
set _in_ports  [remove_from_collection [all_inputs] [get_ports [list clk rst_n]]]
set _out_ports [all_outputs]

set_input_delay  -clock clk -max [expr {$CLK_PERIOD * $INPUT_DELAY_PCT}]  $_in_ports
set_input_delay  -clock clk -min [expr {$CLK_PERIOD * 0.05}]              $_in_ports
set_output_delay -clock clk -max [expr {$CLK_PERIOD * $OUTPUT_DELAY_PCT}] $_out_ports
set_output_delay -clock clk -min [expr {-1.0 * $CLK_PERIOD * 0.05}]       $_out_ports

if {[llength $_in_ports] > 0} {
  catch { set_driving_cell -lib_cell $DRIVE_CELL $_in_ports }
  set_load [expr {$LOAD_FF / 1000.0}] [all_outputs]
}

# --- Design rules ------------------------------------------------------------
catch { set_max_transition  $MAX_TRANS [current_block] }
catch { set_max_capacitance [expr {$MAX_CAP_FF / 1000.0}] [current_block] }

# --- Clock gating ------------------------------------------------------------
# The latch_array uses ICG cells; give CTS a gating check margin.
catch { set_clock_gating_check -setup $CLK_UNC_SETUP -hold $CLK_UNC_HOLD [get_clocks clk] }

# --- Path groups (helps reporting/optimization priorities) --------------------
catch { group_path -name REG2REG -from [all_clocks] -to [all_clocks] }
catch { group_path -name REG2OUT -from [all_clocks] -to [all_outputs] }
catch { group_path -name IN2REG -from [all_inputs] -to [all_clocks] }
