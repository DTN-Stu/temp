# =============================================================================
# 00_setup.tcl - environment / app-option setup
# =============================================================================
if {![info exists ::PROJ_ROOT]} { error "config.tcl not sourced" }

file mkdir $::WORK_DIR
file mkdir $::OUT_DIR

set_app_options -name design.bus_delimiters -value {[]}

# SystemVerilog enablement (names differ slightly between releases).
catch { set_app_options -name hdlin.enable_sverilog -value true }
catch { set_app_options -name hdlin.verilog.enable_sverilog -value true }

# Timing / implementation defaults that are safe for an RTL bring-up.
catch { set_app_options -name time.remove_clock_reconvergence_pessimism -value true }
catch { set_app_options -name place.coarse.continue_on_missing_scandef -value true }
catch { set_app_var sh_continue_on_error false }

catch {
  set _nproc [lindex [split [exec nproc] "\n"] 0]
  set_host_options -max_cores $_nproc
}

puts "INFO: setup complete (design=$::DESIGN, lib=$::LIB_NAME)"
