# =============================================================================
# mcmm_setup.tcl - create modes / corners / scenarios and load the SDC.
# Sourced by scripts/30_scenarios.tcl.
#
# Corner process labels must match the -process_label used in pdk/20_build_ndm.tcl
# (tt / ff / ss).
# =============================================================================
if {[llength [info commands remove_scenarios]]} { catch { remove_scenarios -all } }
catch { remove_corners -all }
catch { remove_modes   -all }

catch { create_mode func }

foreach spec $::SCENARIOS {
  lassign $spec sname label volt temp

  catch { create_corner $sname }
  catch { create_scenario -mode func -corner $sname -name $sname }
  current_scenario $sname

  catch { set_process_label $label }
  catch { set_temperature $temp }

  # Apply voltage only if supply nets exist (they do not exist before PG).
  if {![catch { get_supply_nets VDD } vdd_nets] && [llength $vdd_nets] > 0} {
    catch { set_voltage $volt -object_list $vdd_nets }
    catch { set_voltage 0.0 -object_list [get_supply_nets VSS] }
  }

  if {[file exists $::TLUPLUS_FILE] && [file exists $::LAYERMAP_FILE]} {
    catch { set_parasitic_parameters -early_spec typical -late_spec typical }
  }

  read_sdc $::SDC_FILE
}

foreach_in_collection sc [get_scenarios] {
  catch {
    set_scenario_status $sc -active true -setup true -hold true \
      -max_transition true -max_capacitance true \
      -leakage_power true -dynamic_power true
  }
}

puts "INFO: scenarios: [get_object_name [get_scenarios]]"
