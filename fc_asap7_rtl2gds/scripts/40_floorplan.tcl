# =============================================================================
# 40_floorplan.tcl - initialize the floorplan, layer directions and power grid.
# =============================================================================
# Preferred source: floorplan.tcl emitted by floorplan/estimate_floorplan.py
if {[file exists $::FP_TCL]} {
  puts "INFO: sourcing estimated floorplan: $::FP_TCL"
  source $::FP_TCL
} else {
  puts "WARN: $::FP_TCL not found; using utilization-based floorplan"
  initialize_floorplan -core_utilization $::FP_UTIL \
    -core_offset $::FP_MARGIN -site_def $::FP_SITE -flip_first_row false
}

# -----------------------------------------------------------------------------
# Routing layer directions.  lef2tf.py does not emit DIRECTION into the .tf, so
# restore them from the tech LEF here (M1 vertical, M2 horizontal, ...).
# -----------------------------------------------------------------------------
foreach {lyr dir} {
  M1 vertical   M2 horizontal M3 vertical   M4 horizontal
  M5 vertical   M6 horizontal M7 vertical   M8 horizontal
  M9 vertical
} {
  catch { set_attribute [get_layers $lyr] routing_direction $dir }
}

# Tracks are normally created by initialize_floorplan; this is a best-effort
# fallback for releases that need an explicit call.
catch { create_tracks -auto }

# -----------------------------------------------------------------------------
# Power / ground
# -----------------------------------------------------------------------------
catch { connect_pg_net -automatic }

# Standard-cell rails (VDD/VSS pins are on M1 in the ASAP7 cells).
if {[catch {
  create_pg_std_cell_conn_pattern std_conn
  set_pg_strategy std_strat -pattern {{name: std_conn} {nets: {VDD VSS}}}
  compile_pg -strategies std_strat
} msg]} {
  puts "WARN: standard-cell PG generation skipped: $msg"
}

# A simple core ring on the top two routing layers.
if {[catch {
  create_pg_ring_pattern core_ring_pat \
    -horizontal_layer M8 -horizontal_width 0.5 -horizontal_spacing 0.5 \
    -vertical_layer   M9 -vertical_width   0.5 -vertical_spacing   0.5
  set_pg_strategy core_ring \
    -pattern {{name: core_ring_pat} {nets: {VDD VSS}} {offset: {0.5 0.5}}} -core
  compile_pg -strategies core_ring
} msg]} {
  puts "WARN: core ring generation skipped: $msg"
}

catch { connect_pg_net -automatic }

save_block
save_lib -all
puts "INFO: floorplan complete"
