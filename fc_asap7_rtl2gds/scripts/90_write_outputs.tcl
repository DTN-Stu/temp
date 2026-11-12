# =============================================================================
# 90_write_outputs.tcl - reports and handoff artifacts (netlist, DEF, SDC,
# SPEF, GDS).
# =============================================================================
file mkdir $::OUT_DIR
set base $::OUT_DIR/$::DESIGN

catch { report_qor > ${base}.qor.rpt }
catch { report_timing -max_paths 200 -nworst 10 > ${base}.timing.rpt }
catch { report_area > ${base}.area.rpt }
catch { report_power > ${base}.power.rpt }
catch { report_clock_qor > ${base}.clock_qor.rpt }
catch { report_congestion > ${base}.congestion.rpt }

catch { write_verilog -compress gzip ${base}.v.gz }
catch { write_sdc ${base}.sdc }
catch { write_def -compress gzip ${base}.def.gz }
catch { write_parasitics -compress -output ${base} }

set gds_ok 0
if {[catch { write_gds -hierarchy all ${base}.gds } msg]} {
  puts "WARN: write_gds without layer map failed: $msg"
  set map $::PDK_DIR/streamout.map
  if {[file exists $map]} {
    if {![catch { write_gds -hierarchy all -layer_map $map ${base}.gds }]} {
      set gds_ok 1
    }
  }
  if {!$gds_ok} {
    puts "WARN: GDS not written. Provide a layer map at $map"
  }
}

save_block
save_lib -all

puts "============================================================"
puts "Outputs in: $::OUT_DIR"
catch { puts [exec ls -l $::OUT_DIR] }
puts "============================================================"
