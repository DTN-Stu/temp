# =============================================================================
# 20_build_ndm.tcl - build NDM reference libraries from cell LEF + .db.
# Run with:  icc2_lm_shell -f pdk/20_build_ndm.tcl
#            (or fc_lm_shell on newer Fusion Compiler installs)
#
# One NDM is produced per VT flavour.  Each NDM merges all function groups
# (AO, INVBUF, OA, SEQ, SIMPLE) and all three process corners (TT/FF/SS).
# =============================================================================
set script_dir [file dirname [file normalize [info script]]]
set root [expr {[info exists ::env(PROJ_ROOT)] && $::env(PROJ_ROOT) ne "" \
                ? $::env(PROJ_ROOT) : [file normalize [file join $script_dir ..]]}]
set pdk  [expr {[info exists ::env(PDK_DIR)] && $::env(PDK_DIR) ne "" \
                ? $::env(PDK_DIR) : [file join $root pdk_data]}]

set lefdir $pdk/lef
set dbdir  $pdk/db
set ndmdir $pdk/ndm
set tf     $pdk/asap7_4x.tf

if {![file exists $tf]} {
  error "Missing $tf - run pdk/lef2tf.sh first"
}
file mkdir $ndmdir

# Build RVT by default.  Use:  NDM_VTS="R L SL" icc2_lm_shell -f ...
set vts {R}
if {[info exists ::env(NDM_VTS)] && $::env(NDM_VTS) ne ""} {
  set vts $::env(NDM_VTS)
}

set funcs   {AO INVBUF OA SEQ SIMPLE}
set corners {TT FF SS}

foreach vt $vts {
  # LEF uses R/L/SL/SRAM, the timing libs use RVT/LVT/SLVT/SRAM.
  switch -- $vt {
    R  { set vtl RVT }
    L  { set vtl LVT }
    SL { set vtl SLVT }
    default { set vtl $vt }
  }

  set lef [file join $lefdir asap7sc7p5t_27_${vt}_4x_201211.lef]
  if {![file exists $lef]} {
    puts "WARN: missing cell LEF $lef (skipping VT $vt)"
    continue
  }

  set vt_lc [string tolower $vtl]
  set out   [file join $ndmdir asap7_${vt_lc}.ndm]
  puts "============================================================"
  puts "Building NDM: $out"

  create_workspace ${vt_lc}_ws -technology $tf -flow normal
  read_lef -include cell $lef

  set ndb 0
  foreach f $funcs {
    foreach c $corners {
      set db [file join $dbdir asap7sc7p5t_${f}_${vtl}_${c}_nldm_201020.db]
      if {![file exists $db]} {
        puts "WARN: missing $db"
        continue
      }
      read_db $db -process_label [string tolower $c]
      incr ndb
    }
  }
  if {$ndb == 0} {
    puts "ERROR: no .db read for VT $vt - aborting this workspace"
    catch { remove_workspace }
    continue
  }

  # Default placement site comes from the cell LEF SITE definition.
  if {![catch { get_site_defs -quiet * } sites] && [llength $sites] > 0} {
    set sname [lindex [get_object_name $sites] 0]
    catch { set_attribute [get_site_defs $sname] is_default true }
    catch { set_attribute [get_site_defs $sname] symmetry Y }
  }

  check_workspace -allow_missing
  commit_workspace -force -output $out
  catch { remove_workspace }
  puts "DONE: $out ($ndb timing views)"
}

exit
