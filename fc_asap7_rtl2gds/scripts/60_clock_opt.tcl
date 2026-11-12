# =============================================================================
# 60_clock_opt.tcl - clock tree synthesis and post-CTS optimization.
# =============================================================================
if {[catch {
  clock_opt -from build_clock -to build_clock
} msg]} {
  puts "WARN: 'clock_opt -from build_clock' failed: $msg"
  puts "WARN: falling back to plain clock_opt"
  catch { clock_opt }
}

catch { clock_opt -from final_opto -to final_opto }

save_block
save_lib -all
puts "INFO: clock_opt complete"
