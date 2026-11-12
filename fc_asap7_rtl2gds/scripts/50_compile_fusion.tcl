# =============================================================================
# 50_compile_fusion.tcl - physically aware synthesis + placement.
# =============================================================================
# Optional pre-flight check; ignored on releases that do not support it.
catch { compile_fusion -check_only }

puts "INFO: running compile_fusion"
compile_fusion

save_block
save_lib -all
puts "INFO: compile_fusion complete"
