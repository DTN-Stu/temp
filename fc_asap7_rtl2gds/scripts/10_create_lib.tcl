# =============================================================================
# 10_create_lib.tcl - create the Fusion design library from the technology
# source and the NDM reference libraries.
# =============================================================================
set lib_path $::WORK_DIR/$::LIB_NAME.nlib

if {[file exists $lib_path]} { file delete -force $lib_path }

foreach ref $::REF_LIBS {
  if {![file exists $ref]} {
    error "Missing reference NDM: $ref  (run pdk/20_build_ndm.tcl with the Library Manager)"
  }
}
if {![file exists $::TECH_SOURCE]} {
  error "Missing technology source: $::TECH_SOURCE  (run pdk/lef2tf.sh)"
}

puts "INFO: create_lib $lib_path  (tech=$::TECH_SOURCE, refs=$::REF_LIBS)"
create_lib -technology $::TECH_SOURCE -ref_libs $::REF_LIBS $lib_path

if {[file exists $::TLUPLUS_FILE] && [file exists $::LAYERMAP_FILE]} {
  puts "INFO: reading parasitic tech 'typical'"
  read_parasitic_tech -tlup $::TLUPLUS_FILE -layermap $::LAYERMAP_FILE -name typical
} else {
  puts "WARN: TLU+ / layer map not found; extraction will be estimate-based."
  puts "WARN:   expected: $::TLUPLUS_FILE"
  puts "WARN:   expected: $::LAYERMAP_FILE"
}
