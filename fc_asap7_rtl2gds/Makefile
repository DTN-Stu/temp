# Convenience driver.  Run `make pdk tf db ndm rtl floorplan` then `make fc`.
# Override tool names on the command line, e.g.:
#   make fc FC_SHELL=/path/to/fc_shell LM_SHELL=/path/to/icc2_lm_shell

PROJ     := $(CURDIR)
PDK      := $(PROJ)/pdk_data
FC_SHELL ?= fc_shell
LM_SHELL ?= icc2_lm_shell
LC_SHELL ?= lc_shell
PYTHON   ?= python3

.PHONY: all pdk tf db ndm rtl floorplan fc clean

all: ndm floorplan

pdk:
	bash pdk/fetch_pdk.sh

tf: pdk
	bash pdk/lef2tf.sh

db: pdk
	$(LC_SHELL) -f pdk/10_lib2db.tcl

ndm: tf db
	$(LM_SHELL) -f pdk/20_build_ndm.tcl

rtl:
	bash scripts/00_extract_rtl.sh

floorplan: rtl
	$(PYTHON) floorplan/estimate_floorplan.py \
	  --lef $(PDK)/lef/asap7sc7p5t_27_R_4x_201211.lef \
	  --rtl $(PROJ)/work/rtl \
	  --util 0.65 --aspect 1.0 --margin 5.0 \
	  --mem-bits 36864 \
	  --out $(PROJ)/work/floorplan.tcl

fc: ndm floorplan
	$(FC_SHELL) -f scripts/run_fc.tcl

clean:
	rm -rf $(PROJ)/work/$(LIB_NAME).nlib $(PROJ)/work/outputs
