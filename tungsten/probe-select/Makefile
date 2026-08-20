ATG     ?= /net/fuxic-vm.cerebras.aws/srv/nfs/fuxic-data/ws/atg
SYSROOT ?= $(ATG)/sysroot-x86
FLUTEMK ?= $(SYSROOT)/share/build/flute.mk
TESTMK  ?= $(SYSROOT)/share/build/test.mk
TOP     ?= $(CURDIR)
export PATH := $(SYSROOT)/bin:$(PATH)
include $(FLUTEMK)
include $(TESTMK)
MODE ?= sim
PAINTFLAGS = lib=microcode lib=kinf lib=.. $(PAINTOPTS) arch=${ARCH}
SIMFLAGS   = cycles=20000 $(SIMOPTS)
PAINTVARS  = N
PIPELINEFLAGS = arch=${ARCH}
all : check
bench.img : bench.obj in-x.bin
	cp $< $@
	@set -e; for s in x a b; do \
	  miset image=$@ out=$@ map=bench.map symbol=t.$$s order="n _word" data=in-$$s.bin; \
	done
in-x.bin :
	python3 ../mk.py
test : bench-$(MODE).core
	@for r in k_copy_lit k_copy_var k_lit_var k_expr_lit k_ifelse k_ternary; do \
	  miget image=$< map=bench.map symbol=t.$$r order="n _word" data=out-$$r.bin; done
	python3 ../rd.py
clean : cleanup
	rm -f in-*.bin out-*.bin
.PHONY : all clean test
.PRECIOUS : %.img
