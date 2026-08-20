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
	miset image=$@ out=$@ map=bench.map symbol=t.x order="n _word" data=in-x.bin
in-x.bin :
	python3 ../mk.py
test : bench-$(MODE).core
	@for r in r_max r_min r_split r_where; do \
	  miget image=$< map=bench.map symbol=t.$$r order="n _word" data=out-$$r.bin; done
	python3 ../rd.py
clean : cleanup
	rm -f in-*.bin out-*.bin
.PHONY : all clean test
.PRECIOUS : %.img
