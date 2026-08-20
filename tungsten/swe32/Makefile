# Tungsten shallow-water kernel: build, run under simfabric, and cross-check against
# references/shallow-water/swe.py.
#
# flow/devenv.sh normally exports FLUTEMK/TESTMK and puts sysroot-x86/bin on PATH when you
# cd into the atg tree. This repo is not that tree, so the toolchain is located here.
# flute.mk self-locates its siblings via $(dir $(FLUTEMK)), so pointing at the installed
# copy is enough. Override ATG if the checkout moves.

ATG     ?= /net/fuxic-vm.cerebras.aws/srv/nfs/fuxic-data/ws/atg
SYSROOT ?= $(ATG)/sysroot-x86
FLUTEMK ?= $(SYSROOT)/share/build/flute.mk
TESTMK  ?= $(SYSROOT)/share/build/test.mk
TOP     ?= $(CURDIR)
export PATH := $(SYSROOT)/bin:$(PATH)

include $(FLUTEMK)
include $(TESTMK)

# MODE must be sim. exec-prism.mk clamps cycles to PRISM_CYCLES_MAX = 100000, so a long
# run under prism appears to finish and yields garbage; the check below makes that loud.
MODE    ?= sim
CYCLES  ?= 100000

# simfabric is multithreaded, so this wants to match the cores actually available.
# Hard-coding 8 is a trap: on a 2-core dev VM it oversubscribes and throughput collapses
# (356 cyc/s for the 32x32 grid, against 12 cores on a cluster node). Default to whatever
# the cgroup grants, which is what `srun -c N` sets.
THREADS ?= $(shell nproc)

# lib=.. is required: test.mk builds inside test-<label>/ with VPATH=.., so paint resolves
# code('swe) and code('moat) relative to the build dir, not the source dir.
PAINTFLAGS    = lib=microcode lib=kinf lib=.. $(PAINTOPTS) verbose arch=${ARCH}
SIMFLAGS      = cycles=$(CYCLES) threads=$(THREADS) $(SIMOPTS)
PAINTVARS     = NX NY NSTEP
PIPELINEFLAGS = arch=${ARCH}

SYMS = eta u v alpha beta_c inv1pb mask_u mask_v gdtdx gdtdy dtdx dtdy depth

all : check

# --- Initial state and constants, via miset -------------------------------------------
# One stamp rather than 13 declared outputs: init.py writes them all in one pass, and a
# pattern rule would run it 13 times under -j.
init.stamp : ../init.py ../swe_config.py
	python3 ../init.py --nx $(NX) --ny $(NY) --outdir .
	touch $@

bench.img : bench.obj init.stamp
	cp $< $@
	@set -e; for s in $(SYMS); do \
	  miset image=$@ out=$@ map=bench.map \
	        symbol=grid.$$s order="y x _word" data=init-$$s.bin; \
	done

# --- The gate -------------------------------------------------------------------------
test : validate

validate : bench-$(MODE).core
	@test "$(MODE)" = sim || { echo "MODE=$(MODE) refused: prism clamps cycles to 100000"; exit 1; }
	miget image=$< map=bench.map symbol=grid.eta order="y x _word" data=sim-eta.bin
	miget image=$< map=bench.map symbol=grid.u   order="y x _word" data=sim-u.bin
	miget image=$< map=bench.map symbol=grid.v   order="y x _word" data=sim-v.bin
	python3 ../check.py --nx $(NX) --ny $(NY) --steps $(NSTEP)

# --- Diagnostics ----------------------------------------------------------------------
# Which arithmetic sites the backend actually fused. mirror.py's FUSED table must agree.
fma : bench-listing.json
	@python3 -c "import json,re,sys; \
	    t=json.dumps(json.load(open('bench-listing.json'))); \
	    ops=re.findall(r'(fmac?s|fmss|fmuls|fadds|fsubs)[^\"]*', t); \
	    print(chr(10).join(ops) or 'no sp arithmetic found')"

# Wavelets that left the rect or were never consumed. Both should be zero.
warn : bench-$(MODE).log
	@grep -iE 'orphan|outside|copying' $< || echo "no orphan or off-fabric wavelets"

clean : cleanup
	rm -f init-*.bin sim-*.bin ref-*.bin init.stamp

.PHONY : all clean test validate fma warn
.PRECIOUS : %.img
