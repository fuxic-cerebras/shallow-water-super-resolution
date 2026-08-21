# Tungsten shallow-water kernel

A Cerebras Tungsten (`.w`) kernel that integrates the 2D shallow-water solver from
`references/shallow-water/swe.py` on an `NX x NY` grid, one PE per grid cell, and a
two-tier cross-check against the reference script itself.

At the target size, 32x32, the kernel is **bit-exact** against a float32 mirror of its own
arithmetic and agrees with `swe.py`'s float64 result to 1.7e-7 in `eta`.

## Configuration

`swe.py`'s own settings, evaluated at `N_x = N_y = NX`. This is deliberately **not** the
dataset's coarse leg: the committed `data/raw/.../*.h5` files override `dt` with the
128-grid CFL bound (25.1398 s, D003) and use a multi-bump registry initial condition. Here
the oracle is the reference script, so the grid picks its own `dt` and uses the script's
single fixed Gaussian.

| Quantity | Value at NX=32 |
|---|---|
| `L_x`, `L_y` | 1.0e6 m |
| `g`, `H` | 9.81, 100.0 |
| `f_0`, `beta` | 1.0e-4, 2.0e-11 |
| coriolis, beta-plane | on, on |
| friction, wind, source, sink | all off |
| `dx = dy = L/(NX-1)` | 32258.064516129034 m |
| `dt = 0.1 min(dx,dy)/sqrt(gH)` | 102.9920736796937 s |
| initial condition | the single Gaussian at `swe.py:160` |

Grid size and step count come from `test.rc`, and every derived constant is recomputed from
them, so smaller rungs are self-consistent physics rather than a truncated 32x32.

## Measured results

Every rung is **bit-exact** against the float32 mirror on all three fields (`eta`, `u`, `v`).
The last column is the max absolute difference from `swe.py`'s float64 result, against a
gate of `atol=1e-5` / `rtol=1e-4`.

| Rung | Grid | Steps | tier 1 | max abs error vs swe.py |
|---|---|---:|---|---|
| `smoke` | 4x4 | 1 | exact | 2.3e-11 |
| `two` | 4x4 | 2 | exact | 4.1e-10 |
| `odd` | 5x5 | 3 | exact | 1.9e-09 |
| `rect` | 6x4 | 3 | exact | 3.9e-09 |
| `n8` | 8x8 | 50 | exact | 9.8e-08 |
| `n32` | 32x32 | 50 | exact | 1.7e-07 |
| `full` | 32x32 | 4999 | exact | 2.1e-06 |

The ladder is deliberate: `rect` is non-square so an `i`/`j` transposition cannot hide, `odd`
breaks accidental symmetries, and 4x4 is small enough that every cell is a wall or a corner.

`full` is `swe.py`'s own step count -- 4999 updates, which at this grid's `dt` of 102.992 s
is 143.0 h of physical time. It stays bit-exact against the mirror for the whole trajectory,
and the field ranges agree with `swe.py` to five decimals:

| field | kernel | swe.py |
|---|---|---|
| `eta` | -0.182386 .. 0.262958 | -0.182385 .. 0.262958 |
| `u` | -0.029847 .. 0.045172 | -0.029847 .. 0.045172 |
| `v` | -0.038941 .. 0.039174 | -0.038941 .. 0.039174 |

`sum(eta)` drifts 2.1e-06 over 4999 steps, 1.4e-07 relative -- consistent with float32
roundoff on a telescoping flux sum rather than a leaking wall. Minimum total depth `H+eta`
stays at 99.82 m, the east wall's `u` and the north wall's `v` are exactly `+0.0`, and
nothing goes non-finite.

Cost at 32x32 (1156 tiles), all with `threads` matching available cores:

| Rung | Cycles | Cycles/step | simfabric | Wall |
|---|---:|---:|---|---|
| `n32` (50 steps) | 17,831 | 356 | 356 cyc/s, 2 cores | 59 s |
| `full` (4999 steps) | 1,774,726 | 355 | 1575 cyc/s, 12 cores | 19 min |

Both report **all router state clean and zero orphan wavelets**. The 4.4x throughput gain is
purely the core count -- see the `cbrun` note above.

## Files

| File | Role |
|---|---|
| `swe_config.py` | Single source of truth for every constant, float64 and its float32 cast |
| `init.py` | Writes the initial state and constants as `miset`-loadable binaries |
| `swe.w` | The interior cell kernel |
| `moat.w` | The closed-basin wall; tutorial 27's zero-replying moat |
| `bench.paint` | Layout and colorpair routing; tutorial 27's, unchanged |
| `mirror.py` | Op-for-op float32 mirror of `swe.w` -- the exact oracle |
| `check.py` | Both comparison tiers plus the mass invariant; the gate |
| `trajectory.py` | Records a whole trajectory by checkpoint-restart, for the animations |
| `Makefile`, `test.rc` | Build, run, and the validation ladder |

The pure-Python half of the check is `tests/tungsten/test_mirror_parity.py`, which needs no
toolchain and runs in the normal `pytest` suite. The minimal repro for the codegen bug
described below lives in `../probe-select/`.

It is a **sibling** directory, not a subdirectory, and that placement is load-bearing:
`test.mk` computes `SUBDIRS := $(dir $(wildcard */Makefile))` and treats any directory
containing a child `Makefile` as a branch to recurse into rather than a leaf to test. A
`probe-select/` nested inside this directory silently turns `make check` here into a
recursion that never runs `test.rc` at all -- and because the previous run's `make-log` is
still on disk, it looks like it worked.

## Running

`flow/devenv.sh` is not loaded in this repo, so the Makefile locates the toolchain itself;
override `ATG` if the checkout moves.

```bash
make TEST=smoke MODE=sim     # 4x4, 1 step   -- fastest end-to-end check
make TEST=n32   MODE=sim     # 32x32, 50 steps -- the correctness claim
make TEST=full  MODE=sim CYCLES=3000000   # 32x32, 4999 steps -- swe.py's own count
make check TEST='*'          # every rung
make fma                     # which arithmetic sites the backend fused
make warn                    # orphan or off-fabric wavelets
```

`MODE=sim` is required and enforced: `exec-prism.mk` clamps cycles to 100000, so a long run
under prism would appear to succeed and return garbage. `CYCLES` can be set generously --
simfabric halts on quiescence ("No data movement in the last 50 cycles"), so an oversized
budget costs nothing.

### Use a cluster allocation for the long rungs

simfabric is multithreaded and the 32x32 grid is 1156 tiles, so the run is CPU-bound and
scales with cores. `THREADS` defaults to `nproc`, which is right under an `srun` cgroup but
means a 2-core dev VM gives you 2 threads. Send the long rungs to a cluster node:

```bash
cbrun -t rocky -- srun -c 12 make TEST=full MODE=sim CYCLES=3000000 THREADS=12
```

For reference, the 32x32 50-step rung managed 356 cycles/s with 8 threads on a contended
2-core VM -- which puts the 4999-step trajectory at over an hour. Ask for the cores.

### Animating the trajectory

`check.py` compares one *final* state, because that is all `miget` can see: `swe.w` writes
`eta`, `u` and `v` back after its `j in [0, NSTEP)` loop and the core dump holds only that.
`trajectory.py` gets the intermediate states by paying for them once -- it paints at
`NSTEP = --stride` and then `miset`s each run's output back in as the next run's initial
condition:

```bash
cbrun -t rocky -- srun -c 12 python3 trajectory.py --frames 100 --stride 50 --threads 12
python ../../scripts/visualize_tungsten.py --all      # from the repo root
```

The restart is exact rather than an approximation, and cheap: the scheme is a one-step
recurrence whose entire state is `(eta, u, v)`, every constant is recomputed by `init.py`
at load time, and the probe taps are write-only -- so `--frames F --stride S` costs the same
`F*S` kernel steps one uninterrupted run of `F*S` steps would. It is also *checked* rather
than argued: `mirror.py` is advanced continuously, one step at a time, and every frame is
compared bit-for-bit, so a restart that dropped or repeated a step would fail at frame 1.

At the same 32x32 horizon as the `full` rung, **all 101 frames are bit-exact** against the
mirror -- `max|kernel - mirror| = 0` in `eta`, `u` and `v`, against a float32 ULP of 1.5e-08
at this field magnitude -- and every frame reports clean router state and zero orphans.

| Frames | Stride | Steps | Horizon | Wall (12 cores) | Result |
|---:|---:|---:|---:|---:|---|
| 100 | 50 | 5000 | 143.0 h | 22.8 min, 13.7 s/frame | 101/101 exact |

Painting happens once, so the per-frame overhead is only `miset`/`mielf`/`elfmi`/`miget`
around a 61 MB core dump. It is small: the last frame's log reports simfabric itself at
11.05 s for 17,831 cycles (1613 cyc/s, the same rate the `n32` rung measured), against 13.7 s
of wall clock -- about 2.2 s of image plumbing per frame. End to end that is 22.8 min against
the single `full` run's 19 min, a 20% surcharge for 100 usable states instead of one.

`scripts/visualize_tungsten.py` renders them through the reference repo's own `viz_tools.py`.
Its difference panel is scaled to one float32 ULP rather than autoscaled -- with a bit-exact
kernel the data are identically zero, and an autoscale would turn roundoff-free agreement
into a full-contrast image of nothing.

## Design

### Layout

One `rect(0 0 NX+2 NY+2)` carries uniform colorpair routing for four sockets; a 1-wide moat
ring implements the walls; the `NX x NY` interior runs `swe.w`. Fabric x is `swe.py`'s first
index `i`, fabric y its second index `j`. At 32x32 that is 34x34 = 1156 tiles.

### Two exchange phases per step

Phase 1 sends `eta` to all four neighbours. Phase 2 sends the new `u` horizontally and the
new `v` vertically. Both are single `parallel` blocks of 8 operations; splitting sends from
receives into separate serial blocks deadlocks the whole grid.

Phase 2's leftward and downward sends look redundant and are not, and neither are the
receives of the `u_E`/`v_N` values the scheme never uses. Both were checked by ablation at
4x4, and both fail the same way -- **orphan wavelets and wrong answers spread across the
whole grid**, not the clean deadlock you might expect:

| Variant | Result | Orphan wavelets | max abs error in `eta` |
|---|---|---:|---|
| unmodified | PASS | 0 | exact |
| drop the `b[]`/`d[]` sends | FAIL | yes | 7.6e-04, all four walls plus interior |
| drop the `u_E`/`v_N` receives | FAIL | 129 reports | 1.6e-04, all four walls plus interior |

The mechanism: the moat is purely reactive, so with no leftward send the west moat never
fires and the `x=0` column never gets its `u_W`. Dropping a receive instead leaves a wavelet
in the queue, so the next step's `eta_E` reads it and every socket is permanently one
wavelet out of phase. Because simfabric stops on quiescence rather than hanging, both show
up as a completed run with corrupted output -- which is why `check.py` gates on values and
`make warn` reports orphan wavelets.

### Boundaries

The moat replies `0:sp` to everything, and two per-PE masks zero `u` on the east wall and
`v` on the north wall. That single reply value covers all four walls:

- **west**: `u_W = 0`, so the west flux term vanishes -- `swe.py`'s one-sided
  `uhwe[0,:] = u_np1[0,:]*h_e[0,:]`. The `eta_W` reply is never read, because the upwind
  test on `u_W` cannot select it.
- **east**: the PE zeroes its own `u`, so the moat's `eta_E` only ever reaches a face height
  that is then multiplied by that zero.
- **south** / **north**: the same in `v`.

Corner moat tiles receive nothing and sit idle: only cardinal sends exist.

### No divides, no transcendentals

Tungsten lowers `/` to a Newton-Raphson reciprocal rather than an IEEE divide, and has no
`exp` or `sqrt` at all. So `1/(1+beta_c)`, `dt/dx`, `dt/dy` and the Gaussian are evaluated
on the host and loaded with `miset`. The kernel is `+`, `-`, `*`, `min`, `max` only.

Constants are loaded rather than written as `:sp` literals. Tutorials 27 and 28 inline their
physics and mirror it by hand in `ref.py` with nothing checking that the two copies agree;
here every constant depends on the grid size anyway, and loading them means the kernel and
`mirror.py` cannot disagree about a bit pattern.

## Two findings worth knowing

### Predicated `sp` selects miscompile on `arch=sdr`

A bare `if` (no `else`) that conditionally overwrites a variable **silently does nothing**
when that variable's current value is a plain **copy** of another variable:

```tungsten
sp hn;
hn <- eNc;                            // plain copy
if (vn2 > 0.0:sp) { hn <- ec; }       // never takes effect
```

Register allocation coalesces the destination with the copy source, so the predicated
select is emitted with identical source registers and becomes a no-op:

```
flteqs P0 = D0, 0x4;   P0? select32 D0 = D0, D0, P0;
```

The compare is correct; the select is degenerate. No diagnostic, no crash -- just the
fall-through value every time. Here that meant every upwind cell taking the downwind
height, presenting as a 7e-9 discrepancy in `eta` confined to two walls.

The trigger is the coalescible copy, not the operand kinds and not the comparison. A
literal or an expression default occupies its own register and works; `>`, `<` with swapped
operands, `>=`, and comparing against a variable zero all fail identically. This is also
why the same construct was correct for the wall masks, where `un2` holds an expression
result, and wrong for the face heights, where `hn` was a copy. `../probe-select/` measures
the full matrix.

`if/else` and the ternary `?:` both work and are the straightforward fixes. This kernel
instead drops predication entirely, using the branch-free flux split

```
u * h_upwind  ==  max(u,0) * h_here + min(u,0) * h_downwind
```

which is algebraically identical to `swe.py`'s `where(u > 0, ...)` -- one term is always a
zero product -- and reproduces the strict `> 0` test, since at `u == 0` both terms vanish.
The reason for going further than `if/else` is that the trigger depends on a
register-allocation decision rather than anything visible in the source: the same statement
is correct or incorrect depending on how the value it overwrites happened to be produced.

### Fused multiply-add, and the sign of zero

The backend forms `fmss` from `a - b*c` and rounds **once**, and no pragma disables it
(`math.fast`/`math.ehalf` only *lower* precision). A binary32 FMA evaluated in binary64 is
exact, so `mirror.py` reproduces it as `f32(f64(a) - f64(b)*f64(c))`.

Which sites actually fuse is **read off `bench-listing.json`**, not assumed -- and the
answer is much less than the `a +/- b*c` shape suggests: of eleven candidate sites only the
two corrector subtractions become `fmss`. `mirror.py`'s `FUSED` table records this; `make fma`
regenerates the evidence. Tier 2's tolerance holds under either fusion model, so the
physical claim does not depend on this; tier 1's bit-exactness does.

Separately, masking a wall velocity with a bare `* mask` yields `-0.0` for a negative input,
and that sign survives into the next step's `- alpha*u` wherever the other operand is also
zero -- at 32x32 that is 312 of 1024 cells at step 1. Hence `+ 0.0`.

Subnormals are **preserved**, not flushed: the kernel returns values like `1.4e-44`. Nothing
programs `FP_CTL`, so this was an open question until measured.

## The cross-check

Bit-exact agreement with `swe.py` is impossible in principle -- the kernel is float32, the
script float64, and the kernel multiplies by precomputed reciprocals where the script
divides. So there are two tiers:

- **tier 1**, kernel vs `mirror.py`: float32, op for op, expected **exact**. This is what
  localizes bugs; `check.py` reports which wall a mismatch sits on.
- **tier 2**, kernel vs `swe.py` via `tests/solver/reference_harness.py`: `atol=1e-5`,
  `rtol=1e-4`. Those come from the measured float32-vs-float64 envelope, not a guess.

Plus a reference-free invariant: the scheme is in flux-divergence form with zero flux
through every wall, and PE `i`'s east face flux is the same expression on bit-identical
operands as PE `i+1`'s west face flux, so the sum telescopes and `sum(eta)` is conserved to
roundoff. A broken wall shows up there with no oracle at all.

`swe.py`'s loop is `while time_step < max_time_step` starting at 1, so it performs
`max_time_step - 1` updates; the harness is called with `NSTEP + 1`.
