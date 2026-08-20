# Minimal repro: predicated `sp` select miscompiles on `arch=sdr`

This is the evidence for the branch-free flux split in `../swe.w`. Keeping it runnable
matters because the parent README makes a strong claim about a compiler defect, and that
claim is load-bearing: it is the reason the kernel contains no conditionals.

## What it shows

A single tile, no sockets, eight `sp` inputs spanning `-2.7e-4 .. 100`, and two ways of
computing `v * (v > 0 ? A : B)`:

| Form | Result |
|---|---|
| `sp h; h <- B; if (v > 0.0:sp) { h <- A; }` then `v * h` | **wrong on exactly the 4 inputs where `v > 0`** |
| `max(v,0)*A + min(v,0)*B` | correct on 7/8; differs only at `-0.0`, where it gives `+0.0` rather than `-0.0` |

The `if` body never executes. The listing shows the tell -- a `select32` whose two source
registers are the same:

```
flteqs P0 = D0, 0x4;   P0? select32 D0 = D0, D0, P0;
```

A **literal** right-hand side (`h <- 1.0:sp`) does work, which is why a first version of the
kernel had correct wall masks and silently wrong upwind face heights.

## Run

```bash
make TEST=probe MODE=sim
```

The last two lines of output are the verdict per form. Note the test PASSes either way --
it prints a comparison rather than asserting, because the point is to observe the
difference, not to gate on it.
