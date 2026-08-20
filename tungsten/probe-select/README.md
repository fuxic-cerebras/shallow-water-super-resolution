# Minimal repro: predicated `sp` select miscompiles on `arch=sdr`

Evidence for the branch-free flux split in `../swe32/swe.w`. Keeping it runnable matters
because that design choice rests on this defect being real.

## The bug

A bare `if` (no `else`) that conditionally overwrites a variable **silently does nothing**
when that variable's current value is a plain **copy** of another variable:

```tungsten
sp h;
h <- bv;                        // plain copy
if (v > 0.0:sp) { h <- av; }    // never takes effect
```

Register allocation coalesces the destination with the copy source, so the emitted
predicated select ends up with identical source registers and becomes a no-op:

```
flteqs P0 = D0, 0x4;   P0? select32 D0 = D0, D0, P0;
```

The compare is emitted correctly. It is the select that is degenerate, which is why the
failure is silent: no diagnostic, no crash, just the fall-through value every time.

## What triggers it, measured

`make TEST=probe MODE=sim` checks all six forms against the expected result. Condition
values span `-2.7e-4 .. 100` including both signed zeros; `av`/`bv` are loaded via `miset`
so nothing can be constant-folded.

| Form | Result |
|---|---|
| default = **copy**, branch = literal | **wrong** on every input where the condition holds |
| default = **copy**, branch = variable | **wrong** on every input where the condition holds |
| default = literal, branch = variable | ok |
| default = expression (`bv + zero`), branch = literal | ok |
| `if/else`, both variables | ok |
| ternary `?:`, both variables | ok |

So it is not about the operand kinds as such, and not about the comparison: `>`, `<` with
swapped operands, `>=`, comparing against a variable zero, and negating the value all fail
identically. What matters is whether the fall-through value is a coalescible copy. A
literal or an expression result occupies its own register and the select behaves; adding
`+ 0.0` to the default is enough to make the same code correct.

## Why it cost real time

In `../swe32/swe.w` the same construct appeared twice. The wall mask,

```tungsten
un2 <- (un - bc * uu + al * vv) * ib;   // expression -> own register
if (mu < 0.5:sp) { un2 <- 0.0:sp; }     // worked
```

was fine, because `un2` held an expression result. The upwind face heights,

```tungsten
hn <- eNc;                              // plain copy -> coalesced
if (vn2 > 0.0:sp) { hn <- ec; }         // silently skipped
```

were not. The result was a 7e-9 discrepancy in `eta` confined to two walls -- small enough
to look like a rounding question rather than a wrong branch.

## Workarounds

`if/else` and the ternary both work and are the obvious fixes. `../swe32/swe.w` instead
drops predication altogether, using `max(u,0)*h_here + min(u,0)*h_downwind`, because the
trigger depends on a register-allocation decision rather than on anything visible in the
source -- the same statement is correct or incorrect depending on how the value it
overwrites happened to be produced.

## Run

```bash
make TEST=probe MODE=sim
```

The test PASSes as long as the observed behaviour still matches the table above, and says
so explicitly if it has changed -- so if a compiler fix lands, this reports it.
