import numpy as np

x = np.fromfile("in-x.bin", "<f4")
a = np.fromfile("in-a.bin", "<f4")
b = np.fromfile("in-b.bin", "<f4")
one, two = np.float32(1.0), np.float32(2.0)

CASES = (
    ("k_copy_lit", one, b, "default=copy       branch=literal ", "WRONG"),
    ("k_copy_var", a, b, "default=copy       branch=variable", "WRONG"),
    ("k_lit_var", a, np.full_like(b, two), "default=literal    branch=variable", "ok"),
    ("k_expr_lit", one, b, "default=expression branch=literal ", "ok"),
    ("k_ifelse", a, b, "if/else            both variables ", "ok"),
    ("k_ternary", a, b, "ternary ?:         both variables ", "ok"),
)

print(f"  {'form':36s} {'expected':9s} {'actual':9s}")
failures = 0
for key, true_branch, false_branch, label, expected in CASES:
    got = np.fromfile(f"out-{key}.bin", "<f4")
    want = np.where(x > 0, true_branch, false_branch)
    bad = int((got != want).sum())
    actual = "ok" if bad == 0 else f"WRONG {bad}/8"
    agrees = (expected == "ok") == (bad == 0)
    failures += not agrees
    print(f"  {label:36s} {expected:9s} {actual:9s}{'' if agrees else '   <-- changed!'}")

print()
if failures:
    print(f"  {failures} case(s) no longer match the documented behaviour.")
else:
    print("  Behaviour matches the characterization in README.md.")
