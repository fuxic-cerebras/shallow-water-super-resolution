import numpy as np

x = np.fromfile("in-x.bin", "<f4")
names = ["r_max", "r_min", "r_split", "r_where"]
o = {n: np.fromfile(f"out-{n}.bin", "<f4") for n in names}
A, B = np.float32(10.0), np.float32(20.0)
print(f"{'x':>12} {'max':>10} {'min':>10} {'split':>12} {'where':>12} {'want':>12}")
bad_split = bad_where = 0
for k, v in enumerate(x):
    want = np.float32(v * (A if v > 0 else B))
    s, w = o["r_split"][k], o["r_where"][k]
    bad_split += s.view(np.uint32) != want.view(np.uint32)
    bad_where += w.view(np.uint32) != want.view(np.uint32)
    print(
        f"{v:>12.3e} {o['r_max'][k]:>10.3e} {o['r_min'][k]:>10.3e} "
        f"{s:>12.5e} {w:>12.5e} {want:>12.5e}"
        f"  {'split*' if s != want else ''}{'where*' if w != want else ''}"
    )
print(f"\n  split (branch-free): {'OK' if bad_split == 0 else f'WRONG on {bad_split}/8'}")
print(f"  where (predicated) : {'OK' if bad_where == 0 else f'WRONG on {bad_where}/8'}")
