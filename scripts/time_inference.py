"""Inference latency for every arm on one host, in one process.

`seconds_per_frame` inside each run's `evaluation_test.json` was measured whenever that run was
scored, on whatever node and under whatever load, so those values are not comparable across
runs (D021 says as much, and it is why the frozen artifacts are not re-scored to acquire a
fairer number). This measures all arms back to back instead, which is the only way the
throughput column means anything.
"""

import time
from pathlib import Path

import numpy as np  # noqa: F401  # must precede torch; see swe_sr/__init__
import torch

import swe_sr  # noqa: F401  # numpy-before-torch load-order guard
from swe_sr.models import build_model_from_config, count_parameters
from swe_sr.training.config import model_config_for_run

RUNS = {
    "unet": "20260812T235727Z_unet_e3ce47d7_da865691",
    "edsr": "20260812T230157Z_edsr_aae64836_077d6b53",
    "convmixer": "20260813T232258Z_convmixer_473b97bc_116ef1a8",
    "convmixer_nonorm": "20260814T001237Z_convmixer_nonorm_d8cbb386_116ef1a8",
}

print(f"threads={torch.get_num_threads()}")
print(f"{'model':18s} {'params':>10s} {'ms/frame':>9s}")
for label, run_id in RUNS.items():
    run_dir = Path("runs") / run_id
    _, model = build_model_from_config(Path(model_config_for_run(run_dir)))
    model.load_state_dict(torch.load(run_dir / "checkpoints" / "best.pt", weights_only=True))
    model.eval()
    x = torch.randn(1, 3, 32, 32)
    with torch.no_grad():
        for _ in range(3):
            model(x)
        start = time.perf_counter()
        for _ in range(30):
            model(x)
        elapsed = (time.perf_counter() - start) / 30
    print(f"{label:18s} {count_parameters(model):>10,} {elapsed * 1000:>9.1f}")
