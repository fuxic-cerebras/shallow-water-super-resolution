"""Is ConvMixer's validation gap a BatchNorm running-statistics artifact?

Same checkpoint, same data, same code path -- the only thing that varies is whether BatchNorm
uses its running statistics (eval) or the batch's own (train). If eval is much worse, the gap
is the normalization layer and not the model's fit.
"""

import sys
from pathlib import Path

import numpy as np  # noqa: F401  # must precede torch; see swe_sr/__init__
import torch
import yaml

import swe_sr  # noqa: F401  # numpy-before-torch load-order guard; see swe_sr/__init__
from swe_sr.data.dataset import PairedSnapshotDataset
from swe_sr.data.manifest import load_manifest
from swe_sr.data.normalization import Normalization
from swe_sr.data.processing import AugmentationPolicy
from swe_sr.data.storage import resolve_array_dir
from swe_sr.models import build_model_from_config
from swe_sr.train import REPO_ROOT, _batches, _subset, autocast_context, per_channel_mse
from swe_sr.training.config import TrainingConfig

RUN = Path(sys.argv[1])
raw = yaml.safe_load((RUN / "config.yaml").read_text())
fields = TrainingConfig.__dataclass_fields__
config = TrainingConfig(**{k: v for k, v in raw.items() if k in fields})

name, model = build_model_from_config(REPO_ROOT / config.model_config)
model.load_state_dict(torch.load(RUN / "checkpoints" / "best.pt", weights_only=True))

manifest = load_manifest(REPO_ROOT / config.manifest)
norm = Normalization.from_dict(manifest.normalization)
array_dir = resolve_array_dir(REPO_ROOT / config.manifest, manifest.dataset_id)
ds = PairedSnapshotDataset(
    manifest,
    array_dir,
    split="validation",
    normalization=norm,
    augmentation=AugmentationPolicy(),
    seed=config.seed,
)
idx = _subset(ds, config.max_validation_trajectories, config.max_frames_per_trajectory)
print(f"model={name}  validation samples={len(idx)}  batch_size={config.batch_size}")


@torch.no_grad()
def score(mode: str) -> float:
    getattr(model, mode)()
    total = torch.zeros(3, dtype=torch.float64)
    count = 0
    batches = _batches(ds, idx, config.batch_size, shuffle=False, seed=config.seed, epoch=0)
    for coarse, fine in batches:
        with autocast_context(config):
            prediction = model(coarse)
        total += per_channel_mse(prediction.float(), fine).double() * coarse.shape[0]
        count += coarse.shape[0]
    return float((total / count).mean())


evaluated = score("eval")
trained = score("train")
print(f"\n  eval()  running statistics : {evaluated:.4f}   <-- what the curve reports")
print(f"  train() batch statistics   : {trained:.4f}")
print(f"  ratio                      : {evaluated / trained:.2f}x")
print("\n  EDSR pilot  best_val       : 0.3069")
print("  U-Net pilot best_val       : 0.3028")

bns = [m for m in model.modules() if isinstance(m, torch.nn.BatchNorm2d)]
rm = torch.cat([b.running_mean.flatten() for b in bns])
rv = torch.cat([b.running_var.flatten() for b in bns])
print(f"\nBatchNorm state across {len(bns)} layers:")
print(f"  running_mean  absmax {rm.abs().max():.4g}")
print(f"  running_var   min {rv.min():.4g}   max {rv.max():.4g}")
print(f"  num_batches_tracked {bns[0].num_batches_tracked.item()}   momentum {bns[0].momentum}")

# Per-layer variance drift: a body whose activations blow up with depth is the mechanism that
# would make a lagging running estimate catastrophic rather than merely imprecise.
print("\n  per-layer running_var max, first/last 5 layers:")
per = [float(b.running_var.max()) for b in bns]
head = " ".join(f"{v:8.3g}" for v in per[:5])
tail = " ".join(f"{v:8.3g}" for v in per[-5:])
print(f"    {head}  ...  {tail}")
