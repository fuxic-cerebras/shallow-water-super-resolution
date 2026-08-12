"""Per-channel normalization fitted on the training split only (D-04).

`CLAUDE.md` is explicit: fit statistics on the training split only, persist them in the
manifest, and reuse the same values for validation, test, and inference. `docs/DATASET.md`
adds that statistics come from fine-grid training states, apply to both members of a pair,
and are never shared across pair IDs.

Two implementation choices worth stating because they are not obvious:

- Statistics are fitted on **destaggered** fine fields, not raw staggered ones. Models
  consume the processed cell-centered representation (D011), and destaggering is an
  averaging operator that reduces velocity variance, so fitting on raw arrays would leave
  the model's inputs systematically mis-scaled.
- Accumulation is streaming, in float64, over counts / sums / sums-of-squares. That keeps
  memory flat over a multi-GiB release, and those three accumulators are exactly what
  `docs/DATASET.md` requires the manifest to record, so the statistics can be independently
  recomputed and checked rather than taken on trust.
"""

from __future__ import annotations

import json
from collections.abc import Iterable, Mapping
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import numpy as np

from swe_sr.data.processing import destagger

# docs/DATASET.md: q' = (q - mu) / max(sigma, 1e-8)
SIGMA_FLOOR = 1e-8
CHANNELS = ("eta", "u", "v")


@dataclass(frozen=True)
class ChannelStatistics:
    """Streaming accumulators and the statistics derived from them, for one channel."""

    count: int
    total: float
    total_squared: float

    @property
    def mean(self) -> float:
        if self.count == 0:
            raise ValueError("cannot take a mean over zero samples")
        return self.total / self.count

    @property
    def variance(self) -> float:
        if self.count == 0:
            raise ValueError("cannot take a variance over zero samples")
        # Population variance. Clamped at zero because E[x^2] - E[x]^2 can go slightly
        # negative through cancellation when a channel is very nearly constant.
        return max(self.total_squared / self.count - self.mean**2, 0.0)

    @property
    def std(self) -> float:
        return float(np.sqrt(self.variance))

    @property
    def scale(self) -> float:
        """The divisor actually used, with the documented floor applied."""
        return max(self.std, SIGMA_FLOOR)

    def to_dict(self) -> dict[str, Any]:
        return {
            "count": self.count,
            "sum": self.total,
            "sum_squared": self.total_squared,
            "mean": self.mean,
            "std": self.std,
            "scale": self.scale,
        }

    @classmethod
    def from_dict(cls, payload: Mapping[str, Any]) -> ChannelStatistics:
        return cls(
            count=int(payload["count"]),
            total=float(payload["sum"]),
            total_squared=float(payload["sum_squared"]),
        )


@dataclass(frozen=True)
class Normalization:
    """Per-channel statistics for one pair ID, plus the provenance needed to audit them."""

    pair_id: str
    split: str
    source: str
    channels: tuple[ChannelStatistics, ...]

    def __post_init__(self) -> None:
        if len(self.channels) != 3:
            raise ValueError(f"expected statistics for 3 channels, got {len(self.channels)}")

    @property
    def mean(self) -> np.ndarray:
        return np.asarray([c.mean for c in self.channels], dtype=np.float64)

    @property
    def scale(self) -> np.ndarray:
        return np.asarray([c.scale for c in self.channels], dtype=np.float64)

    def apply(self, fields: np.ndarray) -> np.ndarray:
        """Normalize `[..., channel, y, x]` fields. Broadcasting keeps this shape-generic,
        so the same statistics apply to coarse and fine members alike."""
        mean = self.mean.reshape(-1, 1, 1)
        scale = self.scale.reshape(-1, 1, 1)
        result: np.ndarray = (np.asarray(fields, dtype=np.float64) - mean) / scale
        return result

    def invert(self, normalized: np.ndarray) -> np.ndarray:
        """Recover physical units. Physical diagnostics are computed after de-normalization
        (`docs/ARCHITECTURE.md`), so this is on the reporting path, not just a convenience."""
        mean = self.mean.reshape(-1, 1, 1)
        scale = self.scale.reshape(-1, 1, 1)
        result: np.ndarray = np.asarray(normalized, dtype=np.float64) * scale + mean
        return result

    def to_dict(self) -> dict[str, Any]:
        return {
            "pair_id": self.pair_id,
            "fitted_on_split": self.split,
            "fitted_on": self.source,
            "sigma_floor": SIGMA_FLOOR,
            "channel_order": list(CHANNELS),
            "channels": {
                name: stats.to_dict() for name, stats in zip(CHANNELS, self.channels, strict=True)
            },
        }

    @classmethod
    def from_dict(cls, payload: Mapping[str, Any]) -> Normalization:
        channels = payload["channels"]
        return cls(
            pair_id=str(payload["pair_id"]),
            split=str(payload["fitted_on_split"]),
            source=str(payload["fitted_on"]),
            channels=tuple(ChannelStatistics.from_dict(channels[name]) for name in CHANNELS),
        )

    def write(self, path: Path) -> None:
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text(json.dumps(self.to_dict(), indent=2, sort_keys=True) + "\n")


class NormalizationError(ValueError):
    """Normalization was fitted or applied against the wrong data."""


class _Accumulator:
    """Streaming per-channel count / sum / sum-of-squares in float64."""

    def __init__(self) -> None:
        self._count = np.zeros(3, dtype=np.int64)
        self._total = np.zeros(3, dtype=np.float64)
        self._total_squared = np.zeros(3, dtype=np.float64)

    def update(self, fields: np.ndarray) -> None:
        values = np.asarray(fields, dtype=np.float64)
        if values.shape[-3] != 3:
            raise ValueError(f"expected 3 channels on axis -3, got shape {values.shape}")
        flat = values.reshape(-1, 3, *values.shape[-2:])
        per_channel = flat.transpose(1, 0, 2, 3).reshape(3, -1)
        self._count += per_channel.shape[1]
        self._total += per_channel.sum(axis=1)
        self._total_squared += (per_channel**2).sum(axis=1)

    def finish(self, pair_id: str, split: str, source: str) -> Normalization:
        if int(self._count[0]) == 0:
            raise NormalizationError(
                f"no {split} samples were accumulated for {pair_id}; normalization would be "
                "undefined. Check that the manifest actually contains that split."
            )
        return Normalization(
            pair_id=pair_id,
            split=split,
            source=source,
            channels=tuple(
                ChannelStatistics(
                    count=int(self._count[i]),
                    total=float(self._total[i]),
                    total_squared=float(self._total_squared[i]),
                )
                for i in range(3)
            ),
        )


def fit_normalization(
    fine_field_batches: Iterable[np.ndarray],
    *,
    pair_id: str,
    split: str = "train",
    already_destaggered: bool = False,
) -> Normalization:
    """Fit per-channel statistics from an iterable of fine-grid field arrays.

    Takes an iterable so the caller streams one trajectory at a time and never holds the
    whole release in memory. Each batch is `[time, channel, y, x]`.

    `already_destaggered` exists only for tests; production callers pass raw fields and let
    this destagger them, so the statistics necessarily match the processed representation.
    """
    accumulator = _Accumulator()
    for batch in fine_field_batches:
        accumulator.update(batch if already_destaggered else destagger(batch))
    return accumulator.finish(
        pair_id=pair_id,
        split=split,
        source="fine_grid_destaggered",
    )


def fit_from_manifest(manifest: Any, raw_dir: Path, *, split: str = "train") -> Normalization:
    """Fit normalization for a released dataset, reading only that split's trajectories.

    Deliberately reads the manifest's own split labels rather than re-deriving them, so a
    mismatch between manifest and registry surfaces as a validation failure elsewhere
    instead of being silently papered over here.
    """
    from swe_sr.data.storage import FINE_ARRAY, read_fields

    records = manifest.by_split(split)
    if not records:
        raise NormalizationError(
            f"manifest {manifest.dataset_id} has no {split} trajectories to fit on"
        )

    def batches() -> Iterable[np.ndarray]:
        for record in records:
            yield read_fields(raw_dir / record.relative_path, FINE_ARRAY)

    return fit_normalization(batches(), pair_id=manifest.pair_id, split=split)


def check_pair_id(normalization: Normalization, pair_id: str) -> None:
    """Guard against applying one pair's statistics to another (docs/DATASET.md).

    This is one of the negative tests `docs/VALIDATION.md` names, and the failure it
    prevents is quiet: the model would train happily on mis-scaled inputs.
    """
    if normalization.pair_id != pair_id:
        raise NormalizationError(
            f"normalization was fitted for pair {normalization.pair_id!r} but is being "
            f"applied to {pair_id!r}. Statistics are never shared across pair IDs."
        )
