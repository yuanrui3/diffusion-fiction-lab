"""Means and bootstrap intervals that respect the grouping of samples by item.

The 8 samples drawn for one item are not independent of each other, so every
interval here resamples *items*, never single samples.

A metric is given as {item_id: [values]}, where a value may be None (undefined
for that sample, for example parallelism when the lengths differ). The estimate
is the pooled mean over defined values: sum of values over count of values.
"""

from __future__ import annotations

from typing import Mapping, Optional, Sequence

import numpy as np

ByItem = Mapping[str, Sequence[Optional[float]]]

N_BOOT = 4000
SEED = 20261119


def _sums_counts(by_item: ByItem, ids: Optional[Sequence[str]] = None) -> tuple[np.ndarray, np.ndarray]:
    ids = list(by_item) if ids is None else list(ids)
    sums = np.zeros(len(ids))
    counts = np.zeros(len(ids))
    for k, item_id in enumerate(ids):
        vals = [float(v) for v in by_item.get(item_id, ()) if v is not None]
        sums[k] = sum(vals)
        counts[k] = len(vals)
    return sums, counts


def _ratio(sums: np.ndarray, counts: np.ndarray) -> np.ndarray:
    with np.errstate(invalid="ignore", divide="ignore"):
        return sums / counts


def _boot(sums: np.ndarray, counts: np.ndarray, rng: np.random.Generator, n_boot: int) -> np.ndarray:
    n = len(sums)
    idx = rng.integers(0, n, size=(n_boot, n))
    return _ratio(sums[idx].sum(axis=1), counts[idx].sum(axis=1))


def _interval(draws: np.ndarray) -> tuple[Optional[float], Optional[float]]:
    draws = draws[~np.isnan(draws)]
    if len(draws) == 0:
        return None, None
    lo, hi = np.percentile(draws, [2.5, 97.5])
    return float(lo), float(hi)


def mean_ci(by_item: ByItem, *, n_boot: int = N_BOOT, seed: int = SEED) -> dict:
    """Pooled mean with a 95% bootstrap interval over items."""
    sums, counts = _sums_counts(by_item)
    if counts.sum() == 0:
        return {"mean": None, "lo": None, "hi": None, "n_items": int((counts > 0).sum())}
    rng = np.random.default_rng(seed)
    lo, hi = _interval(_boot(sums, counts, rng, n_boot))
    return {"mean": float(sums.sum() / counts.sum()), "lo": lo, "hi": hi, "n_items": int((counts > 0).sum())}


def diff_independent(a: ByItem, b: ByItem, *, n_boot: int = N_BOOT, seed: int = SEED) -> dict:
    """mean(a) - mean(b) for two different sets of items (e.g. unseen minus recall).

    Each set is resampled on its own.
    """
    sa, ca = _sums_counts(a)
    sb, cb = _sums_counts(b)
    if ca.sum() == 0 or cb.sum() == 0:
        return {"diff": None, "lo": None, "hi": None}
    rng = np.random.default_rng(seed)
    draws = _boot(sa, ca, rng, n_boot) - _boot(sb, cb, rng, n_boot)
    lo, hi = _interval(draws)
    return {"diff": float(sa.sum() / ca.sum() - sb.sum() / cb.sum()), "lo": lo, "hi": hi}


def diff_paired(a: ByItem, b: ByItem, *, n_boot: int = N_BOOT, seed: int = SEED) -> dict:
    """mean(a) - mean(b) for two models on the same items.

    Only items present in both are used, and both are resampled with the same
    draw of items, which removes the item-to-item variation the models share.
    """
    ids = [i for i in a if i in b]
    sa, ca = _sums_counts(a, ids)
    sb, cb = _sums_counts(b, ids)
    if not ids or ca.sum() == 0 or cb.sum() == 0:
        return {"diff": None, "lo": None, "hi": None, "n_items": len(ids)}
    rng = np.random.default_rng(seed)
    idx = rng.integers(0, len(ids), size=(n_boot, len(ids)))
    draws = _ratio(sa[idx].sum(axis=1), ca[idx].sum(axis=1)) - _ratio(sb[idx].sum(axis=1), cb[idx].sum(axis=1))
    lo, hi = _interval(draws)
    return {"diff": float(sa.sum() / ca.sum() - sb.sum() / cb.sum()), "lo": lo, "hi": hi, "n_items": len(ids)}


def diff_of_diffs(a1: ByItem, b1: ByItem, a2: ByItem, b2: ByItem, *, n_boot: int = N_BOOT, seed: int = SEED) -> dict:
    """(a1 - b1) - (a2 - b2): a paired model difference on one set of items
    minus the same paired difference on another set.

    Used for "does the diffusion-minus-autoregressive difference change between
    the unseen set and the recall set". Items are resampled within each set,
    the two models jointly.
    """
    rng = np.random.default_rng(seed)
    parts = []
    points = []
    for a, b in ((a1, b1), (a2, b2)):
        ids = [i for i in a if i in b]
        sa, ca = _sums_counts(a, ids)
        sb, cb = _sums_counts(b, ids)
        if not ids or ca.sum() == 0 or cb.sum() == 0:
            return {"diff": None, "lo": None, "hi": None}
        idx = rng.integers(0, len(ids), size=(n_boot, len(ids)))
        parts.append(_ratio(sa[idx].sum(axis=1), ca[idx].sum(axis=1)) - _ratio(sb[idx].sum(axis=1), cb[idx].sum(axis=1)))
        points.append(sa.sum() / ca.sum() - sb.sum() / cb.sum())
    lo, hi = _interval(parts[0] - parts[1])
    return {"diff": float(points[0] - points[1]), "lo": lo, "hi": hi}


def by_item(rows: Sequence[dict], key: str) -> dict[str, list[Optional[float]]]:
    """Collect one metric from scored rows as {item_id: [values]}."""
    out: dict[str, list[Optional[float]]] = {}
    for r in rows:
        v = r.get(key)
        out.setdefault(r["item_id"], []).append(None if v is None else float(v))
    return out
