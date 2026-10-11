from pathlib import Path

import numpy as np

from src.scripts.utils.classification_data import SplitData

DEFAULT_RATIOS = (0.15, 0.25, 0.5, 1.0)

def ratio_key(value: float) -> str:
    return f'{float(value):.2f}'.replace('.', 'p')

def round_half_up(total: int, fraction: float) -> int:
    return int(np.floor(float(total) * float(fraction) + 0.5))

def _objective(counts: np.ndarray, target: np.ndarray) -> float:
    scale = np.sqrt(np.maximum(target, 1.0))
    standardized = (counts.astype(np.float64) - target) / scale
    return float(np.mean(np.square(standardized)) + 0.05 * np.max(np.abs(standardized)))

def _proportional_subset(labels: np.ndarray, pool_positions: np.ndarray, target_rows: int, target_counts: np.ndarray, rng: np.random.Generator, candidates: int, refinement_steps: int) -> np.ndarray:
    if target_rows == len(pool_positions):
        return np.sort(pool_positions.copy())
    best = None
    best_score = float('inf')
    for _ in range(max(1, int(candidates))):
        candidate = rng.choice(pool_positions, size=target_rows, replace=False)
        score = _objective(labels[candidate].sum(axis=0, dtype=np.int64), target_counts)
        if score < best_score:
            best = candidate.copy()
            best_score = score
    selected = np.asarray(best, dtype=np.int64)
    selected_mask = np.zeros(len(labels), dtype=np.bool_)
    selected_mask[selected] = True
    pool_mask = np.zeros(len(labels), dtype=np.bool_)
    pool_mask[pool_positions] = True
    unselected = np.flatnonzero(pool_mask & ~selected_mask).astype(np.int64)
    counts = labels[selected].sum(axis=0, dtype=np.int64)
    for _ in range(max(0, int(refinement_steps))):
        selected_slot = int(rng.integers(len(selected)))
        unselected_slot = int(rng.integers(len(unselected)))
        row_out = int(selected[selected_slot])
        row_in = int(unselected[unselected_slot])
        proposal = counts - labels[row_out].astype(np.int64) + labels[row_in].astype(np.int64)
        proposal_score = _objective(proposal, target_counts)
        if proposal_score + 1e-12 < best_score:
            selected[selected_slot] = row_in
            unselected[unselected_slot] = row_out
            counts = proposal
            best_score = proposal_score
    return np.sort(selected)

def build_nested_schedule(labels, physical_rows, ratios=DEFAULT_RATIOS, seed: int=2026, candidates: int=64, refinement_steps: int=4000):
    labels = np.asarray(labels, dtype=np.uint8)
    physical_rows = np.asarray(physical_rows, dtype=np.int64)
    ratios = tuple(sorted({float(value) for value in ratios}))
    rng = np.random.default_rng(int(seed))
    full_counts = labels.sum(axis=0, dtype=np.int64)
    positions = np.arange(len(labels), dtype=np.int64)
    selected = {1.0: positions.copy()}
    pool = positions
    for ratio in sorted((value for value in ratios if value < 1.0), reverse=True):
        target_counts = np.asarray([round_half_up(int(value), ratio) for value in full_counts], dtype=np.int64)
        pool = _proportional_subset(labels, pool, round_half_up(len(labels), ratio), target_counts, rng, candidates, refinement_steps)
        selected[ratio] = pool.copy()
    return {ratio_key(ratio): physical_rows[selected[ratio]] for ratio in ratios}

class IndexedConcatenatedArray:

    def __init__(self, arrays, row_mappings):
        self._arrays = tuple(arrays)
        self._rows = tuple((np.asarray(rows, dtype=np.int64) for rows in row_mappings))
        trailing_shape = tuple(self._arrays[0].shape[1:])
        dtype = np.dtype(self._arrays[0].dtype)
        self._lengths = np.asarray([len(rows) for rows in self._rows], dtype=np.int64)
        self._offsets = np.concatenate(([0], np.cumsum(self._lengths)))
        self.shape = (int(self._offsets[-1]), *trailing_shape)
        self.ndim = len(self.shape)
        self.dtype = dtype

    def __len__(self):
        return self.shape[0]

    def _normalize_row_key(self, key):
        if isinstance(key, slice):
            return (np.arange(len(self), dtype=np.int64)[key], False)
        values = np.asarray(key)
        if values.ndim == 0:
            row = int(values)
            if row < 0:
                row += len(self)
            return (np.asarray([row], dtype=np.int64), True)
        if values.dtype == np.bool_:
            values = np.flatnonzero(values)
        values = values.astype(np.int64, copy=False)
        values = np.where(values < 0, values + len(self), values)
        return (values, False)

    def __getitem__(self, key):
        trailing_key = ()
        row_key = key
        if isinstance(key, tuple):
            if not key:
                return self
            (row_key, trailing_key) = (key[0], key[1:])
        (logical_rows, scalar) = self._normalize_row_key(row_key)
        logical_shape = logical_rows.shape
        flat_rows = logical_rows.reshape(-1)
        result = np.empty((len(flat_rows), *self.shape[1:]), dtype=self.dtype)
        for (source_index, (start, stop)) in enumerate(zip(self._offsets[:-1], self._offsets[1:])):
            mask = (flat_rows >= start) & (flat_rows < stop)
            if mask.any():
                local_rows = flat_rows[mask] - start
                physical_rows = self._rows[source_index][local_rows]
                result[mask] = np.asarray(self._arrays[source_index][physical_rows])
        result = result.reshape((*logical_shape, *self.shape[1:]))
        if scalar:
            result = result[0]
            return result[trailing_key] if trailing_key else result
        if trailing_key:
            prefix = (slice(None),) * len(logical_shape)
            result = result[prefix + trailing_key]
        return result


def augment_training_split(real_train, synthetic_root, prefix, ratio):
    rows = np.asarray(real_train.indices, dtype=np.int64)
    schedule = build_nested_schedule(real_train.labels[rows], rows)
    synthetic_rows = schedule[ratio_key(ratio)]
    synthetic_data = np.load(Path(synthetic_root) / f"{prefix}_train_data.npy", mmap_mode="r")
    data = IndexedConcatenatedArray((real_train.data, synthetic_data), (rows, synthetic_rows))
    labels = IndexedConcatenatedArray((real_train.labels, real_train.labels), (rows, synthetic_rows))
    return SplitData(data, labels, np.arange(len(data), dtype=np.int64))
