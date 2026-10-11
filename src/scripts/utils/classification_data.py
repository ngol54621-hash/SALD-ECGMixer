from dataclasses import dataclass
import csv
import json
from pathlib import Path
import numpy as np
import torch
from torch.utils.data import DataLoader, Dataset
SSSD_8_LEADS = ('I', 'V1', 'V2', 'V3', 'V4', 'V5', 'V6', 'aVF')
STANDARD_12_LEADS = ('I', 'II', 'III', 'aVR', 'aVL', 'aVF', 'V1', 'V2', 'V3', 'V4', 'V5', 'V6')

@dataclass
class SplitData:
    data: np.ndarray
    labels: np.ndarray
    indices: np.ndarray

def sssd8_to_standard12(x):
    arr = np.asarray(x, dtype=np.float32)
    squeeze = arr.ndim == 2
    if squeeze:
        arr = arr[None]
    if arr.shape[1] == len(SSSD_8_LEADS):
        arr8 = arr
    elif arr.shape[2] == len(SSSD_8_LEADS):
        arr8 = np.transpose(arr, (0, 2, 1))
    lead_i = arr8[:, 0]
    (v1, v2, v3, v4, v5, v6) = [arr8[:, idx] for idx in range(1, 7)]
    avf = arr8[:, 7]
    lead_ii = avf + 0.5 * lead_i
    lead_iii = lead_ii - lead_i
    avr = -0.5 * (lead_i + lead_ii)
    avl = lead_i - 0.5 * lead_ii
    out = np.stack([lead_i, lead_ii, lead_iii, avr, avl, avf, v1, v2, v3, v4, v5, v6], axis=1)
    return out[0] if squeeze else out

class ECGArrayDataset(Dataset):

    def __init__(self, split, mean, std):
        self.split = split
        self.mean = np.asarray(mean, dtype=np.float32)
        self.std = np.asarray(std, dtype=np.float32)

    def __len__(self):
        return int(len(self.split.indices))

    def __getitem__(self, item):
        row = int(self.split.indices[item])
        raw_signal = self.split.data[row]
        signal = sssd8_to_standard12(raw_signal)
        signal = (signal - self.mean[:, None]) / self.std[:, None]
        label = np.asarray(self.split.labels[row], dtype=np.float32)
        return (torch.from_numpy(np.array(signal, copy=True)), torch.from_numpy(np.array(label, copy=True)))

def compute_train_normalization(split, batch_size=512, eps=1e-06):
    total = np.zeros(len(STANDARD_12_LEADS), dtype=np.float64)
    total_sq = np.zeros(len(STANDARD_12_LEADS), dtype=np.float64)
    count = 0
    for start in range(0, len(split.indices), batch_size):
        rows = split.indices[start:start + batch_size]
        batch = sssd8_to_standard12(selected_signal_batch(split, rows))
        total += batch.sum(axis=(0, 2), dtype=np.float64)
        total_sq += np.square(batch, dtype=np.float64).sum(axis=(0, 2))
        count += batch.shape[0] * batch.shape[2]
    mean = total / max(count, 1)
    variance = total_sq / max(count, 1) - mean * mean
    std = np.sqrt(np.maximum(variance, eps))
    return (mean.astype(np.float32), std.astype(np.float32))

def selected_signal_batch(split, rows):
    rows = np.asarray(rows, dtype=np.int64)
    return np.asarray(split.data[rows])

def seed_worker(worker_id):
    worker_seed = torch.initial_seed() % 2 ** 32
    np.random.seed(worker_seed)

def make_loader(split, mean, std, batch_size, shuffle, num_workers, seed):
    num_workers = int(num_workers)
    if num_workers > 0:
        torch.multiprocessing.set_sharing_strategy('file_system')
    generator = torch.Generator()
    generator.manual_seed(int(seed))
    return DataLoader(ECGArrayDataset(split, mean, std), batch_size=int(batch_size), shuffle=bool(shuffle), num_workers=num_workers, pin_memory=torch.cuda.is_available(), drop_last=bool(shuffle), worker_init_fn=seed_worker, generator=generator)


def load_splits(root, prefix, label_space, label_metadata):
    root = Path(root)
    metadata = json.loads(Path(label_metadata).read_text())
    source_space = "label_primary" if prefix == "sph" else "label_all"
    source_names = [str(name) for name in metadata["label_classes"][source_space]]
    classes = source_names
    mapping = None
    if prefix == "ptbxl" and label_space != "label_all":
        with (Path(__file__).resolve().parents[2] / "configs/ptbxl_diagnostic_mapping.csv").open() as handle:
            mapping = list(csv.DictReader(handle))
        classes = sorted(row["code"] for row in mapping) if label_space == "label_diag" else ["CD", "HYP", "MI", "NORM", "STTC"]
    splits = {}
    for name in ("train", "val", "test"):
        data = np.load(root / f"{prefix}_{name}_data.npy", mmap_mode="r")
        labels = np.load(root / f"{prefix}_{name}_labels.npy", mmap_mode="r")
        if mapping is not None:
            targets = np.zeros((len(labels), len(classes)), dtype=np.float32)
            for index, label in enumerate(classes):
                codes = [row["code"] for row in mapping if (row["code"] if label_space == "label_diag" else row["superclass"]) == label]
                targets[:, index] = np.any(labels[:, [source_names.index(code) for code in codes]] > 0, axis=1)
            labels = targets
        indices = np.flatnonzero(np.asarray(labels).sum(axis=1) > 0).astype(np.int64)
        splits[name] = SplitData(data, labels, indices)
    return splits, tuple(classes)
