import argparse
import json
from pathlib import Path
import sys

import h5py
import numpy as np
import pandas as pd
import resampy

if __package__ in (None, ""):
    sys.path.insert(0, str(Path(__file__).resolve().parents[2]))

from src.scripts.utils.common import load_config, project_path

CONFIG_DIR = Path(__file__).resolve().parents[1] / "configs"
STANDARD_LEADS = ["I", "II", "III", "aVR", "aVL", "aVF", "V1", "V2", "V3", "V4", "V5", "V6"]


def primary_codes(value):
    statements = dict.fromkeys(part.strip() for part in str(value).split(";") if part.strip())
    return list(dict.fromkeys(
        next(part.strip() for part in statement.split("+") if part.strip())
        for statement in statements
    ))


def read_record(path, sampling_rate, signal_length, lead_names):
    with h5py.File(path, "r") as handle:
        signal = np.asarray(handle["ecg"], dtype=np.float32)
    data = np.stack([
        resampy.resample(signal[index], 500, sampling_rate).astype(np.float32)
        for index in range(signal.shape[0])
    ])
    length = data.shape[1]
    if length > signal_length:
        start = (length - signal_length) // 2
        data = data[:, start:start + signal_length]
    elif length < signal_length:
        fixed = np.zeros((len(STANDARD_LEADS), signal_length), dtype=np.float32)
        fixed[:, :length] = data
        data = fixed
    return data[[STANDARD_LEADS.index(name) for name in lead_names]]


def preprocess(cfg, raw_dir, output_dir, split_file):
    raw_dir, output_dir = Path(raw_dir), Path(output_dir)
    metadata = pd.read_csv(raw_dir / "metadata.csv", dtype={"ECG_ID": str, "AHA_Code": str}).set_index("ECG_ID")
    classes = json.loads((CONFIG_DIR / "sph_classes.json").read_text())
    label_indices = {str(name): index for index, name in enumerate(classes)}
    splits = pd.read_csv(split_file)
    arrays_dir = output_dir / cfg.data.array_subdir
    arrays_dir.mkdir(parents=True, exist_ok=True)
    for split in ("train", "val", "test"):
        records = metadata.loc[splits.loc[splits["split"] == split, "record_name"]]
        data = np.lib.format.open_memmap(
            arrays_dir / f"sph_{split}_data.npy", mode="w+", dtype=np.float32,
            shape=(len(records), len(cfg.data.lead_names), cfg.data.signal_length),
        )
        labels = np.lib.format.open_memmap(
            arrays_dir / f"sph_{split}_labels.npy", mode="w+", dtype=np.float32,
            shape=(len(records), len(classes)),
        )
        for index, (record_name, row) in enumerate(records.iterrows()):
            data[index] = read_record(raw_dir / "records" / f"{record_name}.h5",
                                      cfg.data.sampling_rate_hz, cfg.data.signal_length,
                                      cfg.data.lead_names)
            target = np.zeros(len(classes), dtype=np.float32)
            for code in primary_codes(row.AHA_Code):
                if code in label_indices:
                    target[label_indices[code]] = 1
            labels[index] = target
        data.flush()
        labels.flush()
    (output_dir / "label_metadata.json").write_text(json.dumps({"label_classes": {"label_primary": classes}}, indent=2))
    print(f"Saved arrays: {arrays_dir}", flush=True)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", default=str(CONFIG_DIR / "mcspnet_sph.yaml"))
    parser.add_argument("--raw-dir", required=True)
    parser.add_argument("--output-dir")
    parser.add_argument("--split-file", default=str(CONFIG_DIR / "sph_splits.csv"))
    args = parser.parse_args()
    cfg = load_config(args.config)
    preprocess(cfg, project_path(args.raw_dir), project_path(args.output_dir) if args.output_dir else cfg.data.root,
               project_path(args.split_file))


if __name__ == "__main__":
    main()
