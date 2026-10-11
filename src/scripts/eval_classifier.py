import argparse
import json
from pathlib import Path
import sys

if __package__ in (None, ""):
    sys.path.insert(0, str(Path(__file__).resolve().parents[2]))

from src.scripts.utils.classification_seed import set_seed

import numpy as np
import torch

from src.scripts.utils.classification import evaluate, load_classifier, load_classification_config, metric_result
from src.scripts.utils.classification_data import load_splits, make_loader
from src.scripts.utils.common import project_path


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", default=str(Path(__file__).resolve().parents[1] / "configs/ecgmixer.yaml"))
    parser.add_argument("--dataset", choices=("ptbxl", "cpsc2018", "sph"), default="ptbxl")
    parser.add_argument("--label-space")
    parser.add_argument("--checkpoint", required=True)
    parser.add_argument("--array-root")
    parser.add_argument("--split", choices=("val", "test"), default="test")
    parser.add_argument("--device")
    parser.add_argument("--seed", type=int, default=24)
    parser.add_argument("--student-factory")
    parser.add_argument("--output", required=True)
    args = parser.parse_args()
    cfg = load_classification_config(args.config)
    dataset = cfg.datasets[args.dataset]
    set_seed(args.seed, cfg.training.deterministic)
    device = torch.device(args.device or cfg.training.device)
    if device.type == "cuda":
        torch.cuda.set_device(device)
    splits, classes = load_splits(project_path(args.array_root) if args.array_root else Path(dataset.root) / "sssd_arrays",
                                 dataset.array_prefix, args.label_space or dataset.label_spaces[0],
                                 Path(dataset.root) / "label_metadata.json")
    model, checkpoint = load_classifier(args.checkpoint, len(classes), cfg.training, device, args.student_factory)
    loader = make_loader(splits[args.split], checkpoint["mean"], checkpoint["std"], cfg.training.batch_size,
                         False, cfg.training.num_workers, args.seed)
    targets, scores = evaluate(model, loader, device)
    metrics, predictions = metric_result(targets, scores, classes)
    output = Path(project_path(args.output))
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(json.dumps(metrics, indent=2) + "\n")
    np.savez_compressed(output.with_suffix(".npz"), y_true=targets, y_score=scores, y_pred=predictions)
    print(f"Saved results: {output}", flush=True)


if __name__ == "__main__":
    main()
