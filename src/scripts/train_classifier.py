import argparse
import json
from pathlib import Path
import sys

if __package__ in (None, ""):
    sys.path.insert(0, str(Path(__file__).resolve().parents[2]))

from src.scripts.utils.classification_seed import set_seed

import numpy as np
import torch

from src.models.synthetic_knowledge import inference_model
from src.scripts.utils.augmentation import DEFAULT_RATIOS, augment_training_split
from src.scripts.utils.classification import (
    attach_teacher, build_classifier, evaluate, load_classification_config,
    metric_result, save_weights, train_epoch,
)
from src.scripts.utils.classification_data import compute_train_normalization, load_splits, make_loader
from src.scripts.utils.common import project_path
from src.scripts.utils.paths import prepare_run_dir

DEFAULT_CONFIG = Path(__file__).resolve().parents[1] / "configs/ecgmixer.yaml"


def main(argv=None, cfg=None):
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", default=str(DEFAULT_CONFIG))
    parser.add_argument("--dataset", choices=("ptbxl", "cpsc2018", "sph"), default="ptbxl")
    parser.add_argument("--label-space", default=None)
    parser.add_argument("--seed", type=int)
    parser.add_argument("--device")
    parser.add_argument("--run-name")
    parser.add_argument("--synthetic-root")
    parser.add_argument("--protocol", choices=("TRTR", "TSTS", "TSTR", "TRTS", "AUGMENT"), default="TRTR")
    parser.add_argument("--augmentation-ratio", type=float, choices=DEFAULT_RATIOS, default=1.0)
    parser.add_argument("--teacher-checkpoint")
    parser.add_argument("--student-factory", help="Importable module:factory for an external receiving classifier")
    parser.add_argument("--distillation-weight", type=float)
    parser.add_argument("--temperature", type=float)
    args = parser.parse_args(argv)
    cfg = load_classification_config(args.config) if cfg is None else cfg
    args.seed = cfg.seeds[0] if args.seed is None else args.seed
    training = cfg.training
    dataset = cfg.datasets[args.dataset]
    label_space = args.label_space or dataset.label_spaces[0]
    device = torch.device(args.device or training.device)
    if device.type == "cuda":
        torch.cuda.set_device(device)
    real, classes = load_splits(Path(dataset.root) / "sssd_arrays", dataset.array_prefix,
                               label_space, Path(dataset.root) / "label_metadata.json")
    mean, std = compute_train_normalization(real["train"])
    domains = {"TRTR": (0, 0, 0), "TSTS": (1, 1, 1), "TSTR": (1, 1, 0), "TRTS": (0, 0, 1)}
    sources = [real]
    if args.protocol in ("TSTS", "TSTR", "TRTS"):
        synthetic, _ = load_splits(project_path(args.synthetic_root), dataset.array_prefix,
                                   label_space, Path(dataset.root) / "label_metadata.json")
        sources.append(synthetic)
    if args.protocol == "AUGMENT":
        splits = dict(real)
        splits["train"] = augment_training_split(real["train"], project_path(args.synthetic_root),
                                                  dataset.array_prefix, args.augmentation_ratio)
    else:
        splits = {name: sources[index][name] for name, index in zip(("train", "val", "test"), domains[args.protocol])}
    set_seed(args.seed, training.deterministic)
    loaders = {name: make_loader(split, mean, std, training.batch_size, name == "train",
                                 training.num_workers, args.seed) for name, split in splits.items()}
    model = build_classifier(len(classes), training, args.student_factory).to(device)
    criterion_factory = getattr(model, "classification_criterion", None)
    criterion = (criterion_factory() if criterion_factory else torch.nn.BCEWithLogitsLoss()).to(device)
    if args.teacher_checkpoint:
        model = attach_teacher(model, args.teacher_checkpoint, len(classes), training, mean, std, device,
                               cfg.knowledge_transfer.weight if args.distillation_weight is None else args.distillation_weight,
                               cfg.knowledge_transfer.temperature if args.temperature is None else args.temperature)
    optimizer = torch.optim.AdamW((p for p in model.parameters() if p.requires_grad),
                                  lr=training.lr, weight_decay=training.weight_decay)
    scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(optimizer, mode="max", factor=0.5,
                                                          patience=training.lr_patience, min_lr=training.min_lr)
    amp = bool(training.amp and device.type == "cuda")
    amp_dtype = getattr(torch, training.amp_dtype)
    scaler = torch.amp.GradScaler("cuda", enabled=bool(amp and amp_dtype == torch.float16))
    condition = f"AUGMENT_{args.augmentation_ratio:g}" if args.protocol == "AUGMENT" else args.protocol
    run = Path(prepare_run_dir(cfg, args.dataset + "/" + label_space,
                               args.run_name or f"{condition}_seed{args.seed}"))
    checkpoints = run / "checkpoints"
    checkpoints.mkdir()
    best_score, bad_epochs = -float("inf"), 0
    for epoch in range(training.epochs):
        train_epoch(model, loaders["train"], criterion, optimizer, scaler, device,
                    training.grad_clip, amp, amp_dtype)
        targets, scores = evaluate(inference_model(model), loaders["val"], device)
        metrics, _ = metric_result(targets, scores, classes)
        score = float(metrics[training.selection_metric])
        scheduler.step(score)
        save_weights(checkpoints / "last.pt", model, mean, std)
        if np.isfinite(score) and score > best_score:
            best_score, bad_epochs = score, 0
            save_weights(checkpoints / "best.pt", model, mean, std)
        else:
            bad_epochs += 1
        if bad_epochs >= training.early_stop_patience:
            break
    checkpoint = torch.load(checkpoints / "best.pt", map_location=device, weights_only=False)
    inference_model(model).load_state_dict(checkpoint["model"])
    targets, scores = evaluate(inference_model(model), loaders["test"], device)
    metrics, predictions = metric_result(targets, scores, classes)
    (run / "metrics.json").write_text(json.dumps(metrics, indent=2) + "\n")
    np.savez_compressed(run / "predictions.npz", y_true=targets, y_score=scores, y_pred=predictions)
    print(f"Saved weights and results: {run}", flush=True)
    return run


if __name__ == "__main__":
    main()
