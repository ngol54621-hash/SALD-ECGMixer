import importlib
from pathlib import Path

import numpy as np
import torch
import yaml

from src.models.ecgmixer import ECGMixer
from src.models.synthetic_knowledge import SyntheticKnowledgeTrainingModel, inference_model
from src.scripts.metrics.classification import classification_metrics, fixed_thresholds
from src.scripts.utils.common import Config, project_path


def load_classification_config(path):
    cfg = Config(yaml.safe_load(Path(path).read_text()))
    cfg.output_root = project_path(cfg.output_root)
    for dataset in cfg.datasets.values():
        dataset.root = project_path(dataset.root)
    return cfg


def build_classifier(num_classes, training, factory=None, model_kwargs=None):
    constructor = ECGMixer
    if factory:
        module, name = factory.split(":")
        constructor = getattr(importlib.import_module(module), name)
    kwargs = dict(model_kwargs) if model_kwargs is not None else {
        "in_ch": 12, "num_classes": num_classes,
        "drop": float(training.drop), "drop_path_rate": float(training.drop_path_rate),
    }
    return constructor(**kwargs)


def load_classifier(path, num_classes, training, device, factory=None):
    checkpoint = torch.load(project_path(path), map_location="cpu", weights_only=False)
    model = build_classifier(num_classes, training, factory, checkpoint.get("model_kwargs"))
    model.load_state_dict(checkpoint["model"])
    return model.to(device), checkpoint


def attach_teacher(student, path, num_classes, training, mean, std, device, weight, temperature):
    with torch.random.fork_rng(devices=[]):
        teacher, checkpoint = load_classifier(path, num_classes, training, "cpu")
    return SyntheticKnowledgeTrainingModel(student, teacher, weight, mean, std,
                                            checkpoint["mean"], checkpoint["std"],
                                            temperature=temperature).to(device)


def train_epoch(model, loader, criterion, optimizer, scaler, device, grad_clip, amp, amp_dtype):
    model.train()
    for signals, labels in loader:
        signals = signals.to(device, non_blocking=True)
        labels = labels.to(device, non_blocking=True)
        optimizer.zero_grad(set_to_none=True)
        with torch.amp.autocast(device_type=device.type, enabled=amp, dtype=amp_dtype):
            logits = model(signals)
            loss = criterion(logits.float(), labels.float())
            auxiliary = getattr(model, "auxiliary_loss", None)
            if auxiliary is not None:
                extra = auxiliary(labels)
                if extra is not None:
                    loss = loss + extra.float()
        if scaler.is_enabled():
            scaler.scale(loss).backward()
            scaler.unscale_(optimizer)
        else:
            loss.backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), grad_clip)
        if scaler.is_enabled():
            scaler.step(optimizer)
            scaler.update()
        else:
            optimizer.step()


@torch.no_grad()
def evaluate(model, loader, device):
    model.eval()
    scores, targets = [], []
    for signals, labels in loader:
        signals = signals.to(device, non_blocking=True)
        logits = model(signals)
        scores.append(torch.sigmoid(logits.float()).cpu().numpy())
        targets.append(labels.numpy())
    return np.concatenate(targets), np.concatenate(scores)


def metric_result(targets, scores, classes):
    metrics, predictions = classification_metrics(targets, scores, fixed_thresholds(len(classes)), classes)
    metrics["F1_PR"] = metrics.pop("F1")
    return metrics, predictions


def save_weights(path, model, mean, std):
    torch.save({"model": inference_model(model).state_dict(), "mean": mean, "std": std}, path)
