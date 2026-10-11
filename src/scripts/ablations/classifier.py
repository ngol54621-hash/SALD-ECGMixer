import argparse
import copy
import csv
import json
from pathlib import Path
import sys

if __package__ in (None, ""):
    sys.path.insert(0, str(Path(__file__).resolve().parents[3]))

import numpy as np
import yaml

from src.scripts import train_classifier
from src.scripts.utils.classification import load_classification_config
from src.scripts.utils.common import Config, project_path
from src.scripts.utils.paths import prepare_run_dir

DEFAULT_CONFIG = Path(__file__).resolve().parents[2] / "configs/ablations/classifier.yaml"


def experiment_conditions(suite, experiment, variants, factory):
    if experiment == "components":
        return [(name, {"factory": suite.variants[name]}) for name in (variants or suite.variants)]
    if experiment == "learning_rate":
        return [(f"lr_{value:g}", {"factory": factory, "lr": value}) for value in suite.learning_rates]
    if experiment == "batch_size":
        return [(f"batch_{value}", {"factory": factory, "batch_size": value}) for value in suite.batch_sizes]
    return [(f"weight_{weight:g}_T{temperature:g}", {
        "factory": factory, "weight": weight, "temperature": temperature,
    }) for weight in suite.distillation_weights for temperature in suite.temperatures]


def main(argv=None):
    parser = argparse.ArgumentParser(description="ECGMixer ablations; defaults to PTB-XL with 71 labels.")
    parser.add_argument("--config", default=str(DEFAULT_CONFIG))
    parser.add_argument("--base-config")
    parser.add_argument("--experiment", choices=("components", "learning_rate", "batch_size", "knowledge_transfer"), default="components")
    parser.add_argument("--variants", nargs="+", help="Component variants from the ablation config")
    parser.add_argument("--dataset", choices=("ptbxl", "cpsc2018", "sph"))
    parser.add_argument("--label-space")
    parser.add_argument("--seeds", type=int, nargs="+")
    parser.add_argument("--device")
    parser.add_argument("--output-root")
    parser.add_argument("--student-factory", default="src.models.ecgmixer:ECGMixer")
    parser.add_argument("--teacher-checkpoint", help="Required for knowledge_transfer; accepts {seed}")
    args = parser.parse_args(argv)
    suite = Config(yaml.safe_load(Path(args.config).read_text()))
    base = load_classification_config(project_path(args.base_config or suite.base_config))
    dataset = args.dataset or suite.dataset
    label_space = args.label_space or (suite.label_space if dataset == suite.dataset else base.datasets[dataset].label_spaces[0])
    seeds = args.seeds or base.seeds
    base.output_root = project_path(args.output_root or suite.output_root)
    if args.device:
        base.training.device = args.device
    root = Path(prepare_run_dir(base, f"{dataset}/{label_space}/{args.experiment}"))
    conditions = experiment_conditions(suite, args.experiment, args.variants, args.student_factory)
    summary = []
    for name, condition in conditions:
        values = []
        for seed in seeds:
            cfg = copy.deepcopy(base)
            cfg.output_root = str(root / name)
            for key in ("lr", "batch_size"):
                if key in condition:
                    cfg.training[key] = condition[key]
            command = ["--dataset", dataset, "--label-space", label_space,
                       "--seed", str(seed), "--run-name", f"seed{seed}",
                       "--student-factory", condition["factory"]]
            if args.experiment == "knowledge_transfer":
                command += ["--teacher-checkpoint", args.teacher_checkpoint.format(seed=seed),
                            "--distillation-weight", str(condition["weight"]),
                            "--temperature", str(condition["temperature"])]
            run = train_classifier.main(command, cfg=cfg)
            values.append(json.loads((run / "metrics.json").read_text()))
        row = {"variant": name}
        for metric in values[0]:
            scores = np.asarray([item[metric] for item in values])
            row[metric + "_mean"] = float(scores.mean())
            row[metric + "_std"] = float(scores.std(ddof=1)) if len(scores) > 1 else 0.0
        summary.append(row)
    with (root / "summary.csv").open("w", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(summary[0]))
        writer.writeheader()
        writer.writerows(summary)
    print(f"Saved results: {root / 'summary.csv'}", flush=True)


if __name__ == "__main__":
    main()
