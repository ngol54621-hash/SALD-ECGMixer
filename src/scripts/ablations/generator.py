import argparse
import copy
from pathlib import Path
import sys

if __package__ in (None, ""):
    sys.path.insert(0, str(Path(__file__).resolve().parents[3]))

import yaml

from src.scripts import eval_generation_quality, generate_dataset, train_dm
from src.scripts.utils.common import Config, load_config, project_path
from src.scripts.utils.paths import find_latest_run, prepare_run_dir

DEFAULT_CONFIG = Path(__file__).resolve().parents[2] / "configs/ablations/generator.yaml"


def variant_config(base, suite, variant, output_root):
    cfg = copy.deepcopy(base)
    cfg.output_root = str(Path(output_root) / cfg.data.dataset_id / variant)
    if variant == "without_frequency":
        cfg.ldm.fft_weight = 0.0
    elif variant == "self_attention_dit":
        cfg.ldm.factory = "src.models.ablations.standard_dit:StandardDiT1D"
        cfg.ldm.standard_dit = copy.deepcopy(suite.standard_dit)
    elif variant != "full":
        cfg.ldm.factory = "src.models.ablations.mcspnet:MCSPNetAblation"
        cfg.ldm.ablation_variant = variant
    return cfg


def main(argv=None):
    parser = argparse.ArgumentParser(description="Generator ablations; defaults to PTB-XL with 71 labels.")
    parser.add_argument("action", choices=("train", "generate", "evaluate"))
    parser.add_argument("--config", default=str(DEFAULT_CONFIG))
    parser.add_argument("--base-config", help="Generator config for the selected dataset")
    parser.add_argument("--variants", nargs="+", help="Subset of variants listed in the ablation config")
    parser.add_argument("--output-root")
    parser.add_argument("--vae-run")
    parser.add_argument("--dm-run", help="Trained denoiser run directory; accepts {variant}")
    parser.add_argument("--synthetic-root", help="Generated arrays; accepts {variant} and {guidance}")
    parser.add_argument("--device", help="Single device used by this run, e.g. cuda:0")
    parser.add_argument("--seed", type=int)
    parser.add_argument("--splits", default="test")
    parser.add_argument("--batch-size", type=int, default=128)
    parser.add_argument("--guidance-scale", type=float)
    parser.add_argument("--sweep-guidance", action="store_true", help="Generate or evaluate the configured CFG values")
    parser.add_argument("--hubert-model-dir")
    parser.add_argument("--real-split", choices=("val", "test"), default="test")
    args = parser.parse_args(argv)
    suite = Config(yaml.safe_load(Path(args.config).read_text()))
    base = load_config(project_path(args.base_config or suite.base_config))
    if args.device:
        base.device = args.device
    if args.seed is not None:
        base.seed = args.seed
    output_root = project_path(args.output_root or suite.output_root)
    variants = args.variants or suite.variants
    for variant in variants:
        cfg = variant_config(base, suite, variant, output_root)
        if args.action == "train":
            command = ["--run-name", f"seed{cfg.seed}"]
            if args.vae_run:
                command += ["--vae-run", project_path(args.vae_run)]
            train_dm.main(command, cfg=cfg)
            continue
        scales = suite.guidance_scales if args.sweep_guidance else [
            cfg.sample.guidance_scale if args.guidance_scale is None else args.guidance_scale
        ]
        for guidance in scales:
            tag = f"g{guidance:g}_seed{cfg.seed}".replace(".", "p")
            if args.action == "generate":
                dm_run = project_path(args.dm_run.format(variant=variant)) if args.dm_run else find_latest_run(cfg, "dm")
                command = ["--dm-run", dm_run, "--run-name", tag,
                           "--splits", args.splits, "--batch-size", str(args.batch_size),
                           "--guidance-scale", str(guidance), "--devices", str(cfg.device)]
                if args.vae_run:
                    command += ["--vae-run", project_path(args.vae_run)]
                generate_dataset.main(command, cfg=cfg)
            else:
                if args.synthetic_root:
                    arrays = project_path(args.synthetic_root.format(variant=variant, guidance=f"{guidance:g}"))
                else:
                    candidates = [path for path in (Path(cfg.output_root) / "generate").iterdir()
                                  if path.name == tag or path.name.startswith(tag + "-")]
                    arrays = str(max(candidates, key=lambda path: path.stat().st_mtime) / "sssd_arrays")
                output = Path(prepare_run_dir(cfg, "evaluation", tag)) / "metrics.json"
                command = ["--synthetic-root", arrays, "--real-split", args.real_split,
                           "--output", str(output)]
                if args.hubert_model_dir:
                    command += ["--hubert-model-dir", args.hubert_model_dir]
                eval_generation_quality.main(command, cfg=cfg)


if __name__ == "__main__":
    main()
