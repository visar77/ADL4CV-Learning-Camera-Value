"""Launch FastGS training with the renderer's minimal training configuration."""

import argparse
import json
import subprocess
import sys
from pathlib import Path


ROOT = Path(__file__).resolve().parent
CONFIG_PATH = ROOT / "training_config.json"


def load_config():
    with CONFIG_PATH.open(encoding="utf-8") as file:
        config = json.load(file)
    if not isinstance(config, dict):
        raise ValueError(f"Training config must be a JSON object: {CONFIG_PATH}")

    fastgs_dir = Path(config["fastgs_dir"])
    config["fastgs_dir"] = (
        (ROOT / fastgs_dir).resolve()
        if not fastgs_dir.is_absolute()
        else fastgs_dir.resolve()
    )
    config["python"] = config.get("python") or sys.executable
    return config


CONFIG = load_config()


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--source-path", type=Path, required=True)
    parser.add_argument("--model-path", type=Path, required=True)
    parser.add_argument("--iterations", type=int, required=True)
    parser.add_argument("--init-ply", type=Path)
    parser.add_argument("--fastgs-dir", type=Path, default=CONFIG["fastgs_dir"])
    parser.add_argument("--python", default=CONFIG["python"])
    args = parser.parse_args(argv)

    fastgs_dir = args.fastgs_dir.expanduser().resolve()
    train_script = fastgs_dir / "train.py"
    if not train_script.is_file():
        parser.error(f"FastGS train.py not found: {train_script}")
    if args.iterations < 1:
        parser.error("--iterations must be positive")

    args.source_path = args.source_path.expanduser().resolve()
    args.model_path = args.model_path.expanduser().resolve()
    args.model_path.mkdir(parents=True, exist_ok=True)
    command = [
        args.python,
        str(ROOT / "fastgs_adapter.py") if args.init_ply else str(train_script),
    ]
    if args.init_ply:
        init_ply = args.init_ply.expanduser().resolve()
        if not init_ply.is_file():
            parser.error(f"Initialization PLY not found: {init_ply}")
        command.extend(
            [
                "--eval",
                "--fastgs-dir",
                str(fastgs_dir),
                "--init_ply",
                str(init_ply),
                "--adaptation_lr_scale",
                str(CONFIG["adaptation_lr_scale"]),
                "--adaptation_densify_from_iter",
                str(CONFIG["adaptation_densify_from_iter"]),
                "--adaptation_densification_interval",
                str(CONFIG["adaptation_densification_interval"]),
            ]
        )
    else:
        command.extend(
            [
                "--eval",
                "--densification_interval",
                str(CONFIG["densification_interval"]),
                "--optimizer_type",
                "default",
                "--grad_abs_thresh",
                str(CONFIG["grad_abs_thresh"]),
                "--test_iterations",
                "-1",
            ]
        )
    command.extend(
        [
            "--source_path",
            str(args.source_path),
            "--model_path",
            str(args.model_path),
            "--iterations",
            str(args.iterations),
            "--save_iterations",
            str(args.iterations),
        ]
    )
    subprocess.run(command, cwd=fastgs_dir, check=True)


if __name__ == "__main__":
    main()
