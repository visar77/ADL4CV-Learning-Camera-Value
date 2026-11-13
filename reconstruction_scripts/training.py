"""FastGS commands, provenance, completion checks, and subprocess receipts."""

import hashlib
import json
import subprocess
import time

from reconstruction_scripts.datasets import training_manifests
from reconstruction_scripts.io import ROOT, load_json, save_json

ADAPTER_SCRIPT = ROOT / "fastgs_adapter.py"


def point_cloud_path(path, iterations):
    return path / "point_cloud" / f"iteration_{iterations}" / "point_cloud.ply"


def reconstruction_complete(path, iterations=30000, spec=None):
    if not point_cloud_path(path, iterations).is_file():
        return False
    if spec is None:
        return True
    receipt = path / "training_run.json"
    if receipt.is_file():
        run = load_json(receipt)
        return run.get("status") == "complete" and run.get("spec") == spec
    # Existing pipeline outputs have no receipt. Only accept them as legacy
    # 30k scratch runs, never as evidence that a warm-start run finished.
    return spec["initialization"] == "scratch" and iterations == 30000


def training_spec(args, global_train, indices, iterations, initial_sha256=None):
    frames = [global_train["frames"][i] for i in indices]
    manifest = {"camera_angle_x": global_train["camera_angle_x"], "frames": frames}
    spec = {
        "version": 3,
        "initialization": "base" if initial_sha256 else "scratch",
        "iterations": iterations,
        "train_lattice_indices": indices,
        "train_manifest_sha256": hashlib.sha256(
            json.dumps(manifest, sort_keys=True).encode()
        ).hexdigest(),
    }
    if getattr(args, "fastgs_sha256", None):
        spec["fastgs_sha256"] = args.fastgs_sha256
    if getattr(args, "data_fingerprint", None):
        spec["input_data_sha256"] = args.data_fingerprint
    if initial_sha256:
        spec.update(
            {
                "adapter_sha256": hashlib.sha256(
                    ADAPTER_SCRIPT.read_bytes()
                    + (ROOT / "fastgs_schedule.py").read_bytes()
                ).hexdigest(),
                "initialization_sha256": initial_sha256,
                "adaptation_lr_scale": args.adaptation_lr_scale,
                "adaptation_densify_from_iter": args.adaptation_densify_from_iter,
                "adaptation_densify_until_iter": args.adaptation_densify_until_iter,
                "adaptation_densification_interval": args.adaptation_densification_interval,
            }
        )
    return spec


def training_command(
    args, object_dir, reconstruction_dir, iterations, initial_ply=None
):
    command = [
        args.python,
        str(ADAPTER_SCRIPT if initial_ply else args.fastgs_dir / "train.py"),
        "--source_path",
        str(object_dir),
        "--model_path",
        str(reconstruction_dir),
        "--eval",
        "--iterations",
        str(iterations),
        "--densification_interval",
        "500",
        "--optimizer_type",
        "default",
        "--save_iterations",
        str(iterations),
        "--grad_abs_thresh",
        "0.0012",
        # Evaluation is separate by default, avoiding duplicate LPIPS/rendering.
        "--test_iterations",
    ]
    if args.evaluate_during_training:
        command.append(str(iterations))
    elif not initial_ply:
        # Unmodified FastGS requires at least one value here. No positive
        # iteration matches this sentinel, so its evaluation remains disabled.
        command.append("-1")
    if initial_ply:
        command.extend(
            [
                "--fastgs-dir",
                str(args.fastgs_dir),
                "--init_ply",
                str(initial_ply),
                "--adaptation_lr_scale",
                str(args.adaptation_lr_scale),
                "--adaptation_densify_from_iter",
                str(args.adaptation_densify_from_iter),
                "--adaptation_densify_until_iter",
                str(args.adaptation_densify_until_iter),
                "--adaptation_densification_interval",
                str(args.adaptation_densification_interval),
            ]
        )
    return command


def train_reconstruction(
    args,
    object_dir,
    reconstruction_dir,
    global_train,
    indices,
    spec,
    initial_ply=None,
    force=False,
):
    command = training_command(
        args, object_dir, reconstruction_dir, spec["iterations"], initial_ply
    )
    if not force and reconstruction_complete(
        reconstruction_dir, spec["iterations"], spec
    ):
        print(f"Skipping complete reconstruction: {reconstruction_dir}", flush=True)
        return
    if args.dry_run:
        if not hasattr(args, "planned_reconstructions"):
            args.planned_reconstructions = set()
        key = (str(reconstruction_dir.resolve()), json.dumps(spec, sort_keys=True))
        if key in args.planned_reconstructions:
            return
        args.planned_reconstructions.add(key)
        print(json.dumps({"command": command, "training": spec}, indent=2))
        return
    reconstruction_dir.mkdir(parents=True, exist_ok=True)
    receipt_path = reconstruction_dir / "training_run.json"
    receipt = {
        "status": "running",
        "spec": spec,
        "command": command,
        "environment": getattr(args, "runtime_environment", None),
        "timing_definition": "subprocess wall time including startup, loading, training and saving; excludes evaluation",
    }
    save_json(receipt_path, receipt)
    start = time.perf_counter()
    final_ply = point_cloud_path(reconstruction_dir, spec["iterations"])
    before = final_ply.stat().st_mtime_ns if final_ply.is_file() else None
    try:
        with training_manifests(object_dir, global_train, indices):
            start = time.perf_counter()
            subprocess.run(command, cwd=args.fastgs_dir, check=True)
        if not final_ply.is_file() or final_ply.stat().st_mtime_ns == before:
            raise RuntimeError(
                f"FastGS exited without its expected final PLY: {reconstruction_dir}"
            )
        receipt["status"] = "complete"
    except BaseException:
        receipt["status"] = "failed"
        raise
    finally:
        receipt["wall_time_seconds"] = time.perf_counter() - start
        save_json(receipt_path, receipt)
