"""Build, read, and migrate JSONL manifests using renderer results and cameras."""
from __future__ import annotations

from dataclasses import replace
from pathlib import Path
import json
import math
import shutil

from PIL import Image
import numpy as np
import torch

from .records import GainSample, audit_samples, deduplicate_samples, require_calibration, sample_key
from gain_predictor.artifacts import PROJECT_ROOT, relative_path, resolve_reference
from gain_predictor.geometry import to_opencv


def _records(path):
    payload = json.loads(path.read_text(encoding="utf-8"))
    if isinstance(payload, dict) and "results" in payload:
        payload = payload["results"]
    return payload if isinstance(payload, list) else [payload]


def _intrinsics(payload, frame, image_path=None, *, image_size=None):
    if image_size is None:
        with Image.open(image_path) as image:
            image_size = image.size
    width, height = image_size
    values = {**payload, **frame}
    if "fl_x" in values:
        fx = float(values["fl_x"])
        fy = float(values.get("fl_y", fx))
    elif "camera_angle_x" in values:
        fx = width / (2 * math.tan(float(values["camera_angle_x"]) / 2))
        fy = (height / (2 * math.tan(float(values["camera_angle_y"]) / 2))
              if "camera_angle_y" in values else fx)
    else:
        raise ValueError(f"Missing calibrated intrinsics for {image_path}")
    return (fx / width, fy / height, float(values.get("cx", width / 2)) / width,
            float(values.get("cy", height / 2)) / height)


def build_manifest(results_root: Path | list[Path], views_root: Path, output: Path, *,
                   metric="PSNR", object_ids=None, limit=None,
                   duplicates="mean") -> list[GainSample]:
    samples = []
    approximated_redundancy_candidates = []
    roots = [results_root] if isinstance(results_root, Path) else list(results_root)
    requested_object_ids = set(object_ids or ())
    found_object_ids = set()
    if metric not in {"PSNR", "SSIM", "LPIPS"}:
        raise ValueError("metric must be PSNR, SSIM, or LPIPS")
    metric_name = metric
    result_files = sorted({path for root in roots for path in root.rglob("results.json")})
    if not result_files:
        roots_text = ", ".join(str(root.resolve()) for root in roots)
        raise ValueError(
            f"No results.json files found under {roots_text}. "
            "This rebuild reads reconstruction results, not an existing JSONL manifest. "
            "To add calibration to an existing dataset without rebuilding its labels, use "
            "python prepare_data.py --upgrade <existing-manifest.jsonl>. "
            "Otherwise specify --results-root and --views-root for your existing renderer outputs."
        )
    for results_path in result_files:
        for raw in _records(results_path):
            scene_id = str(raw["object_id"])
            if requested_object_ids and scene_id not in requested_object_ids:
                continue
            found_object_ids.add(scene_id)
            view_dir = views_root / scene_id
            reconstructions = raw.get("reconstructions", {})
            base_reconstruction = reconstructions.get("base", {})
            context_indices = raw.get("train_lattice_indices")
            if context_indices is None:
                context_indices = base_reconstruction.get("train_lattice_indices")
            if context_indices is None:
                raise ValueError(f"Missing base training camera indices in {results_path}")
            if not context_indices:
                raise ValueError(f"Base training camera indices are empty in {results_path}")
            redundancy_ids = sorted(
                candidate_id for candidate_id, candidate in reconstructions.items()
                if candidate_id != "base" and candidate.get("status") == "redundant"
            )
            redundancy_camera_indices = {
                candidate_id: context_indices[index % len(context_indices)]
                for index, candidate_id in enumerate(redundancy_ids)
            }
            # Lattice indices always refer to the global manifest, never a temporary subset.
            transforms_path = view_dir / "global_transforms" / "transforms_train.json"
            if not transforms_path.is_file():
                transforms_path = view_dir / "transforms_train.json"
            payload = json.loads(transforms_path.read_text())
            frames = payload["frames"]
            def camera(index):
                frame = frames[int(index)]
                image_path = view_dir / frame["file_path"]
                if not image_path.suffix:
                    image_path = image_path.with_suffix(".png")
                if not image_path.is_file():
                    raise FileNotFoundError(image_path)
                pose = tuple(float(v) for row in frame["transform_matrix"] for v in row)
                return relative_path(image_path, output.parent), pose, _intrinsics(payload, frame, image_path)
            context = [camera(i) for i in context_indices]
            try:
                base = float(raw["metrics"]["base"]["averages"][metric_name])
            except KeyError as error:
                raise ValueError(f"Missing base {metric_name} score in {results_path}") from error
            if not math.isfinite(base) or (metric_name == "PSNR" and abs(base) < 1e-6):
                raise ValueError(f"Invalid base {metric_name} score in {results_path}")
            source = relative_path(results_path, output.parent)
            for candidate_id, candidate in reconstructions.items():
                if candidate_id == "base" or candidate.get("status") not in {"complete", "redundant"}:
                    continue
                is_redundant = candidate.get("status") == "redundant"
                candidate_scores = raw["metrics"].get(candidate_id, {}).get("averages", {})
                if metric_name not in candidate_scores and not is_redundant:
                    continue
                index = candidate.get("candidate_lattice_index")
                candidate_indices = candidate.get("train_lattice_indices")
                if index is None and candidate_indices is not None:
                    added_indices = set(map(int, candidate_indices)) - set(map(int, context_indices))
                    if len(added_indices) == 1:
                        index = added_indices.pop()
                    elif added_indices or not is_redundant:
                        raise ValueError(f"Expected one added camera for {candidate_id} in {results_path}; "
                                         f"found {len(added_indices)}.")
                if index is None and is_redundant:
                    index = redundancy_camera_indices[candidate_id]
                    approximated_redundancy_candidates.append({
                        "source_results": source,
                        "candidate_id": str(candidate_id),
                        "candidate_lattice_index": int(index),
                        "reason": "candidate index missing; assigned a context camera as its redundant pose",
                    })
                if index is None:
                    raise ValueError(f"Missing candidate camera index for {candidate_id} in {results_path}")
                _, pose, intrinsics = camera(index)
                value = (float(candidate_scores[metric_name])
                         if metric_name in candidate_scores else base)
                if not math.isfinite(value):
                    raise ValueError(f"Non-finite candidate {metric_name} score in {results_path}")
                candidate_score = base if is_redundant else value
                score_direction = -1 if metric_name == "LPIPS" else 1
                metric_gain = 0.0 if is_redundant else score_direction * (value - base)
                gain = metric_gain / abs(base) if metric_name == "PSNR" else metric_gain
                candidate_psnr = candidate_score if metric_name == "PSNR" else None
                sample = GainSample(scene_id, str(raw["experiment_id"]), str(candidate_id),
                                    tuple(c[0] for c in context), tuple(c[1] for c in context),
                                    pose, gain, source,
                                    base if metric_name == "PSNR" else None, candidate_psnr,
                                    tuple(c[2] for c in context), intrinsics,
                                    payload.get("camera_convention", "blender"),
                                    "redundancy_control" if is_redundant else "retrained",
                                    observed_gains=(0.0,) if is_redundant else (),
                                    path_base="manifest", path_root=output.resolve().parent,
                                    metric=metric_name, base_metric=base, candidate_metric=candidate_score)
                samples.append(sample)
    missing_object_ids = sorted(requested_object_ids - found_object_ids)
    if missing_object_ids:
        raise ValueError(f"No result records found for object ID(s): {', '.join(missing_object_ids)}")
    if not samples:
        roots_text = ", ".join(str(root.resolve()) for root in roots)
        raise ValueError(f"No complete {metric_name} candidate samples found in {len(result_files)} results.json files "
                         f"under {roots_text}. Check candidate statuses and {metric_name} metrics. "
                         "An existing JSONL can instead be calibrated with --upgrade; labels are preserved.")
    report = audit_samples(samples)
    report["metric"] = metric_name
    report["skipped_candidates"] = []  # Retain the legacy audit field.
    report["approximated_redundancy_candidates"] = approximated_redundancy_candidates
    if duplicates == "mean":
        samples = deduplicate_samples(samples)
    elif duplicates != "keep":
        raise ValueError("duplicates must be mean or keep")
    if limit is not None:
        samples = samples[:limit]
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text("".join(json.dumps(s.to_json()) + "\n" for s in samples))
    report["written_records"] = len(samples)
    output.with_suffix(".audit.json").write_text(json.dumps(report, indent=2))
    return samples


def load_manifest(path: Path) -> list[GainSample]:
    samples = []
    for line_number, line in enumerate(path.read_text().splitlines(), 1):
        if not line.strip():
            continue
        try:
            raw = json.loads(line)
            for key in ("context_poses", "context_intrinsics"):
                if key in raw:
                    raw[key] = tuple(tuple(float(v) for v in row) for row in raw[key])
            for key in ("candidate_pose", "candidate_intrinsics", "context_image_paths",
                        "label_sources", "observed_gains"):
                if raw.get(key) is not None:
                    raw[key] = tuple(raw[key])
            root = path.resolve().parent if raw.get("path_base") == "manifest" else PROJECT_ROOT
            sample = GainSample(**raw, path_root=root)
            if sample.path_base not in {None, "manifest"}:
                raise ValueError("path_base must be manifest")
            if not sample.context_poses or len(sample.context_image_paths) != len(sample.context_poses):
                raise ValueError("context image/pose lengths differ")
            poses = (*sample.context_poses, sample.candidate_pose)
            if any(len(p) != 16 or not all(math.isfinite(v) for v in p) for p in poses):
                raise ValueError("poses must be finite flattened 4x4 matrices")
            if sample.camera_convention not in {"blender", "opencv"}:
                raise ValueError("camera_convention must be blender or opencv")
            if sample.context_intrinsics and len(sample.context_intrinsics) != len(sample.context_poses):
                raise ValueError("context intrinsics/pose lengths differ")
            intrinsics = (*sample.context_intrinsics, *((sample.candidate_intrinsics,) if sample.candidate_intrinsics else ()))
            if any(len(k) != 4 or not all(math.isfinite(v) for v in k) or min(k[:2]) <= 0 for k in intrinsics):
                raise ValueError("intrinsics must contain finite fx,fy,cx,cy with positive focal lengths")
            values = (sample.psnr_gain, *sample.observed_gains,
                      *((sample.base_psnr,) if sample.base_psnr is not None else ()),
                      *((sample.candidate_psnr,) if sample.candidate_psnr is not None else ()),
                      *((sample.base_metric,) if sample.base_metric is not None else ()),
                      *((sample.candidate_metric,) if sample.candidate_metric is not None else ()))
            if not all(math.isfinite(v) for v in values):
                raise ValueError("labels must be finite")
            if sample.metric not in {"PSNR", "SSIM", "LPIPS"}:
                raise ValueError("metric must be PSNR, SSIM, or LPIPS")
            samples.append(sample)
        except (KeyError, TypeError, ValueError) as error:
            raise ValueError(f"Invalid manifest record at line {line_number}: {error}") from error
    if not samples:
        raise ValueError(f"No records found in {path}")
    if len({sample_key(s) for s in samples}) != len(samples):
        raise ValueError("Manifest contains duplicate sample IDs.")
    return samples


def relativize_manifest(path: Path):
    """Migrate paths in place, preserving labels, observations and provenance IDs."""
    samples = load_manifest(path)
    migrated = []
    for sample in samples:
        for image in sample.image_paths:
            if not image.is_file():
                raise FileNotFoundError(f"Cannot migrate missing image {image}. Regenerate from local renderer data.")
        source = resolve_reference(sample.source_results, sample.path_root)
        migrated.append(replace(sample,
                                context_image_paths=tuple(relative_path(p, path.parent) for p in sample.image_paths),
                                source_results=relative_path(source, path.parent),
                                path_base="manifest", path_root=path.resolve().parent))
    temporary = path.with_name(path.name + ".part")
    temporary.write_text("".join(json.dumps(s.to_json()) + "\n" for s in migrated))
    temporary.replace(path)
    path.with_suffix(".audit.json").write_text(json.dumps(audit_samples(migrated), indent=2))
    return migrated


def _transform_paths(root):
    paths = []
    for name in ("transforms_train.json", "transforms_test.json"):
        global_path = root / "global_transforms" / name
        candidate = global_path if global_path.is_file() else root / name
        if candidate.is_file():
            paths.append(candidate)
    return paths


def _calibration_root(sample, views_root):
    if views_root is not None:
        root = Path(views_root) / sample.scene_id
        if _transform_paths(root):
            return root
        raise ValueError(f"No camera transform JSON found under {root}.")
    # Locate metadata from the existing image, not a reconstruction-results directory.
    image = sample.image_paths[0]
    for location in (image.absolute(), image.resolve()):
        for ancestor in location.parents:
            if _transform_paths(ancestor):
                return ancestor
    raise ValueError(f"Cannot find camera transforms beside {sample.image_paths[0]}. "
                     "Supply --views-root pointing to the existing directory containing scene folders. "
                     "Calibration cannot be inferred from RGB/poses alone; no default FOV is assumed.")


def upgrade_manifest(path: Path, *, views_root=None):
    """Fill legacy calibration from existing camera JSONs, without rebuilding labels.

    No result scores, reconstruction files, GPU, or candidate RGB are required.
    Match context images and poses, and match candidate poses against the global
    camera catalogue. Validate every record before writing or creating a backup.
    """
    samples = load_manifest(path)
    catalogues = {}
    migrated = []
    for sample in samples:
        image_sizes = []
        for image_path in sample.image_paths:
            with Image.open(image_path) as image:
                image_sizes.append(image.size)
        context_intrinsics = sample.context_intrinsics
        candidate_intrinsics = sample.candidate_intrinsics
        if not context_intrinsics or candidate_intrinsics is None:
            root = _calibration_root(sample, views_root)
            if root not in catalogues:
                frames = []
                for transform in _transform_paths(root):
                    payload = json.loads(transform.read_text())
                    convention = payload.get("camera_convention", "blender")
                    for frame in payload["frames"]:
                        image = root / frame["file_path"]
                        if not image.suffix:
                            image = image.with_suffix(".png")
                        pose = to_opencv(torch.tensor(frame["transform_matrix"], dtype=torch.float64), convention).numpy()
                        frames.append((payload, frame, image.resolve(), pose))
                catalogues[root] = frames
            catalogue = catalogues[root]
            def calibrated(pose_values, image_path=None, fallback_size=None):
                pose = to_opencv(torch.tensor(pose_values, dtype=torch.float64).reshape(4, 4),
                                 sample.camera_convention).numpy()
                matches = [entry for entry in catalogue if np.allclose(entry[3], pose, rtol=0, atol=1e-5)]
                if image_path is not None:
                    by_path = [entry for entry in catalogue if entry[2] == image_path.resolve()]
                    if by_path:
                        matches = [entry for entry in by_path if np.allclose(entry[3], pose, rtol=0, atol=1e-5)]
                if not matches:
                    raise ValueError(f"Camera pose for {sample_key(sample)} is absent from {root}; "
                                     "verify that its existing images and transforms belong to the same corpus.")
                values = []
                for payload, frame, image, _ in matches:
                    info = {**payload, **frame}
                    size = (info["w"], info["h"]) if "w" in info and "h" in info else None
                    if size is None and image.is_file():
                        with Image.open(image) as source:
                            size = source.size
                    if size is None:
                        size = fallback_size
                    if size is None:
                        raise ValueError(f"Image dimensions are required for candidate calibration in {root}; "
                                         "store w/h in its camera metadata when image sizes differ.")
                    value = _intrinsics(payload, frame, image_size=size)
                    if not all(math.isfinite(x) for x in value) or min(value[:2]) <= 0:
                        raise ValueError(f"Invalid calibration in {root}.")
                    values.append(value)
                if any(not np.allclose(values[0], value, rtol=0, atol=1e-7) for value in values[1:]):
                    raise ValueError(f"Ambiguous candidate calibration for {sample_key(sample)} in {root}.")
                return values[0]
            if not context_intrinsics:
                context_intrinsics = tuple(calibrated(pose, image, size) for pose, image, size in
                                           zip(sample.context_poses, sample.image_paths, image_sizes))
            if candidate_intrinsics is None:
                # Shared-size rendered corpora need no candidate RGB for FOV calibration.
                common_size = image_sizes[0] if len(set(image_sizes)) == 1 else None
                candidate_intrinsics = calibrated(sample.candidate_pose, fallback_size=common_size)
        source = resolve_reference(sample.source_results, sample.path_root)
        migrated.append(replace(sample, context_intrinsics=context_intrinsics,
                                candidate_intrinsics=candidate_intrinsics,
                                context_image_paths=tuple(relative_path(p, path.parent) for p in sample.image_paths),
                                source_results=relative_path(source, path.parent),
                                path_base="manifest", path_root=path.resolve().parent))
    require_calibration(migrated, path)
    content = "".join(json.dumps(s.to_json()) + "\n" for s in migrated)
    if content != path.read_text():
        backup = path.with_name(path.name + ".bak")
        if backup.exists():
            # Never replace a previous backup with a different original dataset.
            index = 1
            while backup.with_name(backup.name + f".{index}").exists():
                index += 1
            backup = backup.with_name(backup.name + f".{index}")
        shutil.copy2(path, backup)
        temporary = path.with_name(path.name + ".part")
        temporary.write_text(content)
        temporary.replace(path)
    path.with_suffix(".audit.json").write_text(json.dumps(audit_samples(migrated), indent=2))
    return migrated
