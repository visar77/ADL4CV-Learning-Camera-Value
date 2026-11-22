"""Render the approved street scene from the configured free camera positions."""

import json
import subprocess

from street_common import COMPLEX, load_street_config, run_dir, street_run_spec
from renderer import find_blender


def corpus_ready(directory, object_id):
    object_dir = directory / "views" / object_id
    metadata_path = object_dir / "metadata.json"
    train_path = object_dir / "global_transforms" / "transforms_train.json"
    test_path = object_dir / "global_transforms" / "transforms_test.json"
    if not all(path.is_file() for path in (metadata_path, train_path, test_path, object_dir / "grounded_scene.blend")):
        return False
    if json.loads(metadata_path.read_text(encoding="utf-8"))["object_id"] != object_id:
        raise ValueError("Saved corpus has the wrong object ID")
    for manifest in (train_path, test_path):
        frames = json.loads(manifest.read_text(encoding="utf-8"))["frames"]
        if not frames:
            raise ValueError(f"Empty camera manifest: {manifest}")
        for frame in frames:
            image = (object_dir / f"{frame['file_path']}.png").resolve()
            if not image.is_relative_to(object_dir.resolve()):
                raise ValueError(f"Image path escapes the demo corpus: {image}")
            if not image.is_file():
                return False
    return True


def main():
    config = load_street_config()
    run_root = run_dir(config["object_id"])
    spec = street_run_spec(config)
    spec_path = run_root / "run_spec.json"
    if spec_path.exists():
        if json.loads(spec_path.read_text(encoding="utf-8")) != spec:
            raise ValueError(f"Street scene or settings changed; choose a new run ID: {run_root}")
    else:
        if run_root.exists() and any(run_root.iterdir()):
            raise ValueError(f"Existing street run has no run specification: {run_root}")
        run_root.mkdir(parents=True, exist_ok=True)
        spec_path.write_text(json.dumps(spec, indent=2), encoding="utf-8")

    object_dir = run_root / "views" / config["object_id"]
    blender = find_blender()
    if not corpus_ready(run_root, config["object_id"]):
        if (run_root / "views").exists() or (run_root / "results").exists() or (run_root / "reconstructions").exists():
            raise RuntimeError("Incomplete street run; choose a new run ID instead of overwriting it")
        subprocess.run([
            blender, "--background", spec["source_scene"], "--python-exit-code", "1",
            "--python", str(COMPLEX / "render_street_corpus.py"),
        ], check=True)
        if not corpus_ready(run_root, config["object_id"]):
            raise RuntimeError("Street rendering did not produce a complete corpus")
    else:
        print(f"Reusing rendered street corpus: {object_dir}", flush=True)

    overview = run_root / "camera_overview.blend"
    image = run_root / "camera_overview.png"
    if overview.is_file() and image.is_file():
        print(f"Preserving existing overview: {overview}", flush=True)
    elif overview.exists() or image.exists():
        raise RuntimeError("Incomplete overview; preserve it and choose a new run ID")
    else:
        subprocess.run([
            blender, "--background", str(object_dir / "grounded_scene.blend"),
            "--python-exit-code", "1", "--python", str(COMPLEX / "render_overview.py"),
        ], check=True)
    if not overview.is_file() or not image.is_file():
        raise RuntimeError("Street overview renderer did not produce both expected files")
    print(f"Street camera views ready: {run_root}", flush=True)


if __name__ == "__main__":
    main()
