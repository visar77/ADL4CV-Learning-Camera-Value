"""Score individual camera acquisitions with upstream FisherRF on saved FastGS bases."""
import argparse
import ast
import json
import math
from pathlib import Path
from statistics import mean
import subprocess
import sys
from time import perf_counter
from types import SimpleNamespace

import torch


DEFAULT_FISHERRF = Path(__file__).resolve().parents[3] / "FisherRF"


def read_json(path):
    return json.loads(path.read_text())


def actions(reconstructions):
    base = reconstructions["base"]["train_lattice_indices"]
    base_set = set(base)
    candidates = {}
    for name, run in reconstructions.items():
        if name == "base" or run["status"] != "complete":
            continue
        views = set(run["train_lattice_indices"])
        new = views - base_set
        if views == base_set:
            continue
        if not base_set <= views or len(new) != 1:
            raise ValueError(f"{name} is not a base-plus-one reconstruction")
        candidates.setdefault(new.pop(), []).append(name)
    return base, candidates


def make_camera(frame, fov_x, width, height):
    from scene.cameras import MiniCam
    from utils.graphics_utils import getProjectionMatrix

    c2w = torch.tensor(frame["transform_matrix"], dtype=torch.float64)
    c2w[:3, 1:3] *= -1  # Blender to the camera axes expected by the rasterizer.
    view = c2w.inverse().T.float().contiguous().cuda()
    fov_y = 2 * math.atan(math.tan(fov_x / 2) * height / width)
    proj = getProjectionMatrix(0.01, 100., fov_x, fov_y).T.cuda()
    return MiniCam(width, height, fov_y, fov_x, .01, 100., view, view @ proj)


def load_base(folder, iteration):
    from scene.gaussian_model import GaussianModel

    # FastGS saves repr(argparse.Namespace). Never execute the stored arguments.
    config = ast.parse((folder / "cfg_args").read_text(), mode="eval").body
    args = {k.arg: ast.literal_eval(k.value) for k in config.keywords
            if k.arg in {"sh_degree", "white_background", "resolution"}}
    if args.get("resolution", -1) not in (-1, 1):
        raise ValueError("This adapter expects native-resolution FastGS checkpoints")
    model = GaussianModel(args["sh_degree"])
    model.load_ply(str(folder / "point_cloud" / f"iteration_{iteration}" / "point_cloud.ply"))
    background = torch.full((3,), float(args["white_background"]), device="cuda")
    return model, background


def diagonal(camera, model, pipeline, background):
    from gaussian_renderer import modified_render

    # Its custom backward computes the diagonal information terms. Ordinary
    # rendering gradients (including their squares) would give another score.
    image = modified_render(camera, model, pipeline, background)["render"]
    params = (model._xyz, model._features_dc, model._features_rest, model._scaling, model._opacity)
    terms = torch.autograd.grad(image, params, grad_outputs=torch.ones_like(image))
    return torch.cat([term.reshape(-1) for term in terms])


def acquisition_scores(base, candidates, information, clock):
    start = clock()
    total = information(base[0])
    for index in base[1:]:
        total += information(index)
    inverse = (total + 1e-6).reciprocal()
    prepared = clock()
    scores = {index: (information(index) * inverse).sum().item() for index in candidates}
    end = clock()
    if not all(math.isfinite(s) for s in scores.values()):
        raise ValueError("FisherRF produced a nonfinite score")
    return scores, {"base_information": prepared - start, "candidate_scoring": end - prepared}


def evaluate(scores, aliases, metrics):
    best = max(scores.values())
    selected = min(index for index, score in scores.items() if score == best)
    psnr_base = metrics["base"]["averages"]["PSNR"]
    rows = []
    for index, score in scores.items():
        gain = mean(metrics[name]["averages"]["PSNR"] for name in aliases[index]) - psnr_base
        rows.append(dict(camera=index, aliases=aliases[index], score=score, gain_db=gain))
    selected_gain = next(row["gain_db"] for row in rows if row["camera"] == selected)
    return dict(candidates=rows, selected_camera=selected, selected_gain_db=selected_gain,
                regret_db=max(row["gain_db"] for row in rows) - selected_gain)


def cuda_clock():
    torch.cuda.synchronize()
    return perf_counter()


def score_context(root, path, result, base, candidates):
    scene, experiment = path.parent.parent.name, path.parent.name
    folder = root / "reconstructions" / scene / experiment / "base"
    started = cuda_clock()
    model, bg = load_base(folder, result["reconstructions"]["base"]["iterations"])
    loaded = cuda_clock()
    transforms = read_json(root / "views" / scene / "global_transforms" / "transforms_train.json")
    (size,) = {(cam["width"], cam["height"]) for cam in read_json(folder / "cameras.json")}
    cameras = {}
    for i in [*base, *candidates]:
        if not 0 <= i < len(transforms["frames"]):
            raise ValueError(f"Invalid physical camera index {i} for {scene}")
        cameras[i] = make_camera(transforms["frames"][i], transforms["camera_angle_x"], *size)
    pipe = SimpleNamespace(debug=False, compute_cov3D_python=False, convert_SHs_python=False)
    def information(i):
        return diagonal(cameras[i], model, pipe, bg)
    scores, seconds = acquisition_scores(base, candidates, information, cuda_clock)
    seconds.update(checkpoint_loading=loaded - started, total=cuda_clock() - started)
    return dict(scene=scene, experiment=experiment, seconds=seconds,
                **evaluate(scores, candidates, result["metrics"]))


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("run_root", type=Path)
    parser.add_argument("--fisherrf", type=Path, default=DEFAULT_FISHERRF,
                        help="FisherRF checkout (default: FisherRF/ in the project root)")
    parser.add_argument("--output", required=True, type=Path)
    parser.add_argument("--object")
    parser.add_argument("--experiment")
    args = parser.parse_args()
    upstream = args.fisherrf.resolve()
    if not (upstream / "active/H_reg.py").is_file():
        parser.error(f"FisherRF checkout missing at {upstream}. From the project root run "
                     "'git submodule update --init --recursive -- FisherRF' or supply --fisherrf.")
    if not torch.cuda.is_available():
        parser.error("FisherRF requires CUDA and its compiled rasterizers")
    # Upstream exposes gaussian_renderer, scene and utils as top-level modules.
    sys.path.insert(0, str(upstream))
    from gaussian_renderer import modified_render  # Verify the extension before processing contexts.

    contexts, skipped = [], []
    for path in sorted((args.run_root / "results").glob("*/*/results.json")):
        scene, experiment = path.parent.parent.name, path.parent.name
        if (args.object and args.object != scene) or (args.experiment and args.experiment != experiment):
            continue
        result = read_json(path)
        runs = result["reconstructions"]
        if runs["base"]["status"] != "complete":
            skipped.append(dict(scene=scene, experiment=experiment, reason="incomplete base"))
            continue
        base, candidates = actions(runs)
        if not base or not candidates:
            skipped.append(dict(scene=scene, experiment=experiment, reason="empty base or candidate set"))
            continue
        contexts.append(score_context(args.run_root, path, result, base, candidates))
        print(f"{scene} / {experiment}: {contexts[-1]['regret_db']:.3f} dB regret", flush=True)
    if not contexts and not skipped:
        parser.error("No contexts matched the supplied run root and filters")
    revision = subprocess.check_output(["git", "-C", str(upstream), "rev-parse", "HEAD"], text=True).strip()
    report = dict(fisherrf_revision=revision, gpu=torch.cuda.get_device_name(),
                  contexts=contexts, skipped=skipped)
    payload = json.dumps(report, indent=2, allow_nan=False)
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(payload + "\n")


if __name__ == "__main__":
    main()
