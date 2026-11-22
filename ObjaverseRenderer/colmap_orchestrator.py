"""Train FastGS bases initialized by COLMAP triangulation of their posed views."""

import argparse
from functools import partial
import math
from pathlib import Path
from tempfile import TemporaryDirectory

import orchestrator


def verify_matches_from_poses(database, poses, min_matches):
    import numpy as np
    import pycolmap

    cameras = database.read_all_cameras()
    inverse_k = np.linalg.inv(cameras[0].calibration_matrix())
    world_to_camera = {image.image_id: poses[image.name] for image in database.read_all_images()}
    pair_ids, pair_matches = database.read_all_matches()
    database.clear_two_view_geometries()
    verified_pairs = 0
    for pair_id, matches in zip(pair_ids, pair_matches):
        if len(matches) < min_matches:
            continue
        first, second = pycolmap.pair_id_to_image_pair(pair_id)
        relative_pose = world_to_camera[second] * world_to_camera[first].inverse()
        essential = pycolmap.essential_matrix_from_pose(relative_pose)
        fundamental = inverse_k.T @ essential @ inverse_k
        first_xy = database.read_keypoints(first)[matches[:, 0], :2]
        second_xy = database.read_keypoints(second)[matches[:, 1], :2]
        errors = pycolmap.compute_squared_sampson_error(first_xy, second_xy, fundamental)
        # Screen tentative descriptor matches with a two-pixel Sampson threshold.
        inliers = matches[np.asarray(errors) <= 2.0 ** 2]
        if len(inliers) < min_matches:
            continue
        geometry = pycolmap.TwoViewGeometry(
            config=pycolmap.TwoViewGeometryConfiguration.CALIBRATED.value,
            E=essential, F=fundamental, cam2_from_cam1=relative_pose, inlier_matches=inliers,
        )
        database.write_two_view_geometry(first, second, geometry)
        verified_pairs += 1
    return verified_pairs


def triangulate_subset(object_dir, indices, work_dir):
    import numpy as np
    from PIL import Image

    try:
        import pycolmap
    except ImportError as error:
        raise RuntimeError("Install PyCOLMAP with: python -m pip install pycolmap==3.13.0") from error

    if len(indices) < 2:
        raise RuntimeError("COLMAP needs at least two base views to triangulate a point cloud.")
    manifest = orchestrator.load_json(object_dir / "global_transforms/transforms_train.json")
    masks = work_dir / "masks"
    poses = {}
    size = None
    for index in indices:
        frame = manifest["frames"][index]
        name = Path(frame["file_path"]).with_suffix(".png")
        with Image.open(object_dir / name) as image:
            if size is None:
                size = image.size
            if image.size != size:
                raise RuntimeError("COLMAP base views must share one image size.")
            mask = masks / (name.as_posix() + ".png")
            mask.parent.mkdir(parents=True, exist_ok=True)
            image.convert("RGBA").getchannel("A").save(mask)
        # Blender camera-to-world uses Y up and looks along -Z. COLMAP uses
        # Y down and +Z forward, with a world-to-camera transform.
        c2w = np.array(frame["transform_matrix"], dtype=float)
        c2w[:3, 1:3] *= -1
        poses[name.as_posix()] = pycolmap.Rigid3d(np.linalg.inv(c2w)[:3])

    width, height = size
    focal = width / (2 * math.tan(manifest["camera_angle_x"] / 2))
    database_path = str(work_dir / "database.db")
    # Keep more tentative matches for small subsets, then verify with the
    # known cameras below.
    extraction = pycolmap.FeatureExtractionOptions()
    extraction.sift.peak_threshold = 0.003
    matching = pycolmap.FeatureMatchingOptions()
    matching.sift.max_ratio = 0.88
    pycolmap.extract_features(
        database_path, str(object_dir), image_names=list(poses),
        camera_mode=pycolmap.CameraMode.SINGLE, camera_model="PINHOLE",
        reader_options=pycolmap.ImageReaderOptions(
            camera_params=f"{focal},{focal},{width / 2},{height / 2}", mask_path=str(masks)
        ),
        extraction_options=extraction,
        device=pycolmap.Device.cpu,
    )
    pycolmap.match_exhaustive(database_path, matching_options=matching, device=pycolmap.Device.cpu)

    options = pycolmap.IncrementalPipelineOptions(fix_existing_frames=True, min_num_matches=4)
    options.triangulation.ignore_two_view_tracks = False
    reconstruction = pycolmap.Reconstruction()
    with pycolmap.Database.open(database_path) as database:
        # The supplied calibration gives the epipolar geometry even for pairs
        # with too few SIFT matches to estimate that geometry reliably.
        if not verify_matches_from_poses(database, poses, options.min_num_matches):
            raise RuntimeError(
                f"COLMAP found no image pair with enough verified matches "
                f"for {object_dir.name} views {indices}."
            )
        for camera in database.read_all_cameras():
            reconstruction.add_camera(camera)
        for rig in database.read_all_rigs():
            reconstruction.add_rig(rig)
        for frame in database.read_all_frames():
            image = database.read_image(frame.image_ids[0].id)
            reconstruction.add_frame(frame)
            image.frame_id = frame.frame_id
            reconstruction.add_image(image)
            reconstruction.frame(frame.frame_id).set_cam_from_world(image.camera_id, poses[image.name])
            reconstruction.register_frame(frame.frame_id)

    sparse = work_dir / "triangulated"
    sparse.mkdir()
    cloud = pycolmap.triangulate_points(
        reconstruction, database_path, str(object_dir), str(sparse), options=options
    )
    points = list(cloud.points3D.values())
    # FastGS initializes each Gaussian's scale from its three nearest neighbors.
    if len(points) < 4:
        raise RuntimeError(
            f"COLMAP triangulated {len(points)} points for {object_dir.name} views {indices}; "
            "FastGS initialization requires at least 4."
        )
    ply = work_dir / "points3d.ply"
    with ply.open("w", encoding="ascii") as file:
        file.write(f"ply\nformat ascii 1.0\nelement vertex {len(points)}\n")
        file.write("property float x\nproperty float y\nproperty float z\n")
        file.write("property float nx\nproperty float ny\nproperty float nz\n")
        file.write("property uchar red\nproperty uchar green\nproperty uchar blue\nend_header\n")
        for point in points:
            x, y, z = point.xyz
            red, green, blue = point.color
            file.write(f"{x} {y} {z} 0 0 0 {red} {green} {blue}\n")
    print(f"Triangulated {len(points)} COLMAP points for {object_dir.name} views {indices}.", flush=True)


def train_with_colmap(object_dir, model_dir, indices, iterations, init_ply, retrain, *, train):
    if model_dir.name != "base":
        return train(object_dir, model_dir, indices, iterations, init_ply, retrain)
    checkpoint = orchestrator.checkpoint_path(model_dir, iterations)
    if checkpoint.is_file() and not retrain:
        return checkpoint
    model_dir.parent.mkdir(parents=True, exist_ok=True)
    with TemporaryDirectory(prefix=".colmap-", dir=model_dir.parent) as temporary:
        dataset = Path(temporary)
        triangulate_subset(object_dir, indices, dataset)
        # The native Blender dataset reader initializes from points3d.ply.
        # Stage it alongside links so each subset has its own seed cloud.
        for name in ("images", "global_transforms"):
            (dataset / name).symlink_to(object_dir / name, target_is_directory=True)
        return train(dataset, model_dir, indices, iterations, None, retrain)


def add_run_root_argument(parser):
    parser.add_argument("--run-root", type=Path, help="Override the configured input run directory.")
    return parser


def main(argv=None):
    options = add_run_root_argument(argparse.ArgumentParser(add_help=False, allow_abbrev=False))
    args, forwarded = options.parse_known_args(argv)
    if "-h" in forwarded or "--help" in forwarded:
        parser = add_run_root_argument(orchestrator.build_parser())
        parser.description = __doc__
        parser.epilog = (
            "Install pycolmap==3.13.0. Outputs are saved under <run_root>/colmap/, "
            "with the initialization cloud in each base/input.ply."
        )
        parser.print_help()
        return

    config = orchestrator.load_config()
    if args.run_root is not None:
        config["run_root"] = args.run_root.expanduser().resolve()
        raw = orchestrator.load_json(orchestrator.CONFIG_PATH)
        for key in ("objects_root", "views_root"):
            config[key] = orchestrator.resolve_config_path(raw[key], config["run_root"])
    output_root = config["run_root"] / "colmap"
    config.update(run_root=output_root, reconstructions_root=output_root / "reconstructions",
                  results_root=output_root / "results")

    # Reuse the existing CLI and job handling with a base initializer scoped
    # to this entry point.
    original_config = orchestrator.CONFIG
    original_train = orchestrator.run_training
    try:
        orchestrator.CONFIG = config
        orchestrator.run_training = partial(train_with_colmap, train=original_train)
        orchestrator.main(forwarded)
    finally:
        orchestrator.CONFIG = original_config
        orchestrator.run_training = original_train


if __name__ == "__main__":
    main()
