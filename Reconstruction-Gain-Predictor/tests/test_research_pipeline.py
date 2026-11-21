from dataclasses import replace
from pathlib import Path
from unittest.mock import patch
import json
import tempfile
import unittest

from PIL import Image
import torch

from gain_predictor.checkpoints import load_run
from gain_predictor.data.batches import collate_gain_samples, prepare_images
from gain_predictor.data.manifest import build_manifest, load_manifest
from gain_predictor.data.records import GainSample, audit_samples, context_key, deduplicate_samples, target_values
from gain_predictor.data.splits import make_splits, restored_split
from gain_predictor.feature_cache import FeatureStore, extract_features
from gain_predictor.geometry import apply_frame, context_frame, rays, to_opencv
from gain_predictor.metrics import selection_metrics
from gain_predictor.models.backbone import FeatureBackbone
from gain_predictor.models.lvsm import _LVSMSceneEncoder
from gain_predictor.scoring import score_samples
from gain_predictor.training import train
from tests.verify_lvsm import reference_forward
from train import parse_args


def example(root, context=0, candidate=0, gain=.1):
    image = root / f"view{context}.png"
    if not image.exists():
        Image.new("RGB", (12, 16), color=(80 + context, 50, 0)).save(image)
    pose = torch.eye(4)
    pose[0, 3] = context
    query = pose.clone()
    query[1, 3] = candidate
    return GainSample("object", f"context{context}", f"candidate{candidate}", (str(image),),
                      (tuple(pose.flatten().tolist()),), tuple(query.flatten().tolist()), gain,
                      "source.json", 20., 20.*(1+gain), ((1., 1., .5, .5),), (1., 1., .5, .5), "opencv")


def tiny_encoder():
    return _LVSMSceneEncoder(4, dimension=32, head_dimension=8, encoder_depth=2,
                             decoder_depth=2, latent_capacity=8).eval().requires_grad_(False)


class TinyBackbone(FeatureBackbone):
    feature_dim = 3
    encoded = 0
    decoded = 0

    def encode_context(self, images, context_poses, mask, context_intrinsics):
        type(self).encoded += 1
        return {"mean": images.float().mean().reshape(1, 1)}

    def decode_candidates(self, context, candidate_poses, candidate_intrinsics):
        type(self).decoded += candidate_poses.shape[1]
        return torch.stack((candidate_poses[..., 7], candidate_poses[..., 3],
                            context["mean"].expand(candidate_poses.shape[:2])), -1)


class CameraTests(unittest.TestCase):
    def test_blender_center_ray_and_calibrated_fov(self):
        pose = torch.eye(4).reshape(1, 1, 4, 4)
        pose[..., 2, 3] = 3
        cv = to_opencv(pose, "blender")
        k = torch.tensor([1., 1., .5, .5]).reshape(1, 1, 4)
        origin, direction = rays(cv, k, 3, 3)
        torch.testing.assert_close(direction[0, 0, :, 1, 1], torch.tensor([0., 0., -1.]))
        k2 = k.clone(); k2[..., :2] *= 2
        _, narrow = rays(cv, k2, 3, 3)
        self.assertLess(abs(float(narrow[0, 0, 0, 1, 0])), abs(float(direction[0, 0, 0, 1, 0])))
        torch.testing.assert_close(origin[0, 0, :, 1, 1], torch.tensor([0., 0., 3.]))

    def test_context_normalization_global_rigid_and_scale_invariance(self):
        poses = torch.eye(4).repeat(1, 2, 1, 1)
        poses[0, 1, 0, 3] = 2
        query = poses[:, :1].clone(); query[..., 1, 3] = 2
        mask = torch.ones(1, 2, dtype=torch.bool)
        frame, scale = context_frame(poses, mask)
        rotation = torch.tensor([[0., -1., 0.], [1., 0., 0.], [0., 0., 1.]])
        def transform(p):
            result = p.clone()
            result[..., :3, :3] = rotation @ p[..., :3, :3]
            result[..., :3, 3] = (rotation @ p[..., :3, 3, None]).squeeze(-1) * 3 + 5
            return result
        changed = transform(poses)
        f2, s2 = context_frame(changed, mask)
        torch.testing.assert_close(apply_frame(poses, frame, scale), apply_frame(changed, f2, s2))
        torch.testing.assert_close(apply_frame(query, frame, scale), apply_frame(transform(query), f2, s2))
        self.assertTrue(torch.isfinite(context_frame(poses[:, :1], mask[:, :1])[1]).all())

    def test_uint8_dark_pixels_are_scaled(self):
        image = torch.ones(1, 1, 3, 8, 8, dtype=torch.uint8)
        self.assertAlmostEqual(prepare_images(image, 8).max().item(), 1/255)


class BackboneTests(unittest.TestCase):
    def setUp(self):
        torch.manual_seed(3)
        self.model = tiny_encoder()
        self.images = torch.rand(1, 1, 3, 16, 16)
        self.poses = torch.eye(4).flatten().reshape(1, 1, 16)
        self.k = torch.tensor([1., 1., .5, .5]).reshape(1, 1, 4)

    def test_padding_invariance_with_arbitrary_padding(self):
        mask = torch.ones(1, 1, dtype=torch.bool)
        actual = self.model(self.images, self.poses, self.poses[:, 0], mask, self.k, self.k[:, 0])
        images = torch.cat((self.images, torch.rand_like(self.images)), 1)
        poses = torch.cat((self.poses, torch.zeros_like(self.poses)), 1)
        padded = self.model(images, poses, self.poses[:, 0], torch.tensor([[True, False]]),
                            torch.cat((self.k, torch.zeros_like(self.k)), 1), self.k[:, 0])
        torch.testing.assert_close(actual, padded, rtol=2e-5, atol=2e-5)

    def test_independent_official_equations_and_candidate_batching(self):
        mask = torch.ones(1, 1, dtype=torch.bool)
        actual = self.model(self.images, self.poses, self.poses[:, 0], mask, self.k, self.k[:, 0])
        expected = reference_forward(self.model, self.images, self.poses, self.poses[:, 0], mask, self.k, self.k[:, 0])
        torch.testing.assert_close(actual, expected, rtol=2e-5, atol=2e-5)
        context = self.model.encode_context(self.images, self.poses, mask, self.k)
        poses = self.poses.expand(-1, 2, -1)
        batch = self.model.decode_candidates(context, poses, self.k.expand(-1, 2, -1), 16, 16)
        torch.testing.assert_close(batch[:, 0], actual)
        torch.testing.assert_close(batch[:, 1], actual)

    def test_incomplete_wrong_shape_and_unknown_checkpoint_rejected(self):
        with tempfile.TemporaryDirectory() as d:
            path = Path(d)/"model.pt"
            state = self.model.state_dict()
            missing = dict(state); missing.pop("transformer_encoder.0.norm1.weight")
            wrong = dict(state); wrong["image_tokenizer.1.weight"] = torch.ones(1)
            unknown = {**state, "typo.weight": torch.ones(1)}
            for bad in (missing, wrong, unknown):
                torch.save({"model": bad}, path)
                with self.assertRaises(ValueError):
                    self.model.load_checkpoint(path)
            torch.save({"model": {**state, "image_token_decoder.0.weight": torch.ones(1)}}, path)
            self.model.load_checkpoint(path)


class DatasetAndSplitTests(unittest.TestCase):
    def test_duplicate_controls_audit_and_context_grouping(self):
        with tempfile.TemporaryDirectory() as d:
            root = Path(d)
            a = example(root, gain=.1)
            b = replace(a, experiment_id="alias", candidate_id="duplicate", psnr_gain=.3, candidate_psnr=26.)
            merged = deduplicate_samples([a, b])
            self.assertEqual(len(merged), 1)
            self.assertAlmostEqual(merged[0].psnr_gain, .2)
            self.assertEqual(len(merged[0].observed_gains), 2)
            self.assertAlmostEqual(audit_samples([a,b])["duplicates"][0]["gain_range"], .2)
            self.assertEqual(context_key(a), context_key(b))

    def test_single_object_context_splits_and_explicit_overfit(self):
        with tempfile.TemporaryDirectory() as d:
            root = Path(d)
            samples = [example(root, c, q) for c in range(5) for q in range(3)]
            groups, metadata = make_splits(samples)
            self.assertEqual([len(groups[k]) for k in ("train", "validation", "test")], [9,3,3])
            keys = [{context_key(s) for s in g} for g in groups.values()]
            self.assertFalse(keys[0] & keys[1] or keys[0] & keys[2] or keys[1] & keys[2])
            with self.assertRaises(ValueError):
                make_splits(samples, mode="scene")
            overfit, _ = make_splits(samples[:1], mode="overfit")
            self.assertFalse(overfit["validation"] or overfit["test"])
            two = [replace(samples[0], scene_id=str(i)) for i in range(2)]
            with self.assertRaises(ValueError):
                make_splits(two, mode="scene")
            groups, _ = make_splits(two, mode="scene", test_fraction=0)
            self.assertEqual(len(groups["validation"]), 1)

    def test_regret_pairwise_and_constant_tie_fairness(self):
        with tempfile.TemporaryDirectory() as d:
            root = Path(d)
            samples = [example(root, candidate=i, gain=v) for i,v in enumerate((0., .1, .3))]
            good = selection_metrics(samples, [0.,1.,2.])
            self.assertEqual(good["ranking"]["regret"], 0.)
            self.assertEqual(good["ranking"]["pairwise_accuracy"], 1.)
            tied = selection_metrics(samples, [0.,0.,0.])
            self.assertAlmostEqual(tied["ranking"]["selected_gain"], tied["ranking"]["random_gain"])
            self.assertEqual(tied["ranking"]["pairwise_accuracy"], .5)

    def test_resized_rgb_cache_refreshes_changed_files(self):
        with tempfile.TemporaryDirectory() as d:
            sample = example(Path(d))
            first = collate_gain_samples([sample], 8)["images"]
            import os
            image = Path(sample.context_image_paths[0])
            old_time = image.stat().st_mtime_ns
            Image.new("RGB", (12, 16), color="red").save(image)
            os.utime(image, ns=(old_time + 1_000_000, old_time + 1_000_000))
            second = collate_gain_samples([sample], 8)["images"]
            self.assertFalse(torch.equal(first, second))
            torch.testing.assert_close(second[0, 0, :, 0, 0], torch.tensor([255, 0, 0], dtype=torch.uint8))

    def test_builder_preserves_controls_intrinsics_and_uses_global_frames(self):
        with tempfile.TemporaryDirectory() as d:
            root = Path(d)
            views = root/"views"/"object"
            (views/"global_transforms").mkdir(parents=True)
            for i in range(2):
                Image.new("RGB", (12,16)).save(views/f"{i}.png")
            frames = [{"file_path": str(i), "transform_matrix": torch.eye(4).tolist()} for i in range(2)]
            payload = {"camera_angle_x": .8, "frames": frames}
            (views/"global_transforms"/"transforms_train.json").write_text(json.dumps(payload))
            (views/"transforms_train.json").write_text(json.dumps({"frames": []}))
            result = root/"results"; result.mkdir()
            raw = {"object_id":"object", "experiment_id":"e", "train_lattice_indices":[0],
                   "metrics": {"base":{"averages":{"PSNR":20}}, "control":{"averages":{"PSNR":20}}},
                   "reconstructions":{"control":{"status":"redundant", "candidate_lattice_index":1}}}
            (result/"results.json").write_text(json.dumps(raw))
            output = root/"manifest.jsonl"
            samples = build_manifest(result, root/"views", output)
            self.assertEqual(samples[0].label_kind, "redundancy_control")
            self.assertEqual(samples[0].psnr_gain, 0.)
            self.assertAlmostEqual(samples[0].context_intrinsics[0][0], 1/(2*__import__('math').tan(.4)))
            self.assertEqual(load_manifest(output), samples)
            self.assertTrue(output.with_suffix(".audit.json").is_file())


class CacheAndTrainingTests(unittest.TestCase):
    def test_legacy_labels_with_and_without_base_psnr(self):
        with tempfile.TemporaryDirectory() as directory:
            sample = example(Path(directory), gain=-.15)
            base_only = replace(sample, candidate_psnr=None)
            relative_only = replace(base_only, base_psnr=None)
            torch.testing.assert_close(target_values([sample, base_only], "delta-db"),
                                       torch.tensor([[-3.], [-3.]]))
            with self.assertRaisesRegex(ValueError, "--target relative"):
                target_values([relative_only], "delta-db")
            torch.testing.assert_close(target_values([relative_only], "relative"), torch.tensor([[-.15]]))

    def test_compare_evaluate_and_rank_cli_share_saved_splits(self):
        import contextlib
        import io
        import compare
        import evaluate
        import predict
        with tempfile.TemporaryDirectory() as d, contextlib.redirect_stdout(io.StringIO()):
            root = Path(d)
            samples = [example(root, c, q, .1*(q-1)) for c in range(5) for q in range(3)]
            manifest = root/"data.jsonl"
            manifest.write_text(''.join(json.dumps(s.to_json())+'\n' for s in samples))
            output = compare.main(["--dataset", str(manifest), "--mode", "context",
                                   "--models", "mean", "geometry",
                                   "--output-dir", str(root/"comparison")])
            report = json.loads(output.read_text())
            self.assertEqual(set(report["models"]), {"mean", "geometry"})
            model = root/"comparison"/"geometry"/"model.pt"
            evaluation = evaluate.main(["--checkpoint", str(model), "--split", "test"])
            self.assertEqual(evaluation["target"], "delta-db")
            self.assertEqual(evaluation["samples"], 3)
            ranked = predict.main(["--checkpoint", str(model), "--experiment-id", "context0", "--rank"])
            self.assertEqual(len(ranked), 3)
            self.assertLess(min(r["ground_truth_relative_gain"] for r in ranked), 0.)
            for row in ranked:
                self.assertEqual(row["target"], "delta-db")
                self.assertEqual(row["predicted_delta_db"], row["predicted_gain"])
            self.assertEqual([r["predicted_utility"] for r in ranked],
                             sorted([max(0., r["predicted_gain"]) for r in ranked], reverse=True))

    def test_cache_reuse_invalidation_and_head_only_training(self):
        with tempfile.TemporaryDirectory() as d:
            root = Path(d)
            samples = [example(root, c, q, gain=.1*q) for c in range(2) for q in range(3)]
            TinyBackbone.encoded = TinyBackbone.decoded = 0
            path = extract_features(samples, root/"cache", TinyBackbone, {"name":"tiny", "version":1},
                                    image_size=8, candidate_batch_size=2)
            self.assertEqual(TinyBackbone.encoded, 2)
            self.assertEqual(TinyBackbone.decoded, 6)
            store = FeatureStore(path)
            matrix = store.matrix(samples)
            def forbidden():
                raise AssertionError("A cached run must not initialize a backbone")
            extract_features(samples, root/"cache", forbidden, {"name":"tiny", "version":1}, image_size=8)
            torch.testing.assert_close(FeatureStore(path).matrix(samples), matrix)
            manifest = root/"data.jsonl"
            manifest.write_text(''.join(json.dumps(s.to_json())+'\n' for s in samples))
            args = parse_args(["--dataset",str(manifest),"--model","lvsm","--features",str(path),
                               "--mode","overfit","--epochs","150","--batch-size","6", "--learning-rate",".01",
                               "--scheduler","none","--early-stopping-patience","0", "--log-every","150",
                               "--output",str(root/"head.pt"),"--log-dir",str(root/"logs")])
            with patch("gain_predictor.models.lvsm.LVSMFeatureBackbone.__init__", side_effect=AssertionError("Backbone instantiated")):
                train(args)
                checkpoint, model = load_run(args.output)
                prediction = score_samples(checkpoint, model, samples, path)
            self.assertEqual(checkpoint["target"], "delta-db")
            expected = target_values(samples, "delta-db")
            self.assertLess(float((prediction - expected).square().mean()), 4e-3)
            self.assertLess(args.output.stat().st_size, 100_000)
            self.assertEqual(len(restored_split(load_manifest(manifest), checkpoint, "train")), 6)
            changed = [replace(samples[0], psnr_gain=.9), *samples[1:]]
            with self.assertRaises(ValueError):
                restored_split(changed, checkpoint, "train")
            Image.new("RGB", (12,16), color="red").save(samples[0].context_image_paths[0])
            with self.assertRaises(ValueError):
                store.matrix(samples)
            changed_camera = [replace(samples[0], candidate_intrinsics=(2.,2.,.5,.5))]
            with self.assertRaises(ValueError):
                store.matrix(changed_camera)


if __name__ == "__main__":
    unittest.main()
