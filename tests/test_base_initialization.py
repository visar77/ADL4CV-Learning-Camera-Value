"""Exercise the real orchestrator with a tiny subprocess standing in for CUDA."""

import contextlib
import io
import json
import sys
import tempfile
import unittest
from pathlib import Path
from unittest.mock import patch

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
import orchestrator as pipeline
import evaluate_metrics as evaluation


FAKE_TRAIN = '''
import argparse, json
from pathlib import Path
p = argparse.ArgumentParser()
p.add_argument('--source_path'); p.add_argument('--model_path'); p.add_argument('--iterations', type=int)
p.add_argument('--init_ply'); p.add_argument('--test_iterations', nargs='*')
a, unknown = p.parse_known_args()
source, model = Path(a.source_path), Path(a.model_path)
manifest = json.loads((source / 'transforms_train.json').read_text())
with (Path(__file__).parent / 'calls.jsonl').open('a') as f:
    f.write(json.dumps({'model': str(model), 'init': a.init_ply, 'frames': manifest['frames'],
                        'iterations': a.iterations, 'tests': a.test_iterations}) + '\\n')
if (Path(__file__).parent / 'fail').exists():
    raise RuntimeError('simulated GPU failure')
if a.init_ply:
    assert Path(a.init_ply).is_file()
ply = model / 'point_cloud' / f'iteration_{a.iterations}' / 'point_cloud.ply'
ply.parent.mkdir(parents=True, exist_ok=True)
ply.write_text(json.dumps(manifest))
'''


class PipelineTests(unittest.TestCase):
    def setUp(self):
        self.temp = tempfile.TemporaryDirectory()
        self.addCleanup(self.temp.cleanup)
        self.root = Path(self.temp.name)
        self.views = self.root / "views" / "object"
        manifests = self.views / "global_transforms"
        manifests.mkdir(parents=True)
        self.manifest = {"camera_angle_x": 0.8, "frames": [
            {"file_path": f"images/train/{i:03}", "transform_matrix": [[1, 0, 0, 0]] * 4}
            for i in range(100)]}
        for name in ("transforms_train.json", "transforms_test.json"):
            (manifests / name).write_text(json.dumps(self.manifest))
        for frame in self.manifest["frames"]:
            image = self.views / (frame["file_path"] + ".png")
            image.parent.mkdir(parents=True, exist_ok=True)
            image.write_bytes(b"test image fingerprint")
        self.experiments = self.root / "experiments"
        self.experiments.mkdir()
        experiment = {"id": "example", "train": {"strategy": "furthest", "count": 14, "seed": 77},
                      "candidates": [{"strategy": "blindspot"}, {"strategy": "random", "count": 6},
                                     {"strategy": "redundancy", "count": 2}, {"strategy": "next", "count": 3}]}
        (self.experiments / "example.json").write_text(json.dumps(experiment))
        self.fastgs = self.root / "FastGS"
        self.fastgs.mkdir()
        (self.fastgs / "train.py").write_text(FAKE_TRAIN)
        self.argv = ["--skip-render", "--views-root", str(self.views.parent),
                     "--experiments-dir", str(self.experiments), "--fastgs-dir", str(self.fastgs),
                     "--reconstructions-root", str(self.root / "reconstructions"),
                     "--results-root", str(self.root / "results"),
                     "--candidate-initialization", "base", "--candidate-iterations", "300",
                     "--gain-reference", "equal-budget-base-control"]
        self.result_path = self.root / "results" / "base_init_300" / "object" / "example" / "results.json"

    def run_pipeline(self, *extra):
        with contextlib.redirect_stdout(io.StringIO()), patch.object(pipeline, "ADAPTER_SCRIPT", self.fastgs / "train.py"):
            pipeline.main(self.argv + list(extra))

    def calls(self):
        return [json.loads(line) for line in (self.fastgs / "calls.jsonl").read_text().splitlines()]

    def test_shared_initializer_independent_candidates_control_and_duplicate_reuse(self):
        self.run_pipeline()
        calls = self.calls()
        self.assertEqual(len(calls), 11)  # initializer, control, nine unique candidates
        self.assertEqual(calls[0]["iterations"], 30000)
        self.assertIsNone(calls[0]["init"])
        initial = str(self.root / "reconstructions/object/example/base/point_cloud/iteration_30000/point_cloud.ply")
        for call in calls[1:]:
            self.assertEqual(call["init"], initial)
            self.assertEqual(call["iterations"], 300)
            self.assertEqual(call["tests"], [])
        self.assertEqual(len(calls[1]["frames"]), 14)
        self.assertTrue(all(len(c["frames"]) == 15 for c in calls[2:]))
        result = pipeline.load_json(self.result_path)
        self.assertEqual(result["status"], "complete")
        self.assertEqual(result["protocol"]["gain_reference"], "equal_budget_base_control")
        duplicate = result["reconstructions"]["random-6-4"]
        self.assertEqual(duplicate["alias_of"], "blindspot-1-1")
        self.assertEqual(duplicate["status"], "complete")
        self.assertEqual(result["reconstructions"]["redundancy-2-1"]["status"], "redundant")
        self.assertFalse((self.views / "transforms_train.json").exists())
        self.run_pipeline()
        self.assertEqual(len(self.calls()), 11)

    def test_changed_schedule_invalidates_completion_and_metrics_but_reuses_initializer(self):
        self.run_pipeline("--candidate-limit", "1")
        result = pipeline.load_json(self.result_path)
        result["metrics"] = {"base": {"averages": {"PSNR": 20}, "per_view": {}}}
        result["metrics_iterations"] = {"base": 300}
        pipeline.save_json(self.result_path, result)
        self.run_pipeline("--candidate-limit", "1")
        self.assertEqual(len(self.calls()), 3)
        self.assertTrue(pipeline.load_json(self.result_path)["metrics"]["base"])
        self.run_pipeline("--candidate-limit", "1", "--adaptation-lr-scale", "0.2")
        self.assertEqual(len(self.calls()), 5)
        self.assertEqual(pipeline.load_json(self.result_path)["metrics"], {})

    def test_existing_legacy_base_is_reused_without_glb(self):
        legacy = self.root / "reconstructions/object/example/base"
        ply = pipeline.point_cloud_path(legacy, 30000)
        ply.parent.mkdir(parents=True)
        ply.write_text("legacy Gaussian PLY")
        self.run_pipeline("--candidate-limit", "1")
        self.assertEqual(len(self.calls()), 2)
        self.assertTrue(all(call["init"] == str(ply) for call in self.calls()))
        self.assertEqual(ply.read_text(), "legacy Gaussian PLY")

    def test_failure_is_recorded_and_temporary_manifest_is_removed(self):
        self.run_pipeline("--candidate-limit", "0")
        (self.fastgs / "fail").touch()
        with self.assertRaises(Exception):
            self.run_pipeline("--candidate-limit", "1")
        self.assertEqual(pipeline.load_json(self.result_path)["status"], "failed")
        self.assertFalse((self.views / "transforms_train.json").exists())
        self.assertFalse(pipeline.reconstruction_complete(
            self.root / "reconstructions/base_init_300/object/example/blindspot-1-1", 300))

    def test_dry_run_writes_nothing(self):
        before = sorted(str(p.relative_to(self.root)) for p in self.root.rglob("*"))
        self.run_pipeline("--dry-run", "--candidate-limit", "1")
        after = sorted(str(p.relative_to(self.root)) for p in self.root.rglob("*"))
        self.assertEqual(before, after)

    def test_scratch_budgets_are_independent_and_default_mode_is_unchanged(self):
        args = pipeline.parse_args([])
        self.assertEqual((args.candidate_initialization, args.base_iterations, args.candidate_iterations),
                         ("scratch", 30000, 30000))
        self.run_pipeline("--candidate-initialization", "scratch", "--gain-reference", "original-base", "--base-iterations", "12",
                          "--candidate-iterations", "7", "--candidate-limit", "1")
        calls = self.calls()
        self.assertEqual([c["iterations"] for c in calls], [12, 7])
        self.assertTrue(all(c["init"] is None for c in calls))

    def test_metrics_uses_recorded_iterations_and_deduplicates_evaluation(self):
        self.run_pipeline()
        metric = {"averages": {"PSNR": 20.0}, "per_view": {"00000": {"PSNR": 20.0}}}
        with patch.object(evaluation, "evaluate_reconstruction", return_value=metric) as evaluate:
            with contextlib.redirect_stdout(io.StringIO()):
                evaluation.main(["--results-root", str(self.result_path.parents[2]),
                                 "--views-root", str(self.views.parent),
                                 "--reconstructions-root", str(self.root / "reconstructions/base_init_300")])
            self.assertEqual(evaluate.call_count, 10)
            self.assertTrue(all(call.args[3] == 300 for call in evaluate.call_args_list))
            self.assertTrue(all(call.args[1] == self.views for call in evaluate.call_args_list))
        result = pipeline.load_json(self.result_path)
        self.assertEqual(len(result["metrics"]), 13)
        self.assertTrue(all(i == 300 for i in result["metrics_iterations"].values()))
        with patch.object(evaluation, "evaluate_reconstruction") as evaluate:
            with contextlib.redirect_stdout(io.StringIO()):
                evaluation.main(["--results-root", str(self.result_path.parents[2])])
            evaluate.assert_not_called()

    def test_metrics_supports_different_scratch_budgets(self):
        self.run_pipeline("--candidate-initialization", "scratch", "--gain-reference", "original-base", "--base-iterations", "12",
                          "--candidate-iterations", "7", "--candidate-limit", "1")
        root = self.root / "results"
        metric = {"averages": {"PSNR": 20.0}, "per_view": {}}
        with patch.object(evaluation, "evaluate_reconstruction", return_value=metric) as evaluate:
            evaluation.main(["--results-root", str(root)])
        self.assertEqual([call.args[3] for call in evaluate.call_args_list], [12, 7])
        result = pipeline.load_json(root / "object/example/results.json")
        self.assertEqual(result["metrics_iterations"]["redundancy-2-1"], 12)
        self.assertNotIn("metrics_iteration", result)


if __name__ == "__main__":
    unittest.main()
