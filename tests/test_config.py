"""Standalone JSON configuration, CLI precedence, and portable path handling."""

import contextlib
import io
import json
import os
import sys
import tempfile
import unittest
from pathlib import Path
from unittest.mock import patch

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from reconstruction_scripts import config


class ConfigTests(unittest.TestCase):
    def setUp(self):
        temporary = tempfile.TemporaryDirectory()
        self.addCleanup(temporary.cleanup)
        self.root = Path(temporary.name)
        self.path = self.root / "custom.json"

    def parse(self, values, *flags):
        self.path.write_text(json.dumps(values), encoding="utf-8")
        return config.parse_args(["--config", str(self.path), *flags])

    def test_shipped_defaults_preserve_existing_behavior(self):
        self.assertEqual(
            config.DEFAULT_CONFIG, config.ROOT / "reconstruction_scripts/config.json"
        )
        args = config.parse_args([])
        self.assertEqual(args.candidate_initialization, "scratch")
        self.assertEqual((args.base_iterations, args.candidate_iterations), (30000, 30000))
        self.assertEqual(args.python, sys.executable)
        for name, directory in (
            ("views_root", "views"), ("objects_root", "objects"),
            ("experiments_dir", "experiments"),
            ("reconstructions_root", "reconstructions"),
            ("results_root", "results"),
            ("base_reconstructions_root", "reconstructions"),
            ("initializations_root", "reconstructions/_initializations"),
        ):
            with self.subTest(name=name):
                self.assertEqual(getattr(args, name), config.ROOT / directory)
        self.assertEqual(args.fastgs_dir, config.ROOT.parent / "FastGS")
        self.assertFalse(args.skip_render)
        warm = config.parse_args(["--candidate-initialization", "base"])
        self.assertEqual(warm.candidate_iterations, 3000)
        self.assertEqual(warm.run_name, "base_init_3000")

    def test_custom_values_and_cli_overrides_including_lists_and_false(self):
        values = {
            "object_id": ["configured"], "experiment_id": ["configured-experiment"],
            "base_iterations": 900, "skip_render": True, "base_control": True,
            "extra_gaussian_seed": 8, "adaptation_lr_scale": 0.3,
        }
        configured = self.parse(values)
        self.assertEqual(configured.object_id, ["configured"])
        self.assertEqual(configured.candidate_iterations, 900)
        self.assertTrue(configured.skip_render)
        args = self.parse(
            values, "--object-ids", "first", "second", "--object-id", "third",
            "--experiment-id", "cli-experiment", "--base-iterations", "700",
            "--no-skip-render", "--no-base-control", "--extra-gaussian-seed", "0",
        )
        self.assertEqual(args.object_id, ["first", "second", "third"])
        self.assertEqual(args.experiment_id, ["cli-experiment"])
        self.assertEqual((args.base_iterations, args.candidate_iterations), (700, 700))
        self.assertFalse(args.skip_render)
        self.assertFalse(args.base_control)
        self.assertEqual(args.extra_gaussian_seed, 0)
        self.assertEqual(args.adaptation_lr_scale, 0.3)

    def test_json_paths_use_config_directory_cli_paths_use_working_directory(self):
        cwd = self.root / "elsewhere"
        cwd.mkdir()
        previous = Path.cwd()
        try:
            os.chdir(cwd)
            defaults = config.parse_args([])
            args = self.parse({"views_root": "images"}, "--results-root", "cli-results")
        finally:
            os.chdir(previous)
        self.assertEqual(args.views_root, self.root / "images")
        self.assertEqual(args.results_root, cwd / "cli-results")
        self.assertEqual(args.fastgs_dir, config.ROOT.parent / "FastGS")
        self.assertEqual(defaults.views_root, config.ROOT / "views")
        self.assertEqual(defaults.fastgs_dir, config.ROOT.parent / "FastGS")

    def test_derived_defaults_follow_merged_values(self):
        args = self.parse({
            "candidate_initialization": "base", "base_candidate_iterations": 600,
            "reconstructions_root": "runs", "results_root": "scores",
        })
        self.assertEqual(args.candidate_iterations, 600)
        self.assertEqual(args.adaptation_densify_until_iter, 300)
        self.assertEqual(args.base_reconstructions_root, self.root / "runs")
        self.assertEqual(args.initializations_root, self.root / "runs/_initializations")
        self.assertEqual(args.reconstructions_root, self.root / "runs/base_init_600")
        self.assertEqual(args.results_root, self.root / "scores/base_init_600")
        overridden = self.parse(
            {"candidate_initialization": "base", "base_candidate_iterations": 600},
            "--candidate-iterations", "800",
        )
        self.assertEqual(overridden.candidate_iterations, 800)

    def test_editing_default_file_and_resetting_nullable_values(self):
        shipped = json.loads(config.DEFAULT_CONFIG.read_text())
        shipped.update(object_id=["default-object"], python="custom-python")
        edited = self.root / "defaults.json"
        edited.write_text(json.dumps(shipped))
        with patch.object(config, "DEFAULT_CONFIG", edited):
            args = config.parse_args([])
            self.assertEqual(args.object_id, ["default-object"])
            reset = self.parse({"object_id": None, "python": None})
        self.assertIsNone(reset.object_id)
        self.assertEqual(reset.python, sys.executable)

    def test_invalid_json_types_keys_and_values_are_actionable(self):
        invalid = [
            ({"base_iteratons": 10}, "Unknown config keys"),
            ({"skip_render": "false"}, "skip_render"),
            ({"base_iterations": True}, "base_iterations"),
            ({"base_iterations": 2.5}, "base_iterations"),
            ({"base_iterations": None}, "base_iterations"),
            ({"base_iterations": -1}, "iteration budgets"),
            ({"candidate_initialization": "unknown"}, "candidate_initialization"),
            ({"object_id": "object"}, "object_id"),
            ({"experiment_id": []}, "experiment_id"),
            ({"views_root": 5}, "views_root"),
            ({"adaptation_lr_scale": float("nan")}, "adaptation_lr_scale"),
            ({"base_candidate_iterations": 0}, "base_candidate_iterations"),
            ([], "must be a JSON object"),
        ]
        for values, message in invalid:
            with self.subTest(values=values):
                stderr = io.StringIO()
                with contextlib.redirect_stderr(stderr), self.assertRaises(SystemExit) as error:
                    self.parse(values)
                self.assertEqual(error.exception.code, 2)
                self.assertIn(message, stderr.getvalue())

    def test_missing_and_malformed_config_fail_cleanly(self):
        for content in (None, "{bad json"):
            if content is not None:
                self.path.write_text(content)
            stderr = io.StringIO()
            with contextlib.redirect_stderr(stderr), self.assertRaises(SystemExit):
                config.parse_args(["--config", str(self.path)])
            self.assertIn("Cannot load config", stderr.getvalue())


if __name__ == "__main__":
    unittest.main()
