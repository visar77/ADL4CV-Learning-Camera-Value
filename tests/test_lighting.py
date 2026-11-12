"""Check the normal launcher after removing the lighting preview system."""

import json
from pathlib import Path
import sys
import tempfile
import unittest
from unittest.mock import patch

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
import renderer


class LightingTests(unittest.TestCase):
    def test_launcher_writes_dataset_config_and_propagates_failure(self):
        with tempfile.TemporaryDirectory() as temp:
            root = Path(temp)
            model = root / "asset.glb"
            model.touch()
            with patch.object(
                renderer, "find_blender", return_value="blender"
            ), patch.object(
                renderer.subprocess, "run", side_effect=RuntimeError("worker failed")
            ) as run:
                with self.assertRaisesRegex(RuntimeError, "worker failed"):
                    renderer.main(
                        ["--glb-path", str(model), "--output-root", str(root / "out")]
                    )
            directory = root / "out/asset"
            config = json.loads((directory / "_worker_config.json").read_text())
            self.assertEqual(config["object_dir"], ".")
            self.assertEqual(config["image_width"], 512)
            self.assertEqual(config["lattice_size"], 100)
            self.assertIn("--factory-startup", run.call_args.args[0])
            self.assertIn("--python-exit-code", run.call_args.args[0])
            self.assertNotIn("preview", config)
            self.assertNotIn("lighting_mode", config)


if __name__ == "__main__":
    unittest.main()
