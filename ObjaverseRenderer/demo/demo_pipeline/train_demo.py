"""Launch the standard FastGS trainer with this demo run's training settings."""

import json
import os
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT))
import training_fastgs


def main():
    spec_path = Path(os.environ["PIPELINE_DEMO_RUN_SPEC"])
    training = json.loads(spec_path.read_text(encoding="utf-8"))["training"]
    training_fastgs.CONFIG.update({key: value for key, value in training.items()
                                   if key not in ("fastgs_dir", "python")})
    training_fastgs.main()


if __name__ == "__main__":
    main()
