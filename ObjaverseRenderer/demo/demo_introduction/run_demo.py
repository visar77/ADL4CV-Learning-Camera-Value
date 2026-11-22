"""Generate the Vespa street camera demo."""

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent / "complex_scene"))
from run_street_demo import main

if __name__ == "__main__":
    main()
