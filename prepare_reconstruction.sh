#!/usr/bin/env bash
# Run on the login node. Use the environment's Python directly; no activation.
set -eo pipefail
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
cd "$SCRIPT_DIR"
RUN_ROOT="${1:-${SCRATCH:?}/fastgs-20-objects}"
mkdir -p "$RUN_ROOT/objects" logs
RUN_ROOT="$(cd "$RUN_ROOT" && pwd)"

env_python="$HOME/miniforge3/envs/adl4cv-learning/bin/python"

"$env_python" ObjaverseRenderer/download_gbuffer_objaverse.py \
  --index ObjaverseRenderer/objects/unflagged20_20261113.json \
  --count 20 --seed 77 \
  --output-dir "$RUN_ROOT/objects"

"$env_python" - "$RUN_ROOT" <<'PY'
import json
from pathlib import Path
import sys

sys.path.insert(0, str(Path('ObjaverseRenderer').resolve()))
import orchestrator
from download_gbuffer_objaverse import load_index, select_objects

root = Path(sys.argv[1])
objects = sorted(path.stem for path in (root / 'objects').glob('*.glb'))
if len(objects) != 20:
    raise SystemExit(f'Expected exactly 20 GLBs in {root / "objects"}, found {len(objects)}')
experiments = orchestrator.load_experiments()

config_path = Path('ObjaverseRenderer/orchestrator_config.json')
config = json.loads(config_path.read_text())
for key, folder in (
    ('objects_root', 'objects'), ('views_root', 'views'),
    ('reconstructions_root', 'reconstructions'), ('results_root', 'results'),
):
    config[key] = str(root / folder)
config['base_cloud_ply'] = None
config_path.write_text(json.dumps(config, indent=2) + '\n')

# Freeze the selection and settings so submission needs no shell variables.
config_files = [config_path, Path('ObjaverseRenderer/training_config.json'),
                Path('ObjaverseRenderer/config.json')]
config_files.extend(sorted(orchestrator.CONFIG['experiments_dir'].glob('*.json')))
manifest = {
    'source_ids': select_objects(load_index(
        Path('ObjaverseRenderer/objects/unflagged20_20261113.json')), count=20, seed=77),
    'selection_seed': 77,
    'objects': objects,
    'experiments': [experiment['id'] for experiment in experiments],
    'configs': {str(path): json.loads(path.read_text()) for path in config_files},
}
(root / 'run_manifest.json').write_text(json.dumps(manifest, indent=2) + '\n')
# Visit all objects for an experiment before proceeding to the next experiment.
(root / 'jobs.tsv').write_text(''.join(
    f"{object_id}\t{experiment['id']}\n"
    for experiment in experiments for object_id in objects
))
print(f'Prepared {len(objects)} objects x {len(experiments)} experiments.')
print(f'Run directory: {root}')
print('Objects:', ', '.join(objects))
print('Submit from the repository root: sbatch reconstruct.sbatch')
PY
