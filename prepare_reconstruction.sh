#!/usr/bin/env bash
# Run on the login node. Use the environment's Python directly; no activation.
set -eo pipefail
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
cd "$SCRIPT_DIR"
check_scratch=0
relocate_from=
if [[ "${1:-}" == --check-scratch ]]; then
  check_scratch=1
  shift
elif [[ "${1:-}" == --relocate-from ]]; then
  relocate_from="${2:?Usage: prepare_reconstruction.sh --relocate-from OLD_RUN_ROOT [NEW_RUN_ROOT]}"
  shift 2
fi
scratch_root="${FASTGS_SCRATCH_ROOT:-/scratch/vc15/${USER:?}}"
RUN_ROOT="${1:-$scratch_root/fastgs-20-objects}"
env_python="$HOME/miniforge3/envs/adl4cv-learning/bin/python"

if (( check_scratch )); then
  torch_cache="${TORCH_HOME:-$scratch_root/.cache/torch}"
  "$env_python" - "$RUN_ROOT" "$torch_cache" <<'PY'
import json
from pathlib import Path
import sys
from tempfile import TemporaryDirectory

sys.path.insert(0, str(Path('ObjaverseRenderer').resolve()))
import orchestrator

orchestrator.build_parser().parse_args(['--use-candidate-base', '--skip-failed-object'])
root = Path(sys.argv[1]).resolve()
manifest = json.loads((root / 'run_manifest.json').read_text())
for filename, expected in manifest['configs'].items():
    if json.loads(Path(filename).read_text()) != expected:
        raise SystemExit(f'Configuration changed since preparation: {filename}')

print('Checking prepared run:', root)
for name in ('objects', 'views', 'reconstructions', 'results'):
    configured = orchestrator.CONFIG[f'{name}_root']
    expected = (root / name).resolve()
    if configured != expected:
        raise SystemExit(f'{name}_root points to {configured}, expected {expected}')

if orchestrator.discover_objects() != manifest['objects']:
    raise SystemExit('Object selection changed since preparation.')
if [item['id'] for item in orchestrator.load_experiments()] != manifest['experiments']:
    raise SystemExit('Experiment selection changed since preparation.')
for object_id in manifest['objects']:
    glb = root / 'objects' / f'{object_id}.glb'
    with glb.open('rb') as asset:
        if asset.read(4) != b'glTF':
            raise SystemExit(f'Invalid GLB header: {glb}')

checkpoints = Path(sys.argv[2]) / 'hub' / 'checkpoints'
for filename in ('vgg16-397923af.pth', 'vgg.pth'):
    weight = checkpoints / filename
    with weight.open('rb') as cached:
        if not cached.read(1):
            raise SystemExit(f'Empty LPIPS weight file: {weight}')
print('GLB headers and LPIPS cache are readable:', checkpoints)

for name in ('views', 'reconstructions', 'results'):
    directory = root / name
    directory.mkdir(exist_ok=True)
    with TemporaryDirectory(prefix='.fastgs-preflight-', dir=directory) as probe:
        pending = Path(probe) / 'pending'
        data = b'scratch check\n'
        pending.write_bytes(data)
        saved = pending.replace(Path(probe) / 'saved')
        if saved.read_bytes() != data:
            raise SystemExit(f'Scratch read-back failed: {directory}')
    print('Write, rename, read, cleanup OK:', directory)
PY
  scratch_paths=("$RUN_ROOT"/{views,reconstructions,results} "$torch_cache")
  df -hP -- "${scratch_paths[@]}"
  df -iP -- "${scratch_paths[@]}"
  echo "Login-node scratch check passed."
  exit 0
fi

mkdir -p logs
if [[ -z "$relocate_from" ]]; then
  mkdir -p "$RUN_ROOT/objects"
  "$env_python" ObjaverseRenderer/download_gbuffer_objaverse.py \
    --index ObjaverseRenderer/objects/unflagged20_20261113.json \
    --count 20 --seed 77 \
    --output-dir "$RUN_ROOT/objects"
fi
RUN_ROOT="$(cd "$RUN_ROOT" && pwd)"

"$env_python" - "$RUN_ROOT" "$relocate_from" <<'PY'
import json
from pathlib import Path
import sys

sys.path.insert(0, str(Path('ObjaverseRenderer').resolve()))
import orchestrator

root = Path(sys.argv[1])
objects = sorted(path.stem for path in (root / 'objects').glob('*.glb'))
if len(objects) != 20:
    raise SystemExit(f'Expected exactly 20 GLBs in {root / "objects"}, found {len(objects)}')
experiments = orchestrator.load_experiments()

config_path = Path('ObjaverseRenderer/orchestrator_config.json')
manifest_path = root / 'run_manifest.json'
experiments_ids = [experiment['id'] for experiment in experiments]
data_folders = ('objects', 'views', 'reconstructions', 'results')

if sys.argv[2]:
    old_root = Path(sys.argv[2]).expanduser().resolve()
    if old_root == root:
        raise SystemExit('Old and new run roots must differ.')
    manifest = orchestrator.load_json(manifest_path)
    if objects != manifest['objects'] or experiments_ids != manifest['experiments']:
        raise SystemExit('The copied objects or current experiments differ from the prepared run.')
    config = manifest['configs'][str(config_path)]
    for filename, expected in manifest['configs'].items():
        if filename != str(config_path) and orchestrator.load_json(Path(filename)) != expected:
            raise SystemExit(f'Configuration changed since preparation: {filename}')
    for folder in data_folders:
        key = f'{folder}_root'
        if Path(config[key]) not in (old_root / folder, root / folder):
            raise SystemExit(f'{key} in the copied run manifest does not match the old or new root.')
        config[key] = str(root / folder)

    def relocate_paths(record, fields):
        changed = False
        for field in fields:
            if field in record:
                path = Path(record[field])
                if path.is_relative_to(old_root):
                    record[field] = str(root / path.relative_to(old_root))
                    changed = True
        return changed

    for path in sorted((root / 'results').glob('*/*/results.json')):
        result = orchestrator.load_json(path)
        changed = relocate_paths(result, ('dataset_path', 'reconstruction_path', 'checkpoint_path'))
        for record in result.get('reconstructions', {}).values():
            if relocate_paths(record, ('reconstruction_path', 'checkpoint_path')):
                changed = True
        if changed:
            orchestrator.save_json(path, result)
    orchestrator.save_json(config_path, config)
    orchestrator.save_json(manifest_path, manifest)
    print(f'Relocated run to {root}.')
    raise SystemExit(0)

from download_gbuffer_objaverse import load_index, select_objects

config = json.loads(config_path.read_text())
for folder in data_folders:
    config[f'{folder}_root'] = str(root / folder)
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
    'experiments': experiments_ids,
    'configs': {str(path): json.loads(path.read_text()) for path in config_files},
}
orchestrator.save_json(manifest_path, manifest)
print(f'Prepared {len(objects)} objects x {len(experiments)} experiments.')
print(f'Run directory: {root}')
print('Objects:', ', '.join(objects))
print('Submit from the repository root: sbatch reconstruct.sbatch')
PY
