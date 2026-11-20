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
  relocate_from="${2:?Usage: prepare_reconstruction.sh --relocate-from OLD_RUN_ROOT}"
  shift 2
fi
if (( $# )); then
  echo "Usage: prepare_reconstruction.sh [--check-scratch | --relocate-from OLD_RUN_ROOT]" >&2
  echo "Set the destination run paths in ObjaverseRenderer/orchestrator_config.json." >&2
  exit 2
fi
scratch_root="${FASTGS_SCRATCH_ROOT:-/scratch/vc15/${USER:?}}"
env_python="$HOME/miniforge3/envs/adl4cv-learning/bin/python"

if (( check_scratch )); then
  torch_cache="${TORCH_HOME:-$scratch_root/.cache/torch}"
  "$env_python" - "$torch_cache" <<'PY'
from pathlib import Path
import subprocess
import sys
from tempfile import TemporaryDirectory

sys.path.insert(0, str(Path('ObjaverseRenderer').resolve()))
import orchestrator

orchestrator.build_parser().parse_args(['--use-candidate-base', '--skip-failed-object'])
config = orchestrator.CONFIG
manifest = orchestrator.check_prepared_run()

print('Checking prepared run:', config['run_root'])
for object_id in manifest['objects']:
    glb = config['objects_root'] / f'{object_id}.glb'
    with glb.open('rb') as asset:
        if asset.read(4) != b'glTF':
            raise SystemExit(f'Invalid GLB header: {glb}')

torch_cache = Path(sys.argv[1])
checkpoints = torch_cache / 'hub' / 'checkpoints'
for filename in ('vgg16-397923af.pth', 'vgg.pth'):
    weight = checkpoints / filename
    with weight.open('rb') as cached:
        if not cached.read(1):
            raise SystemExit(f'Empty LPIPS weight file: {weight}')
print('GLB headers and LPIPS cache are readable:', checkpoints)

directories = [config[f'{name}_root'] for name in ('views', 'reconstructions', 'results')]
for directory in directories:
    directory.mkdir(parents=True, exist_ok=True)
    with TemporaryDirectory(prefix='.fastgs-preflight-', dir=directory) as probe:
        pending = Path(probe) / 'pending'
        data = b'scratch check\n'
        pending.write_bytes(data)
        saved = pending.replace(Path(probe) / 'saved')
        if saved.read_bytes() != data:
            raise SystemExit(f'Scratch read-back failed: {directory}')
    print('Write, rename, read, cleanup OK:', directory)
sys.stdout.flush()
for flags in ('-hP', '-iP'):
    subprocess.run(['df', flags, '--', *directories, torch_cache], check=True)
PY
  echo "Login-node scratch check passed."
  exit 0
fi

mkdir -p logs
"$env_python" - "$relocate_from" <<'PY'
from pathlib import Path
import subprocess
import sys

sys.path.insert(0, str(Path('ObjaverseRenderer').resolve()))
import orchestrator

root = orchestrator.CONFIG['run_root']
objects_root = orchestrator.CONFIG['objects_root']
relocate_from = sys.argv[1]
if not relocate_from:
    objects_root.mkdir(parents=True, exist_ok=True)
    subprocess.run([
        sys.executable, 'ObjaverseRenderer/download_gbuffer_objaverse.py',
        '--index', 'ObjaverseRenderer/objects/unflagged20_20261113.json',
        '--count', '20', '--seed', '77', '--output-dir', objects_root,
    ], check=True)
objects = orchestrator.discover_objects()
if len(objects) != 20:
    raise SystemExit(f'Expected exactly 20 GLBs in {objects_root}, found {len(objects)}')
experiments = orchestrator.load_experiments()

config_path = Path('ObjaverseRenderer/orchestrator_config.json')
manifest_path = root / 'run_manifest.json'
experiments_ids = [experiment['id'] for experiment in experiments]

if relocate_from:
    old_root = Path(relocate_from).expanduser().resolve()
    if old_root == root:
        raise SystemExit('Old and new run roots must differ.')
    manifest = orchestrator.load_json(manifest_path)
    if objects != manifest['objects'] or experiments_ids != manifest['experiments']:
        raise SystemExit('The copied objects or current experiments differ from the prepared run.')
    config = manifest['configs'][str(config_path)]
    for filename, expected in manifest['configs'].items():
        if filename != str(config_path) and orchestrator.load_json(Path(filename)) != expected:
            raise SystemExit(f'Configuration changed since preparation: {filename}')
    for folder in orchestrator.RUN_FOLDERS:
        key = f'{folder}_root'
        old_path = Path(config[key])
        if old_path.is_relative_to(old_root):
            config[key] = str(root / old_path.relative_to(old_root))
    if config != orchestrator.config_snapshot():
        raise SystemExit(f'Configuration changed since preparation: {config_path}')

    def relocate_paths(record, fields):
        changed = False
        for field in fields:
            if field in record:
                path = Path(record[field])
                if path.is_relative_to(old_root):
                    record[field] = str(root / path.relative_to(old_root))
                    changed = True
        return changed

    for path in sorted(orchestrator.CONFIG['results_root'].glob('*/*/results.json')):
        result = orchestrator.load_json(path)
        changed = relocate_paths(result, ('dataset_path', 'reconstruction_path', 'checkpoint_path'))
        for record in result.get('reconstructions', {}).values():
            if relocate_paths(record, ('reconstruction_path', 'checkpoint_path')):
                changed = True
        if changed:
            orchestrator.save_json(path, result)
    orchestrator.save_json(manifest_path, manifest)
    print(f'Relocated run to {root}.')
    raise SystemExit(0)

from download_gbuffer_objaverse import load_index, select_objects

# Freeze the selection and settings so submission needs no shell variables.
config_files = [Path('ObjaverseRenderer/training_config.json'), Path('ObjaverseRenderer/config.json')]
config_files.extend(sorted(orchestrator.CONFIG['experiments_dir'].glob('*.json')))
configs = {str(path): orchestrator.load_json(path) for path in config_files}
configs[str(config_path)] = orchestrator.config_snapshot()
manifest = {
    'source_ids': select_objects(load_index(
        Path('ObjaverseRenderer/objects/unflagged20_20261113.json')), count=20, seed=77),
    'selection_seed': 77,
    'objects': objects,
    'experiments': experiments_ids,
    'configs': configs,
}
orchestrator.save_json(manifest_path, manifest)
print(f'Prepared {len(objects)} objects x {len(experiments)} experiments.')
print(f'Run directory: {root}')
print('Objects:', ', '.join(objects))
print('Submit from the repository root: sbatch reconstruct.sbatch')
PY
