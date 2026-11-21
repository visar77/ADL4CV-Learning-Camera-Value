#!/usr/bin/env bash
#SBATCH --job-name=fastgs-batch
#SBATCH --partition=rtx_5090
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=16
#SBATCH --mem=48G
#SBATCH --time=06:00:00
#SBATCH --output=logs/%x_%j.out
#SBATCH --error=logs/%x_%j.err

# CUDA conda hooks read optional variables, so do not use set -u.
set -eo pipefail
if [[ $# == 1 && "$1" == --check-order ]]; then
  cd "$(dirname "${BASH_SOURCE[0]}")"
  batch_python="$HOME/miniforge3/envs/adl4cv-learning/bin/python"
elif (( $# == 0 )); then
  cd "${SLURM_SUBMIT_DIR:?Submit with sbatch from the repository root}"
  scratch_root="${FASTGS_SCRATCH_ROOT:-/scratch/vc15/${USER:?}}"

  source "$HOME/miniforge3/etc/profile.d/conda.sh"
  conda activate adl4cv-learning
  export LD_LIBRARY_PATH="$CONDA_PREFIX/lib${LD_LIBRARY_PATH:+:$LD_LIBRARY_PATH}"
  export PATH="$HOME/software/blender-4.5.16-linux-x64:$PATH"
  export TORCH_HOME="${TORCH_HOME:-$scratch_root/.cache/torch}"
  export OMP_NUM_THREADS="$SLURM_CPUS_PER_TASK"

  node_name="${SLURMD_NODENAME,,}"
  export TMPDIR="$scratch_root/fastgs-batch-tmp/${SLURM_JOB_ID:?}"
  export CUDA_CACHE_PATH="$scratch_root/.cache/nv/ComputeCache"
  mkdir -p "$TMPDIR" "$CUDA_CACHE_PATH"

  echo "Starting reconstruction batch: job $SLURM_JOB_ID on $node_name"
  batch_python=python
else
  echo "Submit with sbatch reconstruct.sbatch or run bash reconstruct.sbatch --check-order." >&2
  exit 2
fi

"$batch_python" -B -u - "$@" <<'PY'
from pathlib import Path
import sys

sys.path.insert(0, str(Path('ObjaverseRenderer').resolve()))
import orchestrator

manifest = orchestrator.check_prepared_run()
print('Run:', orchestrator.CONFIG['run_root'])
print('Results:', orchestrator.CONFIG['results_root'])
print('Selected objects:', len(manifest['objects']))
print('Selected experiments:', len(manifest['experiments']))

ordered_objects = manifest['objects']
if len(ordered_objects) == 50:
    extension = orchestrator.load_json(Path('ObjaverseRenderer/objects/extension30_20261121.json'))
    additions = {asset['object_id'] for asset in extension['objects']}
    if len(additions) != 30 or not additions.issubset(manifest['objects']):
        raise SystemExit('The prepared run does not match the 30-object extension.')
    ordered_objects = sorted(manifest['objects'], key=lambda name: name in additions)
    discover_objects = orchestrator.discover_objects

    # Override order for this batch process after validating the full manifest.
    def discover_in_batch_order(requested_id=None):
        return ordered_objects if requested_id is None else discover_objects(requested_id)

    orchestrator.discover_objects = discover_in_batch_order
    print('Object order: original 20 first, then the 30 additions.')

if sys.argv[1:] == ['--check-order']:
    skip_path = orchestrator.CONFIG['run_root'] / 'skipped_objects.json'
    skips = orchestrator.load_json(skip_path) if skip_path.is_file() else {}
    for position, name in enumerate(ordered_objects, 1):
        status = ' (saved skip)' if name in skips else ''
        print(f'{position:2}. {name}{status}')
    print('Batch order check passed.')
    raise SystemExit(0)

orchestrator.main(['--use-candidate-base', '--skip-failed-object'])
print('Batch sweep finished.')
PY
