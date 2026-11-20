#!/usr/bin/env bash
#SBATCH --job-name=fastgs-batch
#SBATCH --partition=rtx_5090
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=4
#SBATCH --mem=32G
#SBATCH --time=06:00:00
#SBATCH --output=logs/%x_%j.out
#SBATCH --error=logs/%x_%j.err

# CUDA conda hooks read optional variables, so do not use set -u.
set -eo pipefail
cd "${SLURM_SUBMIT_DIR:?Submit with sbatch from the repository root}"
scratch_root="${FASTGS_SCRATCH_ROOT:-/scratch/vc15/${USER:?}}"
RUN_ROOT="${1:-$scratch_root/fastgs-20-objects}"
RUN_ROOT="$(cd "$RUN_ROOT" && pwd)"

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

# Catch accidental configuration changes between preparation and execution.
python - "$RUN_ROOT" <<'PY'
import json
from pathlib import Path
import sys

sys.path.insert(0, str(Path('ObjaverseRenderer').resolve()))
import orchestrator

root = Path(sys.argv[1])
manifest = json.loads((root / 'run_manifest.json').read_text())
for filename, expected in manifest['configs'].items():
    if json.loads(Path(filename).read_text()) != expected:
        raise SystemExit(f'Configuration changed since preparation: {filename}')
for name in ('objects', 'views', 'reconstructions', 'results'):
    if orchestrator.CONFIG[f'{name}_root'] != root / name:
        raise SystemExit(f'Orchestrator {name}_root does not point into {root}')
if orchestrator.discover_objects() != manifest['objects']:
    raise SystemExit('Object selection changed since preparation.')
if [item['id'] for item in orchestrator.load_experiments()] != manifest['experiments']:
    raise SystemExit('Experiment selection changed since preparation.')
print('Run:', root)
print('Selected objects:', len(manifest['objects']))
print('Selected experiments:', len(manifest['experiments']))
PY

echo "Starting reconstruction batch: job $SLURM_JOB_ID on $node_name"
python -u ObjaverseRenderer/orchestrator.py \
  --use-candidate-base \
  --skip-failed-object

echo "Batch sweep finished. Individual outcomes are in $RUN_ROOT/results."
