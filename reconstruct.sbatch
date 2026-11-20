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
if (( $# )); then
  echo "Set the run paths in ObjaverseRenderer/orchestrator_config.json." >&2
  exit 2
fi
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

# Catch accidental configuration changes between preparation and execution.
python - <<'PY'
from pathlib import Path
import sys

sys.path.insert(0, str(Path('ObjaverseRenderer').resolve()))
import orchestrator

manifest = orchestrator.check_prepared_run()
print('Run:', orchestrator.CONFIG['run_root'])
print('Results:', orchestrator.CONFIG['results_root'])
print('Selected objects:', len(manifest['objects']))
print('Selected experiments:', len(manifest['experiments']))
PY

echo "Starting reconstruction batch: job $SLURM_JOB_ID on $node_name"
python -u ObjaverseRenderer/orchestrator.py \
  --use-candidate-base \
  --skip-failed-object

echo "Batch sweep finished."
