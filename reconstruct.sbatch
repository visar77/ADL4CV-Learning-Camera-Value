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
RUN_ROOT="${1:-${SCRATCH:?}/fastgs-20-objects}"
RUN_ROOT="$(cd "$RUN_ROOT" && pwd)"

source "$HOME/miniforge3/etc/profile.d/conda.sh"
conda activate adl4cv-learning
export LD_LIBRARY_PATH="$CONDA_PREFIX/lib${LD_LIBRARY_PATH:+:$LD_LIBRARY_PATH}"
export PATH="$HOME/software/blender-4.5.16-linux-x64:$PATH"
export TORCH_HOME="${TORCH_HOME:-${SCRATCH:?}/.cache/torch}"
export OMP_NUM_THREADS="$SLURM_CPUS_PER_TASK"

node_name="${SLURMD_NODENAME,,}"
scratch_root="/scratch/$node_name/${USER:?}"
export TMPDIR="$scratch_root/fastgs-batch-tmp/${SLURM_JOB_ID:?}"
export CUDA_CACHE_PATH="$scratch_root/.cache/nv/ComputeCache"
mkdir -p "$TMPDIR" "$CUDA_CACHE_PATH"

# Catch accidental configuration changes between preparation and execution.
python - "$RUN_ROOT" <<'PY'
import json
from pathlib import Path
import sys

root = Path(sys.argv[1])
manifest = json.loads((root / 'run_manifest.json').read_text())
for filename, expected in manifest['configs'].items():
    if json.loads(Path(filename).read_text()) != expected:
        raise SystemExit(f'Configuration changed since preparation: {filename}')
print('Run:', root)
print('Selected objects:', len(manifest['objects']))
print('Selected experiments:', len(manifest['experiments']))
PY

echo "Starting reconstruction batch: job $SLURM_JOB_ID on $node_name"
while IFS=$'\t' read -r object_id experiment_id; do
  echo "Running $object_id / $experiment_id"
  if python -u ObjaverseRenderer/orchestrator.py \
    --object-id "$object_id" \
    --experiment-id "$experiment_id" \
    --use-candidate-base \
    --continue-on-error \
    --retries 1; then
    echo "Finished invocation: $object_id / $experiment_id (see results for individual statuses)"
  else
    exit_code=$?
    echo "Skipping failed invocation: $object_id / $experiment_id (exit $exit_code)" >&2
    printf '%s\t%s\t%s\n' "$object_id" "$experiment_id" "$exit_code" \
      >> "$RUN_ROOT/job_${SLURM_JOB_ID}_errors.tsv"
  fi
done < "$RUN_ROOT/jobs.tsv"

echo "Batch sweep finished. Individual outcomes are in $RUN_ROOT/results."
