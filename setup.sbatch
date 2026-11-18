#!/usr/bin/env bash
#SBATCH --job-name=fastgs-setup
#SBATCH --partition=rtx_5090
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=2
#SBATCH --mem=32G
#SBATCH --time=02:00:00
#SBATCH --output=logs/%x_%j.out
#SBATCH --error=logs/%x_%j.err

# Conda's CUDA activation hooks are not compatible with set -u.
set -eo pipefail

# Submit from the repository root; Slurm copies this file to its spool directory.
cd "${SLURM_SUBMIT_DIR:?Submit this script with sbatch from the repository root}"
if [[ ! -d FastGS/submodules ]]; then
  echo "Submit from the repository root after running conda_setup_env.sh." >&2
  exit 1
fi

source "$HOME/miniforge3/etc/profile.d/conda.sh"
conda activate adl4cv-learning
export CUDA_HOME="$CONDA_PREFIX"
export CC="$CONDA_PREFIX/bin/x86_64-conda-linux-gnu-cc"
export CXX="$CONDA_PREFIX/bin/x86_64-conda-linux-gnu-c++"
export LD_LIBRARY_PATH="$CONDA_PREFIX/lib${LD_LIBRARY_PATH:+:$LD_LIBRARY_PATH}"

# On this cluster /tmp is a RAM disk, so use the allocated node's SSD instead.
node_name="${SLURMD_NODENAME:-$(hostname -s)}"
scratch_root="/scratch/${node_name,,}/${USER:?}"
export TMPDIR="$scratch_root/fastgs-setup/${SLURM_JOB_ID:?}"
export CUDA_CACHE_PATH="$scratch_root/.cache/nv/ComputeCache"
mkdir -p "$TMPDIR" "$CUDA_CACHE_PATH"

export MAX_JOBS=1
export TORCH_CUDA_ARCH_LIST="12.0"

echo "Starting setup: job $SLURM_JOB_ID on $node_name"
nvidia-smi
ninja --version
nvcc --version
"$CXX" --version
cd FastGS

python -u - <<'PY'
import torch
assert torch.version.cuda == '12.8', torch.version.cuda
assert torch.cuda.is_available(), 'PyTorch cannot access the allocated GPU.'
assert torch.cuda.get_device_capability(0) == (12, 0), 'Expected an RTX 5090 GPU.'
assert torch.ones(1, device='cuda').item() == 1
print('GPU preflight passed:', torch.cuda.get_device_name(0))
PY

# Dependencies are already installed on the login node. Do not download or
# resolve packages in this job, or let pip replace the pinned PyTorch build.
for extension in diff-gaussian-rasterization_fastgs simple-knn fused-ssim; do
  echo "Building $extension..."
  python -m pip install --no-build-isolation --no-deps --no-index --no-cache-dir \
    "./submodules/$extension"
done

python -u - <<'PY'
import torch
from diff_gaussian_rasterization_fastgs import _C as rasterizer_cuda
from simple_knn._C import distCUDA2
from fused_ssim import fused_ssim

# Exercise a CUDA kernel in each extension, not just its Python import.
points = torch.rand(1024, 3, device='cuda')
matrix = torch.eye(4, device='cuda')
visible = rasterizer_cuda.mark_visible(points, matrix, matrix)
assert visible.shape == (1024,)
assert torch.isfinite(distCUDA2(points)).all().item()
image = torch.rand(1, 3, 32, 32, device='cuda', requires_grad=True)
score = fused_ssim(image, image.detach())
score.backward()
assert abs(score.item() - 1.0) < 1e-4, score.item()
assert torch.isfinite(image.grad).all().item()
torch.cuda.synchronize()
print('All three FastGS extensions passed CUDA smoke checks.')
PY

python train.py --help
python render.py --help
python metrics.py --help

BLENDER_EXE="${BLENDER_EXE:-$HOME/software/blender-4.5.16-linux-x64/blender}"
"$BLENDER_EXE" --background --factory-startup --python-exit-code 1 \
  --python-expr "import bpy; p = bpy.context.preferences.addons['cycles'].preferences; p.compute_device_type = 'CUDA'; p.refresh_devices(); gpus = [d.name for d in p.devices if d.type == 'CUDA']; print('Blender CUDA devices:', gpus); assert gpus, 'Blender cannot access a CUDA GPU'"

echo "Setup completed successfully."
