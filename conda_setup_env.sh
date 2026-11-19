#!/usr/bin/env bash
# Run on the login node. Install dependencies here; compile extensions via Slurm.
# Do not use set -u: CUDA's conda hooks read optional, unset variables.
set -eo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
cd "$SCRIPT_DIR"

# /tmp is a RAM disk on this cluster. Use scratch for installation temporaries.
export TMPDIR="${SCRATCH:-$HOME/tmp}/fastgs-setup-tmp"
mkdir -p "$TMPDIR"

source "$HOME/miniforge3/etc/profile.d/conda.sh"
if [[ ! -d "$HOME/miniforge3/envs/adl4cv-learning/conda-meta" ]]; then
  conda create -n adl4cv-learning -c conda-forge python=3.11 -y
fi
conda activate adl4cv-learning

# CUDA 12.8 supports GCC 6-14. Pin GCC 13 rather than using the newest compiler.
conda install -c nvidia -c conda-forge -y \
  python=3.11 pip setuptools wheel packaging ninja cmake git \
  gcc_linux-64=13 gxx_linux-64=13 \
  nvidia::cuda-toolkit=12.8 nvidia::cuda-nvcc=12.8 \
  xorg-libxrender xorg-libxfixes xorg-libxi \
  xorg-libsm xorg-libice libgl

# Refresh compiler activation hooks after installing the toolchain.
conda activate adl4cv-learning
export CUDA_HOME="$CONDA_PREFIX"
export CC="$CONDA_PREFIX/bin/x86_64-conda-linux-gnu-cc"
export CXX="$CONDA_PREFIX/bin/x86_64-conda-linux-gnu-c++"
export LD_LIBRARY_PATH="$CONDA_PREFIX/lib${LD_LIBRARY_PATH:+:$LD_LIBRARY_PATH}"

git submodule update --init --recursive
test -f FastGS/submodules/diff-gaussian-rasterization_fastgs/third_party/glm/glm/glm.hpp
for extension in FisherRF/submodules/diff-gaussian-rasterization FisherRF/diff; do
  test -f "$extension/setup.py"
  test -f "$extension/third_party/glm/glm/glm.hpp"
done
for extension in diff-gaussian-rasterization_fastgs simple-knn fused-ssim; do
  test -f "FastGS/submodules/$extension/setup.py"
done

python -m pip install \
  torch==2.7.1 torchvision==0.22.1 torchaudio==2.7.1 \
  --index-url https://download.pytorch.org/whl/cu128

python -m pip install -r requirements.txt
python -m pip check

echo "Checking Python packages and CUDA compiler (no GPU required)..."
# Conda supplies Ninja as an executable, not necessarily a Python module.
ninja --version
python - <<'PY'
import sys
import os
import subprocess
import setuptools, wheel, packaging, numpy, plyfile, tqdm, websockets
import torch
import torchvision
from torch.utils.tensorboard import SummaryWriter
import matplotlib, tensorboard
from PIL import Image

assert sys.version_info[:2] == (3, 11), sys.version
assert torch.__version__.split('+')[0] == '2.7.1', torch.__version__
assert torchvision.__version__.split('+')[0] == '0.22.1', torchvision.__version__
assert torch.version.cuda == '12.8', torch.version.cuda
nvcc_version = subprocess.check_output(
    [os.path.join(os.environ['CUDA_HOME'], 'bin', 'nvcc'), '--version'], text=True)
assert 'release 12.8,' in nvcc_version, nvcc_version
print(f"Python: {sys.executable}")
print(f"PyTorch: {torch.__version__}; torchvision: {torchvision.__version__}")
PY
nvcc --version
"$CXX" --version

# Compile and link one tiny CUDA program without executing it. This checks the
# CUDA headers/libraries, host compiler, and sm_120 support before a GPU job.
build_check_dir="$(mktemp -d)"
trap 'rm -rf "$build_check_dir"' EXIT
cat > "$build_check_dir/check.cu" <<'CU'
#include <cuda_runtime.h>
__global__ void check(float* x) { x[threadIdx.x] = 1.0f; }
int main() { return 0; }
CU
nvcc -std=c++17 -arch=sm_120 -ccbin "$CC" \
  "$build_check_dir/check.cu" -o "$build_check_dir/check"

echo "Checking Blender startup..."
BLENDER_EXE="${BLENDER_EXE:-$HOME/software/blender-4.5.16-linux-x64/blender}"
"$BLENDER_EXE" --background --factory-startup --python-exit-code 1 \
  --python-expr "import bpy; print('Blender startup OK:', bpy.app.version_string)"

# FastGS evaluation needs VGG LPIPS weights. Download them on the login node
# so a later training/evaluation job can use the cache without a download.
echo "Preparing VGG LPIPS weights..."
cd FastGS
python - <<'PY'
import torch
from lpipsPyTorch import LPIPS
LPIPS(net_type='vgg').eval()
print('LPIPS weights ready in:', torch.hub.get_dir())
PY

echo "Login-node preparation passed. Submit setup.sbatch from the repository root."
