#!/usr/bin/env bash
set -euo pipefail

eval "$(conda shell.bash hook)"

git submodule update --init --recursive

conda create -n adl4cv-learning python=3.11 pip ninja cmake -c conda-forge -y

conda activate adl4cv-learning

conda install -c nvidia cuda-toolkit=12.8 -y

python -c "import sys; print(sys.executable)"

python -m pip install \
  torch==2.7.1 torchvision==0.22.1 torchaudio==2.7.1 \
  --index-url https://download.pytorch.org/whl/cu128

python -m pip install plyfile tqdm websockets matplotlib pyqt5 tensorboard pillow

cd FastGS

export CUDA_HOME="$CONDA_PREFIX"

export PATH="$CUDA_HOME/bin:$PATH"

export TORCH_CUDA_ARCH_LIST="12.0"

export MAX_JOBS="$(nproc)"

which nvcc

nvcc --version

python -m pip install --no-build-isolation --no-cache-dir --force-reinstall \
  ./submodules/diff-gaussian-rasterization_fastgs

python -m pip install --no-build-isolation --no-cache-dir \
  ./submodules/simple-knn

python -m pip install --no-build-isolation --no-cache-dir \
  ./submodules/fused-ssim