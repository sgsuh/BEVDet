# BEVDet dev container — adapted from docker/Dockerfile for RTX 4070 (Ada Lovelace, sm_89).
# Original used CUDA 11.3 + PyTorch 1.10 which don't support sm_89.
# First attempt at CUDA 11.7 + PyTorch 1.13.1 failed because:
#   (a) openmmlab has no prebuilt mmcv-full 1.6.x wheel for torch1.13+cu117 (so pip
#       fell back to source build), and
#   (b) PyTorch 1.13's _get_cuda_arch_flags rejects sm_89 ("Unknown CUDA arch (8.9)").
# Bumped to CUDA 11.8 + PyTorch 2.0.1 + mmcv-full 1.7.2 — first PyTorch with native
# sm_89 support, and the last mmcv 1.x that has a torch2.0+cu118 prebuilt wheel.
# (BEVDet uses mmcv 1.x APIs, so mmcv 2.x is not an option.)
FROM nvidia/cuda:11.8.0-cudnn8-devel-ubuntu20.04

ARG PYTHON_VERSION=3.8
ARG TORCH_VERSION=2.0.1
ARG TORCHVISION_VERSION=0.15.2
ARG MMCV_VERSION=1.7.2
ARG CUDA_SHORT=cu118

ENV DEBIAN_FRONTEND=noninteractive
ENV FORCE_CUDA="1"
# Compile CUDA ops for Ampere (3090/A100) + Ada Lovelace (RTX 4070 = 8.9).
ENV TORCH_CUDA_ARCH_LIST="8.0;8.6;8.9"
ENV CUDA_HOME=/usr/local/cuda

# System libs (mirrors removed — original Dockerfile used Tsinghua/Aliyun which
# are slow outside China; if you're behind GFW pass --build-arg to re-enable).
RUN apt-get update && \
    apt-get install -y --no-install-recommends \
        vim git wget curl ca-certificates \
        libsm6 libxext6 libxrender-dev libgl1-mesa-glx \
        libssl-dev libopencv-dev libspdlog-dev \
        build-essential ninja-build && \
    rm -rf /var/lib/apt/lists/*

# Miniforge (conda-forge only) — original Dockerfile used Miniconda, but Anaconda's
# new ToS for pkgs/main and pkgs/r breaks `conda install` in non-interactive builds
# (CondaToSNonInteractiveError). Miniforge sidesteps this entirely.
RUN curl -fsSL -o /tmp/miniforge.sh \
        https://github.com/conda-forge/miniforge/releases/latest/download/Miniforge3-Linux-x86_64.sh && \
    bash /tmp/miniforge.sh -b -p /opt/conda && \
    rm /tmp/miniforge.sh && \
    /opt/conda/bin/conda install -y -n base -c conda-forge python=${PYTHON_VERSION} \
        conda-build pyyaml numpy ipython cython typing_extensions mkl mkl-include ninja && \
    /opt/conda/bin/conda clean -ya
ENV PATH=/opt/conda/bin:$PATH

# PyTorch 2.0.1 + cu118 — first stable release with sm_89 in _get_cuda_arch_flags
RUN pip install --no-cache-dir \
        torch==${TORCH_VERSION} \
        torchvision==${TORCHVISION_VERSION} \
        --index-url https://download.pytorch.org/whl/${CUDA_SHORT}

# mmcv-full 1.7.2 prebuilt wheel for torch 2.0 + cu118.
# Note: BEVDet's mminstall.txt pins mmcv-full<=1.6.0, but that constraint only
# flows through setup.py's extras_require['mim'], not install_requires — the
# editable install in postCreateCommand won't enforce it.
RUN pip install --no-cache-dir mmcv-full==${MMCV_VERSION} \
        -f https://download.openmmlab.com/mmcv/dist/${CUDA_SHORT}/torch2.0/index.html

# mmdet / mmsegmentation — bumped to the last 2.x releases that are tested
# against mmcv 1.7 and PyTorch 2.0. BEVDet uses mmdet 2.x APIs, not 3.x.
RUN pip install --no-cache-dir \
        mmdet==2.28.2 \
        mmsegmentation==0.30.0

# BEVDet runtime deps — kept aligned with docker/Dockerfile, but numba and spconv
# bumped for the newer toolchain (numba 0.53 won't build against modern numpy;
# spconv flavor must match CUDA 11.8).
RUN pip install --no-cache-dir \
        pycuda \
        lyft_dataset_sdk \
        networkx==2.2 \
        numba==0.56.4 \
        "numpy<1.24" \
        nuscenes-devkit \
        plyfile \
        scikit-image \
        tensorboard \
        trimesh==2.35.39 \
        spconv-cu118 \
        open3d \
        onnxruntime-gpu

WORKDIR /workspace
