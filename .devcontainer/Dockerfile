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
        "pycuda==2024.1.2" \
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

# ============================================================================
# MMDeploy (HuangJunJie2017 fork) — TensorRT-only deployment path.
# Follows docs/en/01-how-to-build/linux-x86_64.md, skipping the ONNX Runtime
# and OpenVINO sections. The mmdeploy source itself lives as a submodule at
# /workspace/BEVDet/mmdeploy and is built in devcontainer.json's
# postCreateCommand (so edits to the bind-mounted source take effect).
# ============================================================================

# Build tools: mmdeploy's CMakeLists requires cmake>=3.14 and C++17 (g++>=7).
# ubuntu20.04 apt cmake is 3.16.3 and g++ is 9.x — both fine. build-essential
# / ninja-build were installed earlier.
RUN apt-get update && apt-get install -y --no-install-recommends \
        cmake pkg-config && \
    rm -rf /var/lib/apt/lists/*

# TensorRT 8.6.1 — the GA release that supports sm_89 (Ada Lovelace / RTX 4070)
# on CUDA 11.8. The NVIDIA CUDA apt repo configured by the base image hosts
# these packages. Pin to a single version across all TRT components to avoid
# apt picking mismatched majors.
ARG TENSORRT_VERSION=8.6.1.6-1+cuda11.8
RUN apt-get update && apt-get install -y --no-install-recommends \
        libnvinfer8=${TENSORRT_VERSION} \
        libnvinfer-dev=${TENSORRT_VERSION} \
        libnvinfer-plugin8=${TENSORRT_VERSION} \
        libnvinfer-plugin-dev=${TENSORRT_VERSION} \
        libnvinfer-headers-dev=${TENSORRT_VERSION} \
        libnvinfer-headers-plugin-dev=${TENSORRT_VERSION} \
        libnvonnxparsers8=${TENSORRT_VERSION} \
        libnvonnxparsers-dev=${TENSORRT_VERSION} \
        libnvparsers8=${TENSORRT_VERSION} \
        libnvparsers-dev=${TENSORRT_VERSION} && \
    rm -rf /var/lib/apt/lists/*

# mmdeploy's FindTENSORRT.cmake / FindCUDNN.cmake probe ${TENSORRT_DIR}/{include,lib}
# and ${CUDNN_DIR}/{include,lib(64)}. apt installs into Debian's multiarch
# layout (/usr/include/x86_64-linux-gnu, /usr/lib/x86_64-linux-gnu) which those
# finders don't search — so collect symlinks into /opt with the expected shape.
RUN mkdir -p /opt/tensorrt/include /opt/tensorrt/lib \
             /opt/cudnn/include    /opt/cudnn/lib && \
    for f in /usr/include/x86_64-linux-gnu/Nv*.h; do ln -s "$f" /opt/tensorrt/include/; done && \
    for f in /usr/lib/x86_64-linux-gnu/libnvinfer*       \
             /usr/lib/x86_64-linux-gnu/libnvonnxparsers* \
             /usr/lib/x86_64-linux-gnu/libnvparsers*;    do ln -s "$f" /opt/tensorrt/lib/; done && \
    for f in /usr/include/cudnn*.h; do ln -s "$f" /opt/cudnn/include/; done && \
    for f in /usr/lib/x86_64-linux-gnu/libcudnn*; do ln -s "$f" /opt/cudnn/lib/; done

ENV TENSORRT_DIR=/opt/tensorrt
ENV CUDNN_DIR=/opt/cudnn
ENV LD_LIBRARY_PATH=${TENSORRT_DIR}/lib:${CUDNN_DIR}/lib:${LD_LIBRARY_PATH}

# TensorRT Python bindings for the conda env. Version pinned to the apt-installed
# C++ libs so the loader doesn't see two different libnvinfer majors.
RUN pip install --no-cache-dir \
        --extra-index-url https://pypi.nvidia.com \
        tensorrt==8.6.1

# mmdeploy runtime + build Python deps (runtime.txt + build.txt of the fork).
# protobuf is pinned <=3.20.1 by mmdeploy because onnx-trt parsers break on newer.
RUN pip install --no-cache-dir \
        "protobuf<=3.20.1" \
        onnx \
        h5py \
        multiprocess \
        terminaltables \
        grpcio \
        packaging

# ppl.cv v0.7.0 (CUDA backend). Only needed if you later build the MMDeploy SDK
# (-DMMDEPLOY_BUILD_SDK=ON); the model-converter + custom-ops path doesn't use
# it. Building here so the image is ready for both.
RUN git clone --depth 1 --branch v0.7.0 https://github.com/openppl-public/ppl.cv.git /opt/pplcv && \
    cd /opt/pplcv && \
    ./build.sh cuda
ENV PPLCV_DIR=/opt/pplcv

WORKDIR /workspace
