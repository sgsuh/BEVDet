# BEVDet


![](./resources/nds-fps-dal.png)


## News
- **2024.07.01** DAL is accepted to ECCV24.
- **2023.11.08** Support DAL for 3D object detection with LiDAR-camera fusion. [[Arxiv](https://arxiv.org/abs/2311.07152)]

- [History](./docs/en/news.md)

## Main Results
### Nuscenes Detection
| Config                                                                    | mAP        | NDS        | Latency(ms) | FPS  | Model                                                                                          | Log                                                                                            |
| ------------------------------------------------------------------------- | ---------- | ---------- | ---- | ---- | ---------------------------------------------------------------------------------------------- | ---------------------------------------------------------------------------------------------- |
| [**BEVDet-R50**](configs/bevdet/bevdet-r50.py)                            | 28.3       | 35.0       | 29.1/4.2/33.3| 30.7 | [baidu](https://pan.baidu.com/s/1237QyV18zvRJ1pU3YzRItw?pwd=npe1) | [baidu](https://pan.baidu.com/s/1237QyV18zvRJ1pU3YzRItw?pwd=npe1) |
| [**BEVDet-R50-CBGS**](configs/bevdet/bevdet-r50-cbgs.py)                  | 31.3       | 39.8       |28.9/4.3/33.2 |30.1 | [baidu](https://pan.baidu.com/s/1237QyV18zvRJ1pU3YzRItw?pwd=npe1) | [baidu](https://pan.baidu.com/s/1237QyV18zvRJ1pU3YzRItw?pwd=npe1) |
| [**BEVDet-R50-4D-CBGS**](configs/bevdet/bevdet-r50-4d-cbgs.py) | 31.4/35.4# | 44.7/44.9# | 29.1/4.3/33.4|30.0 | [baidu](https://pan.baidu.com/s/1237QyV18zvRJ1pU3YzRItw?pwd=npe1) |[baidu](https://pan.baidu.com/s/1237QyV18zvRJ1pU3YzRItw?pwd=npe1)|
| [**BEVDet-R50-4D-Depth-CBGS**](configs/bevdet/bevdet-r50-4d-depth-cbgs.py) | 36.1/36.2# | 48.3/48.4# |35.7/4.0/39.7 |25.2 | [baidu](https://pan.baidu.com/s/1237QyV18zvRJ1pU3YzRItw?pwd=npe1) | [baidu](https://pan.baidu.com/s/1237QyV18zvRJ1pU3YzRItw?pwd=npe1) |
| [**BEVDet-R50-4D-Stereo-CBGS**](configs/bevdet/bevdet-r50-4d-stereo-cbgs.py) | 38.2/38.4# | 49.9/50.0# |-  |-  | [baidu](https://pan.baidu.com/s/1237QyV18zvRJ1pU3YzRItw?pwd=npe1) | [baidu](https://pan.baidu.com/s/1237QyV18zvRJ1pU3YzRItw?pwd=npe1) |
| [**BEVDet-R50-4DLongterm-CBGS**](configs/bevdet/bevdet-r50-4dlongterm-cbgs.py) | 34.8/35.4# | 48.2/48.7# | 30.8/4.2/35.0|28.6 | [baidu](https://pan.baidu.com/s/1237QyV18zvRJ1pU3YzRItw?pwd=npe1) | [baidu](https://pan.baidu.com/s/1237QyV18zvRJ1pU3YzRItw?pwd=npe1) |
| [**BEVDet-R50-4DLongterm-Depth-CBGS**](configs/bevdet/bevdet-r50-4d-depth-cbgs.py) | 39.4/39.9# | 51.5/51.9# |38.4/4.0/42.4 |23.6 | [baidu](https://pan.baidu.com/s/1237QyV18zvRJ1pU3YzRItw?pwd=npe1) | [baidu](https://pan.baidu.com/s/1237QyV18zvRJ1pU3YzRItw?pwd=npe1) |
| [**BEVDet-R50-4DLongterm-Stereo-CBGS**](configs/bevdet/bevdet-r50-4dlongterm-stereo-cbgs.py) | 41.1/41.5# | 52.3/52.7# |- |- | [baidu](https://pan.baidu.com/s/1237QyV18zvRJ1pU3YzRItw?pwd=npe1) | [baidu](https://pan.baidu.com/s/1237QyV18zvRJ1pU3YzRItw?pwd=npe1) |
| [**BEVDet-STBase-4D-Stereo-512x1408-CBGS**](configs/bevdet/bevdet-stbase-4d-stereo-512x1408-cbgs.py) | 47.2# | 57.6# |-  |-  | [baidu](https://pan.baidu.com/s/1237QyV18zvRJ1pU3YzRItw?pwd=npe1) | [baidu](https://pan.baidu.com/s/1237QyV18zvRJ1pU3YzRItw?pwd=npe1) |
||
| [**DAL-Tiny**](configs/dal/dal-tiny.py) | 67.4 | 71.3 |-  |16.6 | [baidu](https://pan.baidu.com/s/15rmJL_SWUeQEXG9dYYl8gA?pwd=36g5) | [baidu](https://pan.baidu.com/s/15rmJL_SWUeQEXG9dYYl8gA?pwd=36g5) |
| [**DAL-Base**](configs/dal/dal-base.py) | 70.0 | 73.4 |-  |10.7 | [baidu](https://pan.baidu.com/s/15rmJL_SWUeQEXG9dYYl8gA?pwd=36g5) | [baidu](https://pan.baidu.com/s/15rmJL_SWUeQEXG9dYYl8gA?pwd=36g5) |
| [**DAL-Large**](configs/dal/dal-large.py) | 71.5 | 74.0 |-  |6.10 | [baidu](https://pan.baidu.com/s/15rmJL_SWUeQEXG9dYYl8gA?pwd=36g5) | [baidu](https://pan.baidu.com/s/15rmJL_SWUeQEXG9dYYl8gA?pwd=36g5) |

\# align previous frame bev feature during the view transformation.

Depth: Depth supervised from Lidar as BEVDepth.

Longterm: cat 8 history frame in temporal modeling. 1 by default. 

Stereo: A private implementation that concat cost-volumn with image feature before executing model.view_transformer.depth_net.

The latency includes Network/Post-Processing/Total. Training without CBGS is deprecated.


### Nuscenes Occupancy
| Config                                                                    | mIOU       | Model | Log                                                                                            |
| ------------------------------------------------------------------------- | ---------- | ---------------------------------------------------------------------------------------------- | ---------------------------------------------------------------------------------------------- |
| [**BEVDet-Occ-R50-4D-Stereo-2x**](configs/bevdet_occ/bevdet-occ-r50-4d-stereo-24e.py)                                 | 36.1     | [baidu](https://pan.baidu.com/s/1237QyV18zvRJ1pU3YzRItw?pwd=npe1) | [baidu](https://pan.baidu.com/s/1237QyV18zvRJ1pU3YzRItw?pwd=npe1) |
| [**BEVDet-Occ-R50-4D-Stereo-2x-384x704**](configs/bevdet_occ/bevdet-occ-r50-4d-stereo-24e_384704.py)                  | 37.3     | [baidu](https://pan.baidu.com/s/1237QyV18zvRJ1pU3YzRItw?pwd=npe1) | [baidu](https://pan.baidu.com/s/1237QyV18zvRJ1pU3YzRItw?pwd=npe1) |
| [**BEVDet-Occ-R50-4DLongterm-Stereo-2x-384x704**](configs/bevdet_occ/bevdet-occ-r50-4dlongterm-stereo-24e_384704.py)  | 39.3     | [baidu](https://pan.baidu.com/s/1237QyV18zvRJ1pU3YzRItw?pwd=npe1) | [baidu](https://pan.baidu.com/s/1237QyV18zvRJ1pU3YzRItw?pwd=npe1) |
| [**BEVDet-Occ-STBase-4D-Stereo-2x**](configs/bevdet_occ/bevdet-occ-stbase-4d-stereo-512x1408-24e.py)                  | 42.0     | [baidu](https://pan.baidu.com/s/1237QyV18zvRJ1pU3YzRItw?pwd=npe1) | [baidu](https://pan.baidu.com/s/1237QyV18zvRJ1pU3YzRItw?pwd=npe1) |
## Inference latency with different backends

| Backend       | 256x704 | 384x1056 | 512x1408 | 640x1760 |
| ------------- | ------- | -------- | -------- | -------- |
| PyTorch       | 28.9    | 49.7     | 78.7    | 113.4    |
| TensorRT      | 14.0    | 22.8     | 36.5     | 53.0     |
| TensorRT-FP16 | 4.94     | 7.96     | 12.4     | 17.9     |
| TensorRT-INT8 | 2.93    | 4.41      | 6.58      | 9.19     |                                      
| TensorRT-INT8(Xavier) | 25.0    | -      | -     | -    | 

- Evaluate with [**BEVDet-R50-CBGS**](configs/bevdet/bevdet-r50-cbgs.py) on a RTX 3090 GPU by default. We omit the postprocessing, which spends up to 5 ms with the PyTorch backend.

## Get Started

#### Installation and Data Preparation

step 1. Please prepare environment as that in [Docker](docker/Dockerfile).

step 2. Prepare bevdet repo by.
```shell script
git clone https://github.com/HuangJunJie2017/BEVDet.git
cd BEVDet
pip install -v -e .
```

step 3. Prepare nuScenes dataset as introduced in [nuscenes_det.md](docs/en/datasets/nuscenes_det.md) and create the pkl for BEVDet by running:
```shell
python tools/create_data_bevdet.py
```
step 4. For Occupancy Prediction task, download (only) the 'gts' from [CVPR2023-3D-Occupancy-Prediction](https://github.com/CVPR2023-3D-Occupancy-Prediction/CVPR2023-3D-Occupancy-Prediction) and arrange the folder as:
```shell script
└── nuscenes
    ├── v1.0-trainval (existing)
    ├── sweeps  (existing)
    ├── samples (existing)
    └── gts (new)
```

#### Train model
```shell
# single gpu
python tools/train.py $config
# multiple gpu
./tools/dist_train.sh $config num_gpu
```

#### Test model
```shell
# single gpu
python tools/test.py $config $checkpoint --eval mAP
# multiple gpu
./tools/dist_test.sh $config $checkpoint num_gpu --eval mAP
```

#### Estimate the inference speed of BEVDet

```shell
# with pre-computation acceleration
python tools/analysis_tools/benchmark.py $config $checkpoint --fuse-conv-bn
# 4D with pre-computation acceleration
python tools/analysis_tools/benchmark_sequential.py $config $checkpoint --fuse-conv-bn
# view transformer only
python tools/analysis_tools/benchmark_view_transformer.py $config $checkpoint
```

#### Estimate the flops of BEVDet

```shell
python tools/analysis_tools/get_flops.py configs/bevdet/bevdet-r50.py --shape 256 704
```

#### Visualize the predicted result.

- Private implementation. (Visualization remotely/locally)

```shell
python tools/test.py $config $checkpoint --format-only --eval-options jsonfile_prefix=$savepath
python tools/analysis_tools/vis.py $savepath/pts_bbox/results_nusc.json
```

#### Convert to TensorRT and test inference speed.

```shell
1. install mmdeploy from https://github.com/HuangJunJie2017/mmdeploy
2. convert to TensorRT
python tools/convert_bevdet_to_TRT.py $config $checkpoint $work_dir --fuse-conv-bn --fp16 --int8
3. test inference speed
python tools/analysis_tools/benchmark_trt.py $config $engine
```

## Deployment (deploy branch)

The `deploy` branch adds an end-to-end deployment path for **BEVDet-R50** on a
single consumer GPU (developed/verified on an RTX 4070), including a reproducible
devcontainer, TensorRT export, and a standalone C++ runtime.

#### Devcontainer

A `.devcontainer/` is provided that builds CUDA + TensorRT, the `mmdeploy`
submodule, and the C++ host dependencies (Eigen, yaml-cpp, nlohmann-json,
OpenCV). Open the repo in VS Code and "Reopen in Container", or build the image
directly from `.devcontainer/Dockerfile`. Pull the submodule first:

```shell
git submodule update --init --recursive
```

#### Verify the PyTorch model

Convenience wrappers around the standard tools, pinned to
`configs/bevdet/bevdet-r50.py` + `ckpts/bevdet-r50.pth`:

```shell
bash tools/test_bevdet_r50.sh        # mAP/NDS eval (nuScenes v1.0-mini supported)
bash tools/benchmark_bevdet_r50.sh   # PyTorch inference speed
bash tools/get_flops_bevdet_r50.sh   # FLOPs (256x704)
bash tools/vis_bevdet_r50.sh         # render predictions to work_dirs/bevdet-r50-vis
```

#### Export to TensorRT

```shell
# build the engine (FP16, INT8 calibration) into work_dirs/
bash tools/convert_bevdet_to_TRT.sh
# test TensorRT inference speed
bash tools/benchmark_trt_bevdet_r50.sh
```

This produces `work_dirs/bevdet_int8_fuse.{onnx,engine}` and relies on the
`bev_pool_v2` mmdeploy plugin.

#### C++ TensorRT runtime (`cpp/`)

A standalone C++ application that runs the exported engine directly on nuScenes
frames and saves 6-camera + BEV detection overlays — no Python at inference time.
Highlights:

- `nuscenes_loader` parses `v1.0-mini` JSON into per-keyframe inputs (image
  paths, intrinsics, sensor-to-lidar and ego/lidar poses).
- The `bev_pool_v2` TRT plugin is reimplemented in C++/CUDA.
- Post-processing matches the 1-task / 10-class CenterHead: heatmap-channel class
  labels, sigmoid on the heatmap logits, and `exp()` on the (log-space) box
  dimensions.

```shell
# 1) dump the bev_pool ranks/intervals meta once (reused for every frame)
python tools/export_bev_pool_meta.py \
    configs/bevdet/bevdet-r50.py ckpts/bevdet-r50.pth \
    --out-dir work_dirs/meta

# 2) build
cd cpp && mkdir -p build && cd build && cmake .. && cmake --build . -j

# 3) run (paths/options in cpp/cfg/params.yaml)
./bevdet
```

Overlay PNGs are written to the `vis_output_dir` set in `cpp/cfg/params.yaml`
(default `work_dirs/cpp_vis/`).

## Acknowledgement

This project is not possible without multiple great open-sourced code bases. We list some notable examples below.

- [open-mmlab](https://github.com/open-mmlab)
- [CenterPoint](https://github.com/tianweiy/CenterPoint)
- [Lift-Splat-Shoot](https://github.com/nv-tlabs/lift-splat-shoot)
- [Swin Transformer](https://github.com/microsoft/Swin-Transformer)
- [BEVFusion](https://github.com/mit-han-lab/bevfusion)
- [BEVDepth](https://github.com/Megvii-BaseDetection/BEVDepth)

Beside, there are some other attractive works extend the boundary of BEVDet.

- [BEVerse](https://github.com/zhangyp15/BEVerse)  for multi-task learning.
- [BEVStereo](https://github.com/Megvii-BaseDetection/BEVStereo)  for stero depth estimation.

## Bibtex

If this work is helpful for your research, please consider citing the following BibTeX entries.

```
@article{huang2023dal,
  title={Detecting As Labeling: Rethinking LiDAR-camera Fusion in 3D Object Detection},
  author={Huang, Junjie and Ye, Yun and Liang, Zhujin and Shan, Yi and Du, Dalong},
  journal={arXiv preprint arXiv:2311.07152},
  year={2023}
}

@article{huang2022bevpoolv2,
  title={BEVPoolv2: A Cutting-edge Implementation of BEVDet Toward Deployment},
  author={Huang, Junjie and Huang, Guan},
  journal={arXiv preprint arXiv:2211.17111},
  year={2022}
}

@article{huang2022bevdet4d,
  title={BEVDet4D: Exploit Temporal Cues in Multi-camera 3D Object Detection},
  author={Huang, Junjie and Huang, Guan},
  journal={arXiv preprint arXiv:2203.17054},
  year={2022}
}

@article{huang2021bevdet,
  title={BEVDet: High-performance Multi-camera 3D Object Detection in Bird-Eye-View},
  author={Huang, Junjie and Huang, Guan and Zhu, Zheng and Yun, Ye and Du, Dalong},
  journal={arXiv preprint arXiv:2112.11790},
  year={2021}
}
```
