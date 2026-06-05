"""Export bev_pool_v2 ranks/intervals as raw .bin files for the C++ TRT runner.

The TensorRT engine produced by tools/convert_bevdet_to_TRT.py expects
five auxiliary inputs (ranks_depth, ranks_feat, ranks_bev, interval_starts,
interval_lengths) whose shapes were baked into the engine at build time.
Because BDA is identity at inference and the camera calibration is constant
across a deployment, a single set of meta tensors is reused for every frame.

This script builds the model from the same config used for ONNX export,
calls get_bev_pool_input on one dataloader sample, and dumps each tensor
to a .bin file (int32, little-endian) that the C++ allocMeta() can mmap.
"""
import argparse
import os

import numpy as np
import torch
from mmcv import Config
from mmcv.runner import load_checkpoint

try:
    from mmdet.utils import compat_cfg
except ImportError:
    from mmdet3d.utils import compat_cfg

from mmdet3d.datasets import build_dataloader, build_dataset
from mmdet3d.models import build_model


def parse_args():
    parser = argparse.ArgumentParser(
        description='Dump bev_pool_v2 ranks/intervals tensors as .bin')
    parser.add_argument('config', help='model config (same as used for export)')
    parser.add_argument('checkpoint', help='checkpoint .pth')
    parser.add_argument('--out-dir', default='work_dirs/meta',
                        help='directory to write the five .bin files into')
    return parser.parse_args()


def main():
    args = parse_args()
    os.makedirs(args.out_dir, exist_ok=True)

    cfg = Config.fromfile(args.config)
    cfg.model.pretrained = None
    cfg.model.type = cfg.model.type + 'TRT'
    cfg = compat_cfg(cfg)
    cfg.gpu_ids = [0]

    test_loader_cfg = {
        'samples_per_gpu': 1,
        'workers_per_gpu': 0,
        'dist': False,
        'shuffle': False,
        **cfg.data.get('test_dataloader', {}),
    }
    if isinstance(cfg.data.test, dict):
        cfg.data.test.test_mode = True
    dataset = build_dataset(cfg.data.test)
    data_loader = build_dataloader(dataset, **test_loader_cfg)

    cfg.model.train_cfg = None
    model = build_model(cfg.model, test_cfg=cfg.get('test_cfg'))
    load_checkpoint(model, args.checkpoint, map_location='cpu')
    model.cuda()
    model.eval()

    data = next(iter(data_loader))
    inputs = [t.cuda() for t in data['img_inputs'][0]]
    with torch.no_grad():
        metas = model.get_bev_pool_input(inputs)

    # Index order matches tools/analysis_tools/benchmark_trt.py:165-169:
    #   metas[0] = ranks_bev, metas[1] = ranks_depth, metas[2] = ranks_feat,
    #   metas[3] = interval_starts, metas[4] = interval_lengths
    names = ['ranks_bev', 'ranks_depth', 'ranks_feat',
             'interval_starts', 'interval_lengths']
    for name, tensor in zip(names, metas):
        arr = tensor.int().cpu().numpy().astype(np.int32)
        path = os.path.join(args.out_dir, f'{name}.bin')
        arr.tofile(path)
        print(f'  {name:18s} shape={tuple(arr.shape)}  -> {path}')


if __name__ == '__main__':
    main()
