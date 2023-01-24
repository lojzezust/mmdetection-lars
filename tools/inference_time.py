import argparse
import torch
from tqdm.auto import tqdm
from time import time
import numpy as np
import os
import os.path as osp
import json

import mmcv
from mmcv.runner import wrap_fp16_model

from mmcv.cnn import fuse_conv_bn
from mmdet.datasets import build_dataloader, build_dataset
from mmdet.models import build_detector
from mmdet.utils import (build_dp, compat_cfg, get_device,
                         replace_cfg_vals, update_data_root)

NUM_SAMPLES = 200
NUM_REPEATS = 5
OUTPUT_DIR = 'output/latency'

def get_args():
    parser = argparse.ArgumentParser()
    parser.add_argument('config', type=str, help='Path to the model configuration file')
    parser.add_argument('-n', '--num_samples', type=int, default=NUM_SAMPLES,
                        help='Number of data samples used in the test.')
    parser.add_argument('-r', '--repeats', type=int, default=NUM_REPEATS,
                        help='Number of repetitions.')
    parser.add_argument('-o', '--output_dir', type=str, default=OUTPUT_DIR,
                        help='Output directory.')

    args = parser.parse_args()
    return args

def count_parameters(model):
    return sum(p.numel() for p in model.parameters() if p.requires_grad)

def eval_inference_time(args):
    cfg = mmcv.Config.fromfile(args.config)

    # replace the ${key} with the value of cfg.key
    cfg = replace_cfg_vals(cfg)

    # update data root according to MMDET_DATASETS
    update_data_root(cfg)

    cfg = compat_cfg(cfg)

    # set cudnn_benchmark
    if cfg.get('cudnn_benchmark', False):
        torch.backends.cudnn.benchmark = True

    if 'pretrained' in cfg.model:
        cfg.model.pretrained = None
    elif 'init_cfg' in cfg.model.backbone:
        cfg.model.backbone.init_cfg = None

    if cfg.model.get('neck'):
        if isinstance(cfg.model.neck, list):
            for neck_cfg in cfg.model.neck:
                if neck_cfg.get('rfp_backbone'):
                    if neck_cfg.rfp_backbone.get('pretrained'):
                        neck_cfg.rfp_backbone.pretrained = None
        elif cfg.model.neck.get('rfp_backbone'):
            if cfg.model.neck.rfp_backbone.get('pretrained'):
                cfg.model.neck.rfp_backbone.pretrained = None


    # Build dataset
    dataset = build_dataset(cfg.data.test)

    cfg.gpu_ids = [0]
    cfg.device = get_device()

    test_dataloader_default_args = dict(
        samples_per_gpu=1, workers_per_gpu=2, dist=False, shuffle=False)

    test_loader_cfg = {
        **test_dataloader_default_args,
        **cfg.data.get('test_dataloader', {})
    }


    # build the dataloader
    dataset = build_dataset(cfg.data.test)
    data_loader = build_dataloader(dataset, **test_loader_cfg)

    # build the model and load checkpoint
    cfg.model.train_cfg = None
    model = build_detector(cfg.model, test_cfg=cfg.get('test_cfg'))

    # Get number of parameters
    num_params = count_parameters(model)

    fp16_cfg = cfg.get('fp16', None)
    if fp16_cfg is not None:
        wrap_fp16_model(model)

    # Prepare for inference
    torch.cuda.empty_cache()
    model = build_dp(model, cfg.device, device_ids=cfg.gpu_ids)
    model.eval()

    # Preload data
    data = []
    for i,d in tqdm(enumerate(data_loader), desc='Reading data', total=args.num_samples):
        if i >= args.num_samples:
            break
        data.append(d)


    with torch.no_grad():
        # Network warm-up
        for d in tqdm(data, desc='Network warm-up'):
            model(return_loss=False, rescale=True, **d)

        # Time inference
        times = []
        for it in range(args.repeats):
            start_time = time()
            for d in tqdm(data, desc='Running inference %d/%d' % (it+1, args.repeats)):
                model(return_loss=False, rescale=True, **d)

            end_time = time()
            elapsed = end_time - start_time
            times.append(elapsed)

    # Compute and print statistics
    times = np.array(times)
    per_img = times / args.num_samples
    fps = 1. / per_img

    per_img_m = per_img.mean()
    per_img_std = per_img.std()

    fps_m = fps.mean()
    fps_std = fps.std()

    print('Latency: %.0f ± %.1f ms, FPS: %.1f ± %.2f' % (per_img_m * 1000, 2 * per_img_std * 1000, fps_m, 2 * fps_std))

    summary = {
        'num_samples': args.num_samples,
        'times': times.tolist(),
        'latency': {'mean': per_img_m, 'std': per_img_std},
        'fps': {'mean': fps_m, 'std': fps_std},
        'num_params': num_params
    }

    # Save summary to JSON
    if not osp.exists(OUTPUT_DIR):
        os.makedirs(OUTPUT_DIR)

    cfg_name = osp.splitext(osp.basename(args.config))[0]
    with open(osp.join(OUTPUT_DIR, cfg_name + '.json'), 'w') as file:
        json.dump(summary, file)


if __name__ == '__main__':
    args = get_args()
    eval_inference_time(args)
