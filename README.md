# MMDetection-LaRS

Fork of [MMDetection](https://github.com/open-mmlab/mmdetection) with support for training on LaRS Panoptic dataset.

## Quickstart

1. Convert LaRS panoptic annotations to mmdetection format using [`tools/dataset_converters/LaRSv2.py`](tools/dataset_converters/LaRSv2.py).
2. You can start training from one of the provided training configs (e.g. [`configs/mask2former/mask2former_r50_lsj_8x2_1x_LaRS.py`](configs/mask2former/mask2former_r50_lsj_8x2_1x_LaRS.py))
