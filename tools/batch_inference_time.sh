#!/bin/sh

CONFIGS=(
    configs/panoptic_fpn/panoptic_fpn_r50_fpn_10x_LaRS.py
    configs/panoptic_fpn/panoptic_fpn_r101_fpn_10x_LaRS.py
    configs/mask2former/mask2former_r50_lsj_8x2_10x_LaRS.py
    configs/mask2former/mask2former_r101_lsj_8x2_10x_LaRS.py
    configs/mask2former/mask2former_swin-t-p4-w7-224_lsj_8x2_10x_LaRS.py
    configs/mask2former/mask2former_swin-b-p4-w12-384-in21k_lsj_8x2_20x_LaRS.py
)

for CONFIG in "${CONFIGS[@]}"
do
    echo $CONFIG
    python tools/inference_time.py $CONFIG
done
