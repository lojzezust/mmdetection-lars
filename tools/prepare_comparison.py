import os
from shutil import copy
from rich.progress import track

PREDICTIONS_DIR = '/d/hpc/projects/FRI/lzust/development/mmdetection/output/predictions'
IMAGE_DIR = '/d/hpc/projects/FRI/lzust/data/datasets/LaRS/images'
IMAGE_LIST = '/d/hpc/projects/FRI/lzust/development/mmdetection/output/export_list.txt'
OUTPUT_DIR = '/d/hpc/projects/FRI/lzust/development/mmdetection/output/comparison/2022_08_22'

METHODS = [
    ('mask2former_r50', '/d/hpc/projects/FRI/lzust/development/mmdetection/output/predictions/mask2former_r50_lsj_8x2_10x_LaRS', '.jpg'),
    ('mask2former_swin', '/d/hpc/projects/FRI/lzust/development/mmdetection/output/predictions/mask2former_swin-b-p4-w12-384-in21k_lsj_8x2_10x_LaRS', '.jpg'),
    ('panfpn_r50', '/d/hpc/projects/FRI/lzust/development/mmdetection/output/predictions/panoptic_fpn_r50_fpn_1x_LaRS', '.jpg'),
    ('panfpn_r101', '/d/hpc/projects/FRI/lzust/development/mmdetection/output/predictions/panoptic_fpn_r101_fpn_2x_LaRS', '.jpg'),
    ('wasr', '/d/hpc/projects/FRI/lzust/data/predictions/lars/segmentation/wasr_lars_v1', '.png'),
    ('wasrT', '/d/hpc/projects/FRI/lzust/data/predictions/lars/segmentation/wasrt_lars_v1', '.png'),
]


with open(IMAGE_LIST, 'r') as file:
    image_list = [l.strip() for l in file]

if not os.path.exists(OUTPUT_DIR):
    os.makedirs(OUTPUT_DIR)

for image_name in track(image_list):
    in_path = os.path.join(IMAGE_DIR, image_name + '.jpg')
    out_path = os.path.join(OUTPUT_DIR, image_name + '.jpg')
    copy(in_path, out_path)

    for method_name, pred_dir, ext in METHODS:
        in_path = os.path.join(pred_dir, image_name + ext)
        if not os.path.exists(in_path):
            print('Missing:', in_path)
            continue
        out_path = os.path.join(OUTPUT_DIR,'%s_%s%s' % (image_name, method_name, ext))
        copy(in_path, out_path)
