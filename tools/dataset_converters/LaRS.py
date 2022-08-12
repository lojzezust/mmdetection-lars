# Adapted from Cityscapes script (https://github.com/open-mmlab/mmdetection/blob/master/tools/dataset_converters/cityscapes.py)
# Updated to panoptic

import argparse
import os.path as osp

import mmcv
import numpy as np
import pycocotools.mask as maskUtils
try:
    from panopticapi.utils import id2rgb
except ImportError:
    id2rgb = None

THINGS_START=100
IGNORE_I = 4
DYN_OBST_I = 3

def rand_color(existing):
    color = None
    while color is None or color in existing:
        color = np.random.randint(1, 256*256*256)

    return color

def collect_files(split_file, img_dir, seg_dir, inst_dir, segments_dir):
    with open(split_file, 'r') as file:
        images = [l.strip() for l in file]

    files = []
    for img_name in images:
        img_file = osp.join(img_dir, '%s.jpg' % img_name)
        seg_file = osp.join(seg_dir, '%s.png' % img_name)
        inst_file = osp.join(inst_dir, '%s.png' % img_name)
        out_seg_file = osp.join(segments_dir, '%s.png' % img_name)

        files.append((img_file, seg_file, inst_file, out_seg_file))
    print(f'Loaded {len(files)} images from {split_file}')

    return files


def collect_annotations(files, nproc=1):
    print('Loading annotation images')
    if nproc > 1:
        images = mmcv.track_parallel_progress(
            load_img_info, files, nproc=nproc)
    else:
        images = mmcv.track_progress(load_img_info, files)

    return images


def load_img_info(files):
    img_file, seg_file, inst_file, out_seg_file = files
    inst_img = mmcv.imread(inst_file, 'unchanged')
    seg_img = mmcv.imread(seg_file, 'unchanged')

    seg_img[inst_img>0] = IGNORE_I # Ignore dynamic obstacles in stuff

    seg_mask = np.zeros_like(seg_img, dtype=np.int)
    existing_colors = set()

    # Stuff
    segments_info = []
    for cat_id in np.unique(seg_img):
        if cat_id == IGNORE_I:
            continue

        category_id = cat_id
        mask = np.asarray(seg_img == cat_id, dtype=np.uint8, order='F')
        mask_rle = maskUtils.encode(mask[:, :, None])[0]

        area = maskUtils.area(mask_rle)
        # convert to COCO style XYWH format
        bbox = maskUtils.toBbox(mask_rle)

        # Assign random color to segment
        seg_id = rand_color(existing_colors)
        existing_colors.add(seg_id)
        seg_mask[seg_img==cat_id] = seg_id


        anno = dict(
            id=seg_id,
            category_id=category_id,
            iscrowd=0,
            bbox=bbox.tolist(),
            area=area.tolist())
        segments_info.append(anno)

    # Things
    for obj_id in np.unique(inst_img[inst_img > 0]):
        # All things are dynamic obstacles (id=3)
        category_id = DYN_OBST_I
        mask = np.asarray(inst_img == obj_id, dtype=np.uint8, order='F')
        mask_rle = maskUtils.encode(mask[:, :, None])[0]

        area = maskUtils.area(mask_rle)
        # convert to COCO style XYWH format
        bbox = maskUtils.toBbox(mask_rle)

        # Assign random color to segment
        seg_id = rand_color(existing_colors)
        existing_colors.add(seg_id)
        seg_mask[inst_img == obj_id] = seg_id


        anno = dict(
            id=seg_id,
            category_id=category_id,
            iscrowd=0,
            bbox=bbox.tolist(),
            area=area.tolist())
        segments_info.append(anno)

    # Save RGB id mask
    seg_mask_rgb = id2rgb(seg_mask)
    seg_mask_rgb = mmcv.image.colorspace.rgb2bgr(seg_mask_rgb) # MMCV assumes BGR
    mmcv.imwrite(seg_mask_rgb, out_seg_file, auto_mkdir=True)

    img_info = dict(
        file_name=osp.basename(img_file),
        height=inst_img.shape[0],
        width=inst_img.shape[1],
        segments_info=segments_info,
        seg_file_name=osp.basename(seg_file))

    return img_info


def cvt_annotations(image_infos, out_json_name):
    out_json = dict()
    img_id = 0
    out_json['images'] = []
    out_json['categories'] = []
    out_json['annotations'] = []
    for image_info in image_infos:
        image_info['id'] = img_id
        segments_info = image_info.pop('segments_info')

        annotation = dict(
            image_id = img_id,
            file_name = image_info.pop('seg_file_name'),
            segments_info = segments_info
        )

        out_json['images'].append(image_info)
        out_json['annotations'].append(annotation)
        img_id += 1

    out_json['categories'] = [
        dict(id=0, name='static_obstacle', isthing=0),
        dict(id=1, name='water', isthing=0),
        dict(id=2, name='sky', isthing=0),
        dict(id=3, name='dynamic_obstacle', isthing=1)
    ]

    if len(out_json['annotations']) == 0:
        out_json.pop('annotations')

    mmcv.dump(out_json, out_json_name)
    return out_json


def parse_args():
    parser = argparse.ArgumentParser(
        description='Convert LaRS annotations to COCO format')
    parser.add_argument('dataset_path', help='LaRS data path')
    parser.add_argument('--img-dir', default='images', type=str)
    parser.add_argument('--seg-dir', default='masks', type=str)
    parser.add_argument('--inst-dir', default='instances', type=str)
    parser.add_argument('--segments-dir', default='coco_masks', type=str)
    parser.add_argument('-o', '--out-dir', help='Output directory')
    parser.add_argument('--nproc', default=1, type=int, help='Number of processes')
    args = parser.parse_args()
    return args


def main():
    args = parse_args()
    dataset_path = args.dataset_path
    out_dir = args.out_dir if args.out_dir else dataset_path
    mmcv.mkdir_or_exist(out_dir)

    img_dir = osp.join(dataset_path, args.img_dir)
    seg_dir = osp.join(dataset_path, args.seg_dir)
    inst_dir = osp.join(dataset_path, args.inst_dir)
    out_seg_dir = osp.join(dataset_path, args.segments_dir)

    set_name = {
        'list_train_panoptic.txt': 'panoptic_train.json',
        'list_val_panoptic.txt': 'panoptic_val.json',
        'list_test_panoptic.txt': 'panoptic_test.json',
    }

    for split_file, json_name in set_name.items():
        print(f'Converting {split_file} into {json_name}')
        split_file_path = osp.join(dataset_path, split_file)
        with mmcv.Timer(
                print_tmpl='It took {}s to convert LaRS annotations'):
            files = collect_files(split_file_path, img_dir, seg_dir, inst_dir, out_seg_dir)
            image_infos = collect_annotations(files, nproc=args.nproc)
            cvt_annotations(image_infos, osp.join(out_dir, json_name))


if __name__ == '__main__':
    main()
