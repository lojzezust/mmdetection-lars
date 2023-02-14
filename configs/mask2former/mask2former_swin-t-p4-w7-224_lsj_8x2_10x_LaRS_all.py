_base_ = ['./mask2former_swin-t-p4-w7-224_lsj_8x2_10x_LaRS.py']

data_root = 'data/LaRS/v0.8/'
data = dict(
    train=dict(
        ann_file=data_root + 'all/mmdet_annotations.json',
        img_prefix=data_root + 'all/images/',
        seg_prefix=data_root + 'all/panoptic_masks/'),
    test=dict(
        ann_file=data_root + 'all/mmdet_annotations.json',
        img_prefix=data_root + 'all/images/',
        seg_prefix=data_root + 'all/panoptic_masks/'),
)
