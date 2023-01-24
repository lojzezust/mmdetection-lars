# dataset settings
dataset_type = 'CocoPanopticDataset'
classes = ('boat', 'row_boat', 'paddle_board', 'buoy', 'swimmer', 'animal', 'float', 'other', 'static_obstacle', 'water', 'sky')
data_root = 'data/LaRS/v0.4/'
img_norm_cfg = dict(
    mean=[123.675, 116.28, 103.53], std=[58.395, 57.12, 57.375], to_rgb=True)
train_pipeline = [
    dict(type='LoadImageFromFile'),
    dict(
        type='LoadPanopticAnnotations',
        with_bbox=True,
        with_mask=True,
        with_seg=True),
    dict(type='Resize', img_scale=(1333, 800), keep_ratio=True),
    dict(type='RandomFlip', flip_ratio=0.5),
    dict(type='Normalize', **img_norm_cfg),
    dict(type='Pad', size_divisor=32),
    dict(type='SegRescale', scale_factor=1 / 4),
    dict(type='DefaultFormatBundle'),
    dict(
        type='Collect',
        keys=['img', 'gt_bboxes', 'gt_labels', 'gt_masks', 'gt_semantic_seg']),
]
test_pipeline = [
    dict(type='LoadImageFromFile'),
    dict(
        type='MultiScaleFlipAug',
        img_scale=(1333, 800),
        flip=False,
        transforms=[
            dict(type='Resize', keep_ratio=True),
            dict(type='RandomFlip'),
            dict(type='Normalize', **img_norm_cfg),
            dict(type='Pad', size_divisor=32),
            dict(type='ImageToTensor', keys=['img']),
            dict(type='Collect', keys=['img']),
        ])
]
data = dict(
    samples_per_gpu=2,
    workers_per_gpu=2,
    train=dict(
        type=dataset_type,
        classes=classes,
        ann_file=data_root + 'train/mmdet_annotations.json',
        img_prefix=data_root + 'train/images/',
        seg_prefix=data_root + 'train/panoptic_masks/',
        pipeline=train_pipeline),
    val=dict(
        type=dataset_type,
        classes=classes,
        ann_file=data_root + 'val/mmdet_annotations.json',
        img_prefix=data_root + 'val/images/',
        seg_prefix=data_root + 'val/panoptic_masks/',
        pipeline=test_pipeline),
    test=dict(
        type=dataset_type,
        classes=classes,
        ann_file=data_root + 'test/mmdet_annotations.json',
        img_prefix=data_root + 'test/images/',
        seg_prefix=data_root + 'test/panoptic_masks/',
        pipeline=test_pipeline))
