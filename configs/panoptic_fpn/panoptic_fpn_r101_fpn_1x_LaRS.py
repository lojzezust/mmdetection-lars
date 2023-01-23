_base_ = 'panoptic_fpn_r50_fpn_1x_LaRS.py'
model = dict(
    backbone=dict(
        depth=101,
        init_cfg=dict(type='Pretrained',
                      checkpoint='torchvision://resnet101')))
