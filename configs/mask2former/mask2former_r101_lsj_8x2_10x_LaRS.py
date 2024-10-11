_base_ = 'mask2former_r50_lsj_8x2_1x_LaRS.py'

model = dict(
    backbone=dict(
        depth=101,
        init_cfg=dict(type='Pretrained',
                      checkpoint='torchvision://resnet101')))

max_iters = 50_000

lr_config = dict(step=[35_000])
runner = dict(max_iters=max_iters)
