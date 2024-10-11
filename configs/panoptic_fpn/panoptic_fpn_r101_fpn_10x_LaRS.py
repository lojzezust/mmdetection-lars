_base_ = 'panoptic_fpn_r101_fpn_1x_LaRS.py'

max_iters = 50_000

lr_config = dict(step=[40000, 47500])
runner = dict(max_iters=max_iters)
