_base_ = 'panoptic_fpn_r101_fpn_1x_LaRS.py'

max_iters = 10_000

lr_config = dict(step=[8000, 9500])
runner = dict(max_iters=max_iters)
