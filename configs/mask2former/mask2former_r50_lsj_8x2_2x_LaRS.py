_base_ = 'mask2former_r50_lsj_8x2_1x_LaRS.py'

max_iters = 10_000

lr_config = dict(step=[7000])
runner = dict(max_iters=max_iters)
