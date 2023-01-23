_base_ = ['./mask2former_swin-b-p4-w12-384-in21k_lsj_8x2_10x_LaRS.py']

# 20x stuff
max_iters = 100_000

lr_config = dict(step=[70_000])
runner = dict(max_iters=max_iters)
