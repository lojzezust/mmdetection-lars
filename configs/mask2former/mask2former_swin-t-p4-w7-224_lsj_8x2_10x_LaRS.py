_base_ = ['./mask2former_swin-t-p4-w7-224_lsj_8x2_1x_LaRS.py']

# 10x stuff
max_iters = 50_000

lr_config = dict(step=[35_000])
runner = dict(max_iters=max_iters)
