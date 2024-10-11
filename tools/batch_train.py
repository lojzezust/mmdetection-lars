import subprocess

CONFIGS=[
    ('panfpn50_lars', "configs/panoptic_fpn/panoptic_fpn_r50_fpn_10x_LaRS.py"),
    ('panfpn101_lars', "configs/panoptic_fpn/panoptic_fpn_r101_fpn_10x_LaRS.py"),
    ('m2f_r50_lars', "configs/mask2former/mask2former_r50_lsj_8x2_10x_LaRS.py"),
    ('m2f_r101_lars', "configs/mask2former/mask2former_r101_lsj_8x2_10x_LaRS.py"),
    ('m2f_swint_lars', "configs/mask2former/mask2former_swin-t-p4-w7-224_lsj_8x2_10x_LaRS.py"),
    ('m2f_swinb_lars', "configs/mask2former/mask2former_swin-b-p4-w12-384-in21k_lsj_8x2_20x_LaRS.py")
]

for task_name, cfg in CONFIGS:
    subprocess.run(['tools/slurm_train_my.sh', task_name, cfg])
