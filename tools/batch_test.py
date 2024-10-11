import subprocess
import os.path as osp

WORK_DIRS = 'work_dirs'
PRED_DIRS = 'output/predictions_v0.8'

CONFIGS=[
    ('panfpn50_lars_test', "configs/panoptic_fpn/panoptic_fpn_r50_fpn_10x_LaRS.py"),
    ('panfpn101_lars_test', "configs/panoptic_fpn/panoptic_fpn_r101_fpn_10x_LaRS.py"),
    ('m2f_r50_lars_test', "configs/mask2former/mask2former_r50_lsj_8x2_10x_LaRS.py"),
    ('m2f_r101_lars_test', "configs/mask2former/mask2former_r101_lsj_8x2_10x_LaRS.py"),
    ('m2f_swint_lars_test', "configs/mask2former/mask2former_swin-t-p4-w7-224_lsj_8x2_10x_LaRS.py"),
    ('m2f_swinb_lars_test', "configs/mask2former/mask2former_swin-b-p4-w12-384-in21k_lsj_8x2_20x_LaRS.py")
]

for task_name, cfg in CONFIGS:
    run_name = osp.splitext(osp.basename(cfg))[0]
    checkpoint_pth = osp.join(WORK_DIRS, run_name, 'latest.pth')
    work_dir = osp.join(WORK_DIRS, run_name)
    pred_dir = osp.join(PRED_DIRS, run_name)
    subprocess.run(['tools/slurm_test_my.sh', task_name, cfg, checkpoint_pth, '--work-dir', work_dir,
                    '--eval',  'PQ', '--show-dir', pred_dir])

