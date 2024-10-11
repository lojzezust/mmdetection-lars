
interval = 500
checkpoint_config = dict(by_epoch=False, interval=interval, save_last=True, max_keep_ckpts=3)
# yapf:disable

wandb_logger = dict(
    type='WandbLoggerHook',
    by_epoch=False,
    with_step=False,
    init_kwargs=dict(project='mmdetection'))

log_config = dict(
    interval=50,
    hooks=[
        dict(type='TextLoggerHook', by_epoch=False),
        # dict(type='TensorboardLoggerHook', by_epoch=False),
        wandb_logger
    ])
# yapf:enable
custom_hooks = [dict(type='NumClassCheckHook')]

dist_params = dict(backend='nccl')
log_level = 'INFO'
load_from = None
resume_from = None
workflow = [('train', interval)]
evaluation = dict(interval=interval, metric=['PQ'])

# disable opencv multithreading to avoid system being overloaded
opencv_num_threads = 0
# set multi-process start method as `fork` to speed up the training
mp_start_method = 'fork'

# Default setting for scaling LR automatically
#   - `enable` means enable scaling LR automatically
#       or not by default.
#   - `base_batch_size` = (8 GPUs) x (2 samples per GPU).
auto_scale_lr = dict(enable=False, base_batch_size=16)