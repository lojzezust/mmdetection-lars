
# optimizer
optimizer = dict(type='SGD', lr=0.02, momentum=0.9, weight_decay=0.0001)
optimizer_config = dict(grad_clip=None)

max_iters = 5000 # ~ 50 epoh

# learning policy
lr_config = dict(
    policy='step',
    gamma=0.1,
    by_epoch=False,
    step=[4000, 4700],
    warmup='linear',
    warmup_by_epoch=False,
    warmup_ratio=0.001,
    warmup_iters=500)

runner = dict(type='IterBasedRunner', max_iters=max_iters)
