#main
work_dir="results/gnerf_belender"
distributed=True
dist_params = dict(backend='nccl')
checkpoint_config = dict(interval=1)
log_level = 'INFO'
log_config = dict(
    interval=50,
    hooks=[
        dict(type='TextLoggerHook'),
        # dict(type='TensorboardLoggerHook')
    ])
resume_from = None
load_from = None

state_seq = ['A', 'ABAB', 'B']
image_size=256

# model settings
model = dict(
    type='GanNerf',
    need_info_from_datasets=True,
    model_cfgs=dict(
        model_name= "GanNerf",
        distributed=True,
        azim_range= [ 0., 360. ],  # the range of azimuth
        elev_range= [ 0., 90. ],   # the range of elevation
        radius= [ 4.0, 4.0 ],  # the range of radius
        near= 2.0,
        far= 6.0,
        white_back= True,
        ndc= False,
        look_at_origin= True,
        pose_mode= '3d')
    )

# dataset settings
train_pipeline = [
    dict(type='Resize', img_scale=(400,400), keys=['img']),
    dict(type='ToTensor',keys='img'),
    dict(type='BlendAToRGB'),
    dict(type='NormalizeForGAN')
]
test_pipeline = [
    dict(type='Resize', img_scale=(400,400), keys=['img']),
    dict(type='ToTensor',keys='img'),
    dict(type='NormalizeForGAN')
]
use_data_loaders=False
data = dict(
    samples_per_gpu=2,
    workers_per_gpu=2,
    train=dict(
            type='Blender',
            name='blender',
            state=state_seq[0],
            sort_key=lambda x: int(x.split('/')[-1][x.split('/')[-1].index('_') + 1:  x.split('/')[-1].index('.')]),
            pipeline=train_pipeline
            ),
    val=dict(
        type='DTU',
        name='dtu',
        sort_key=lambda x: int(x.split('/')[-1][5:8]),
        pipeline=test_pipeline))

##runner settings

optimizer_config = dict(type='MultiOptimizerHook',grad_clip=None)
lr_config = dict(
    policy='step',
    warmup='linear',
    warmup_iters=500,
    warmup_ratio=0.001,
    step=[8])
workflow = [('train', 1),('val',1)]
runner = dict(
    type='StateMachineRunner', 
    runner_cfgs=dict(
        distributed=distributed,
        state_switch_mode='iter_steps',
        state_switch_method='once_inorder',
        state_steps = dict(A= 12000,ABAB=20000),
        state_seq = state_seq,
        optimizer = [dict(type='Adam',
                    lr=0.0001, betas=(0.9, 0.999), weight_decay=5e-4),
                    ],
        max_epochs=64))



