#main
from sched import scheduler


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
img_wh=(400,400)

# model settings
model = dict(
    type='GanNerf',
    need_info_from_datasets=True,
    model_cfgs=dict(
        model_name= "GanNerf",
        distributed=True,
        chunk=1024*32,
        #path sampler
        min_scale=0.0,
        max_scale=1.0,
        scale_anneal=0.0002,
        patch_size=16,
        random_scale=True,
        #data
        img_wh=img_wh,
        azim_range= [ 0., 360. ],  # the range of azimuth
        elev_range= [ 0., 90. ],   # the range of elevation
        radius= [ 4.0, 4.0 ],  # the range of radius
        near= 2.0,
        far= 6.0,
        white_back= True,
        ndc= False,
        look_at_origin= True,
        pose_mode= '3d'
        #inversion network
        inv_size=64,
        # discriminator
        ndf=64,
        conditional=True,
        policy=['color', 'cutout'],
        #nerf
        xyz_freq=10,
        dir_freq=4,
        N_samples=64,
        N_importance=64,
        fc_depth=8,
        fc_dim=360,
        decrease_noise=True)
    )

# dataset settings
train_pipeline = [
    dict(type='Resize', img_scale=img_wh, keys=['img']),
    dict(type='ToTensor',keys='img'),
    dict(type='BlendAToRGB'),
    dict(type='NormalizeForGAN')
]
test_pipeline = [
    dict(type='Resize', img_scale=img_wh, keys=['img']),
    dict(type='ToTensor',keys='img'),
    dict(type='NormalizeForGAN')
]
use_data_loaders=True
data = dict(
    samples_per_gpu=2,
    workers_per_gpu=2,
    train=dict(
            type='Blender',
            name='blender_train',
            data_dir='',
            img_wh=img_wh,
            split='train',
            sort_key=lambda x: int(x.split('/')[-1][x.split('/')[-1].index('_') + 1:  x.split('/')[-1].index('.')]),
            pipeline=train_pipeline
            ),
    val=dict(
        type='Blender',
        name='blender_val',
        data_dir='',
        img_wh=img_wh,
        split='val',
        sort_key=lambda x: int(x.split('/')[-1][x.split('/')[-1].index('_') + 1:  x.split('/')[-1].index('.')]),
        pipeline=test_pipeline))

##runner settings

optimizer_config = dict(type='MultiOptimizerHook',grad_clip=None)
lr_config = dict(
    policy='MultiStep',
    lr_schedulers=dict(
        generator=dict(gamma=0.5,step=[8000,16000,24000,32000]),
        discriminator=dict(gamma=0.5,step=[8000,16000,24000,32000]),
        train_pose_params=dict(gamma=0.5,step=[4000,8000,12000,16000,20000,24000,28000,32000]),
        inv_net=dict(gamma=0.5,step=[4000,8000,12000,16000,20000,24000,28000,32000]),
        val_pose_params=dict(gamma=0.5,step=[4000,8000,12000,16000,20000,24000,28000,32000])
    ))
workflow = [('train', 1),('val',1)]
runner = dict(
    type='StateMachineRunner', 
    runner_cfgs=dict(
        distributed=distributed,
        state_switch_mode='iter_steps',
        state_switch_method='once_inorder',
        state_steps = dict(A= 12000,ABAB=20000),
        state_seq = state_seq,
        optimizer = dict(
            generator=dict(type='RMSprop',lr=0.0005,alpha=0.99, eps=1e-8),
            discriminator=dict(type='RMSprop',lr=0.0001,alpha=0.99, eps=1e-8),
            train_pose_params=dict(type='Adam',lr=0.005, betas=(0.9, 0.99)),
            inv_net=dict(type='Adam',lr=0.0001, betas=(0.9, 0.99)),
            val_pose_params=dict(type='Adam',lr=0.005, betas=(0.9, 0.99))
        ),
        max_epochs=64))



