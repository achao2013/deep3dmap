#main
work_dir="results/imgs2face_multipie"
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
joint_train=False
state_seq = ['sup', 'unsup']
image_size=256

# model settings
model = dict(
    type='imgs2mesh',
    model_cfgs=dict(
        model_name= "imgs2face",
        category= "face",
        use_sampling=True,
        image_size=image_size,
        texture_size=image_size,
        template_uvs_path="magicbox/face/diskmap/uvs.npy",
        template_normal_path= "magicbox/face/template_normal.obj",
        shape_param_path= "magicbox/face/Model_Shape.mat",
        exp_param_path= "magicbox/face/Model_Expression.mat",
        other_param_path="magicbox/face/sigma_exp.mat",
        tuplesize=3,
        model_shape=dict(type='Shape3dmmEncoder')
    ))

# dataset settings
#train_pipeline = []
test_pipeline = []
use_data_loaders=False
data = dict(
    samples_per_gpu=2,
    workers_per_gpu=2,
    train_sup=dict(
            type='FaceTexUVAsyncDataset',
            state=state_seq[0],
            tuplesize=3,
            image_size= image_size,
            texture_size=image_size,
            datadir='/home/achao/3d/database',
            imgdir='/media/achao/storage_5tb/data',
            datafile='data/multipie/multipie_uvtex2poseimgs.pkl',
            auxfile='data/multipie/multipie_imgpath2auxinfo.pkl'),
    train_unsup=dict(
            type='FaceImagesAsyncDataset',
            state=state_seq[1],
            tuplesize=3,
            image_size= image_size,
            texture_size=image_size,
            datadir='/home/achao/3d/database',
            imgdir='/media/achao/storage_5tb/data',
            datafile='data/multipie/multipie_idillumexp2poseimgpaths.pkl',
            auxfile='data/multipie/multipie_imgpath2auxinfo.pkl'),
    val=dict(pipeline=test_pipeline))

##runner settings

optimizer_config = dict(grad_clip=None)
lr_config = dict(
    policy='step',
    warmup='linear',
    warmup_iters=500,
    warmup_ratio=0.001,
    step=[8])
workflow = [('train', 1)]
runner = dict(
    type='StateMachineRunner', 
    runner_cfgs=dict(
        distributed=distributed,
        state_switch_mode='epoch_steps',
        state_switch_method='once_inorder',
        state_steps = dict(sup= 16,unsup= 32),
        state_seq = state_seq,
        optimizer = dict(type='Adam',
                    lr=0.0001, betas=(0.9, 0.999), weight_decay=5e-4),
        max_epochs=64))



