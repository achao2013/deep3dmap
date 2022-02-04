#common
work_dir="results/prnet"
distributed=True
dist_params = dict(backend='nccl')
checkpoint_config = dict(interval=1)
log_level = 'INFO'
log_config = dict(
    interval=10,
    hooks=[
        dict(type='TextLoggerHook'),
        # dict(type='TensorboardLoggerHook')
    ])
resume_from = None
load_from = None



#data
img_norm_cfg = dict(
    mean=[103.53, 116.28, 123.675], std=[1.0, 1.0, 1.0], to_rgb=False)
npy_norm_cfg = dict(
    mean=[0, 0, 0], std=[255.0, 255.0, 255.0], to_rgb=False)
train_pipeline = [
    #use prefix 'in' or 'out' to distinguish input and output name; other prefix will be specific
    # if without in/out, input name==out name;
    # key+s means multiple keys
    dict(type='LoadImageFromFile', out_key='faceimg', to_float32=True),
    dict(type='LoadArrayUsingNp', out_key='gt_uvimg', to_float32=True),
    dict(type='Normalize', **img_norm_cfg, keys=['faceimg']),
    dict(type='Normalize', **npy_norm_cfg, keys=['gt_uvimg']),
    dict(type='FaceFormatBundle',imglike_keys=['faceimg', 'gt_uvimg']),
    dict(type='Collect', keys=['faceimg', 'gt_uvimg'])
    #dict(type='')
]
val_pipeline = [
    dict(type='LoadImageFromFile', out_key='img', to_float32=True),
    dict(type='LoadMatDictUsingSio', out_key='gt_md'),
    dict(type='GetKeysFromDict', in_dict_key='gt_md',out_keys=['pt3d_68']),
    #proj2d is the projected 2d points from 3d points
    dict(type='FaceLandmarkCrop', in_lm_key='pt3d_68', in_img_key='img', 
            out_lm_key='gt_kpt_proj2d', out_img_key='faceimg', out_trans_key='tform_mat'),    
    dict(type='Normalize', **img_norm_cfg, keys=['faceimg']),
    dict(type='FaceFormatBundle',imglike_keys=['faceimg'], 
            common_keys=['tfrom_mat','gt_kpt_proj2d']),
    dict(type='Collect', keys=['faceimg', 'tform_mat', 'gt_kpt_proj2d'])
]

use_data_loaders=True
data = dict(
    samples_per_gpu=32,
    workers_per_gpu=4,
    train=dict(
            type='ThreeHundredWLPDataset',
            datapath='/home/achao/data/300wlp-aflw2000/300W_LP_train.txt',
            #datapath='/home/achao/data/300wlp-aflw2000/debug_300wlp.txt',
            img_prefix='/home/achao/data/300wlp-aflw2000/generate_train_data/Data/300W_LP_256',
            pipeline=train_pipeline),
    val=dict(
            type='AFLW2000Dataset',
            datapath='/home/achao/data/300wlp-aflw2000/AFLW2000_test.txt',
            img_prefix='/home/achao/data/300wlp-aflw2000/generate_train_data/Data/AFLW2000',     
            pipeline=val_pipeline))

# model settings
model = dict(
    type='faceimg2uv',
    model_cfgs=dict(
        backbone=dict(type='resfcn256_std'),
        weightmaskfile='magicbox/face/uv_weight_mask.png',
        facemaskfile='magicbox/face/uv_face_mask.png',
        uv_kpt_ind_file='magicbox/face/uv_kpt_ind.txt'
    ),
    test_cfg=None
)


##runner settings
evaluation=dict(interval=1, save_best='nme', metric='nme')
optimizer_config = dict(grad_clip=None)
lr_config = None
workflow = [('train', 1)]
runner = dict(
    type='EpochBasedRunner', 
    runner_cfgs=dict(
        optimizer = dict(type='Adam',lr=0.001, betas=(0.9, 0.999), weight_decay=0.0001),
        max_epochs=32))