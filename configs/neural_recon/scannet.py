#common
N_VIEWS=9
VOXEL_SIZE=0.04
EVAL_EPOCH=47
work_dir="results/neucon_scannet"
distributed=True
find_unused_parameters=True
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



#data
dataset_type = 'ScanNetDataset'
img_norm_cfg = dict(
    mean=[103.53, 116.28, 123.675], std=[1.0, 1.0, 1.0], to_rgb=False)
train_pipeline = [
    dict(type='SeqResizeImage968x1296', size=(640,480),imgs_key='imgs', intrinsics_key='intrinsics'),
    dict(type='SeqToTensor', imgslike_keys=['imgs','depth'], common_keys=['intrinsics','extrinsics'], 
            iter_keys=['tsdf_list_full']),#in sequence, we need to tensor first 
    dict(type='SeqRandomTransformSpace', voxel_dim=[96, 96, 96], voxel_size=VOXEL_SIZE, random_rotation=True, random_translation=True,
                 paddingXY=0.1, paddingZ=0.025, max_epoch=29, in_origin_key='vol_origin', in_epoch_key='epoch', in_tsdf_key='tsdf_list_full', 
                 in_extrinsics_key='extrinsics', in_intrinsics_key='intrinsics', in_imgs_key='imgs', in_depth_key='depth', 
                 out_origin_partial_key='vol_origin_partial', out_tsdf_key='tsdf_list', out_occ_key='occ_list'),
    dict(type='SeqIntrinsicsPoseToProjection', n_views=N_VIEWS, stride=4, in_intrinsics_key='intrinsics', in_extrinsics_key='extrinsics',
        out_world2camera_key='world_to_aligned_camera', out_matrix_key='proj_matrices'),
    dict(type='SeqNormalizeImages', **img_norm_cfg, keys=['imgs']),
    
    #dict(type='Collect', keys=['imgs','depth','intrinsics','extrinsics', 'tsdf_list_full', 'proj_matrices', 'vol_origin',
    #        'vol_origin_partial', 'world_to_aligned_camera','tsdf_list','occ_list'], meta_keys=['fragment','scene','epoch'])
    dict(type='Collect', keys=['imgs', 'proj_matrices', 'vol_origin',
            'vol_origin_partial', 'world_to_aligned_camera','tsdf_list','occ_list'], meta_keys=['fragment','scene','epoch'])
    #dict(type='')
]
test_pipeline = [
    dict(type='SeqResizeImage968x1296', size=(640,480),imgs_key='imgs', intrinsics_key='intrinsics'), 
    dict(type='SeqToTensor', imgslike_keys=['imgs','depth'], common_keys=['intrinsics','extrinsics'], 
            iter_keys=['tsdf_list_full']),#in sequence, we need to tensor first
    dict(type='SeqRandomTransformSpace', voxel_dim=[96, 96, 96], voxel_size=VOXEL_SIZE, random_rotation=False, random_translation=False,
                 paddingXY=0, paddingZ=0, max_epoch=29, in_origin_key='vol_origin', in_epoch_key='epoch', in_tsdf_key='tsdf_list_full', 
                 in_extrinsics_key='extrinsics', in_intrinsics_key='intrinsics', in_imgs_key='imgs', in_depth_key='depth', 
                 out_origin_partial_key='vol_origin_partial', out_tsdf_key='tsdf_list', out_occ_key='occ_list'),
    dict(type='SeqIntrinsicsPoseToProjection', n_views=N_VIEWS, stride=4, in_intrinsics_key='intrinsics', in_extrinsics_key='extrinsics',
        out_world2camera_key='world_to_aligned_camera', out_matrix_key='proj_matrices'),
    dict(type='SeqNormalizeImages', **img_norm_cfg, keys=['imgs']),
    dict(type='Collect', keys=['imgs','depth','intrinsics','extrinsics', 'tsdf_list_full', 'proj_matrices', 'vol_origin',
            'vol_origin_partial', 'world_to_aligned_camera','tsdf_list','occ_list'], meta_keys=['fragment','scene','epoch'])
]
use_data_loaders=True
data = dict(
    samples_per_gpu=1,
    workers_per_gpu=1,
    train=dict(
            type=dataset_type,
            datapath='/media/achao/Innov8/database/scannet-download',
            mode='train_debug',
            nviews=N_VIEWS, 
            n_scales=2,
            pipeline=train_pipeline),
    test=dict(
            type=dataset_type,
            datapath='/media/achao/Innov8/database/scannet-download',
            mode='test',
            nviews=N_VIEWS, 
            n_scales=2,
            epoch=EVAL_EPOCH,        
            pipeline=test_pipeline))

# model settings
model = dict(
    type='NeuralRecon',
    model_cfgs=dict(
        save_scene=True,
        save_scene_params=dict(
            rootdir=work_dir,
            savedir_postfix='save',
            dataset_name=dataset_type,
            voxel_size=VOXEL_SIZE,
            vis_incremental=False
        ),
        N_LAYER=3,
        N_VOX=[96, 96, 96],
        VOXEL_SIZE=VOXEL_SIZE,
        TRAIN_NUM_SAMPLE=[4096, 16384, 65536],
        TEST_NUM_SAMPLE=[4096, 16384, 65536],
        BACKBONE2D=dict(
            ARC='fpn-mnas-1'),
        FUSION=dict(
            FUSION_ON=True,
            #FUSION_ON=False,
            HIDDEN_DIM=64,
            AVERAGE=False,
            FULL=True),
            #FULL=False),
        LW=[1.0, 0.8, 0.64],
        THRESHOLDS=[0, 0, 0],
        POS_WEIGHT=1.5,
        SPARSEREG=dict(DROPOUT=False)
    ),
    test_cfg=dict(diy_load_checkpoint=True)
)


##runner settings

evaluation=dict(data_path="/media/achao/Innov8/database/scannet-download/scans_test",
                save_path=work_dir+'/scene_' + dataset_type+ '_save_fusion_eval_'+str(EVAL_EPOCH),
                gt_path="/media/achao/Innov8/database/scannet-download/scans_test",
                max_depth=10,num_workers=2,loader_num_workers=2,n_proc=2,n_gpu=2)
optimizer_config = dict(grad_clip=dict(max_norm=1, norm_type=2))
lr_config = dict(
    policy='step',
    gamma=0.5,
    step=[12,24,48])
workflow = [('train', 1)]
runner = dict(
    type='EpochBasedRunner', 
    runner_cfgs=dict(
        optimizer = dict(type='Adam',lr=0.001, betas=(0.9, 0.999), weight_decay=0.0),
        max_epochs=29))