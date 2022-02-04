#main
work_dir="results/gan2shape_celeba"
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

# model settings
image_size=128
gn_base = 8 if image_size >= 128 else 16
nf = max(4096 // image_size, 16)
model = dict(
    type='Gan2Shape',
    model_cfgs=dict(
        model_name= "gan2shape_face",
        category= "face",
        distributed=distributed,
        checkpoint_dir=work_dir,
        share_weight= False,  # true: share weight in distributed training
        inner_parallel=True,
        image_size=image_size,
        depth_head=dict(type='EDDeconv', cin=3, cout=1, size=image_size, nf=nf, gn_base=gn_base, zdim=256, activation=None),
        albedo_head=dict(type='EDDeconv', cin=3, cout=3, size=image_size, nf=nf, gn_base=gn_base, zdim=256),
        view_head=dict(type='Encoder', cin=3, cout=6, size=image_size, nf=nf),
        light_head=dict(type='Encoder', cin=3, cout=4, size=image_size, nf=nf),
        encoder_head=dict(type='ResEncoder', cin=3, cout=512, size=image_size, nf=32, activation=None),
        flip1_cfg= [False, False, False, False],
        flip3_cfg= [True, False, False, False],
        relative_enc= True,  # true: use relative latent offset
        use_mask= False,
        add_mean_L= True,
        add_mean_V= True,
        min_depth= 0.9,
        max_depth= 1.1,
        xyz_rotation_range= 60,  # (-r,r) in degrees
        xy_translation_range= 0.1,  # (-t,t) in 3D
        z_translation_range= 0,  # (-t,t) in 3D
        view_scale= 1.2,  # enlarge viewpoint variation
        collect_iters= 100,
        batchsize= 16,
        lam_perc= 0.5,
        lam_smooth= 0.01,
        lam_regular= 0.01,
        pretrain= "pre-model/gan2shape/celeba_pretrain.pth",
        view_mvn_path= "pre-model/gan2shape/view_light/celeba_view_mvn.pth",
        light_mvn_path= "pre-model/gan2shape/view_light/celeba_light_mvn.pth",
        #category in ['face', 'synface']
        parsing_model_path="pre-model/gan2shape/parsing/bisenet.pth",
        #category == 'church'
        #parsing_model_path="pre-model/gan2shape/parsing/pspnet_ade20k.pth",
        #parsing_model_path="pre-model/gan2shape/parsing/pspnet_voc.pth",
        rand_light= [-1,1,-0.2,0.8,-0.1,0.6,-0.6],
        ## GAN
        channel_multiplier= 1,
        gan_size= 128,
        gan_ckpt= "pre-model/gan2shape/stylegan2/stylegan2-celeba-config-e.pt",
        F1_d= 2,  # number of mapping network layers used to regularize the latent offset

        ## renderer
        rot_center_depth= 1.0,
        fov= 10,  # in degrees
        tex_cube_size= 2
    ))

# dataset settings
dataset_type = 'CelebaDataset'
train_pipeline = []
test_pipeline = []
use_data_loaders=False
data = dict(
    samples_per_gpu=16,
    workers_per_gpu=4,
    train=dict(
            type=dataset_type,
            distributed=distributed,
            image_size= 128,
            load_gt_depth= False,
            joint_train=joint_train,
            img_list_path= "data/celeba/list_200-399.txt",
            img_root= "data/celeba",
            latent_root= "data/celeba/latents",
            pipeline=train_pipeline),
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
    type='Gan2ShapeRunner', 
    runner_cfgs=dict(
        distributed=distributed,
        checkpoint_dir = 'results/celeba/checkpoints',
        save_checkpoint_freq= 500,
        keep_num_checkpoint= 1,
        use_logger= True,
        log_freq= 100,
        joint_train= joint_train,  # True: joint train on multiple images
        independent= True,  # True: each process has a different input image
        reset_weight= True,  # True: reset model weights after each epoch
        save_results= True,
        num_stage= 4,
        stage_len_dict = dict(step1= 600,step2= 600,step3= 400),
        stage_len_dict2 = dict(step1= 200,step2= 500,step3= 300),
        optimizer = dict(type='Adam',
                    lr=0.0001, betas=(0.9, 0.999), weight_decay=5e-4),
        max_epochs=300))



