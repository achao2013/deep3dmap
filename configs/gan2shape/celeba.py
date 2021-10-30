#main
work_dir="results/celeba"
distributed=True

# model settings
model = dict(
    type='Gan2Shape',
    model_cfgs=dict(
        model_name= "gan2shape_face",
        category= "face",
        distributed=distributed,
        share_weight= False,  # true: share weight in distributed training
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
        pretrain= "pre-model/gan2shape/celeba_pretrain.pt",
        view_mvn_path= "pre-model/gan2shape/view_light/celeba_view_mvn.pth",
        light_mvn_path= "pre-model/gan2shape/view_light/celeba_light_mvn.pth",
        rand_light= [-1,1,-0.2,0.8,-0.1,0.6,-0.6],
        ## GAN
        channel_multiplier= 1,
        gan_size= 128,
        gan_ckpt= "checkpoints/stylegan2/stylegan2-celeba-config-e.pt",
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
data = dict(
    samples_per_gpu=16,
    workers_per_gpu=4,
    train=dict(
            type=dataset_type,
            distributed=distributed,
            image_size= 128,
            load_gt_depth= False,
            img_list_path= "data/celeba/list_200-399.txt",
            img_root= "data/celeba",
            latent_root= "data/celeba/latents",
            pipeline=train_pipeline),
    val=dict(pipeline=test_pipeline))

##runner settings
workflow = [('train', 1)]
runner = dict(type='Gan2ShapeRunner', 
    distributed=distributed,
    checkpoint_dir = 'results/celeba/checkpoints',
    save_checkpoint_freq= 500,
    keep_num_checkpoint= 1,
    use_logger= True,
    log_freq= 100,
    joint_train= False,  # True: joint train on multiple images
    independent= True,  # True: each process has a different input image
    reset_weight= True,  # True: reset model weights after each epoch
    save_results= True,
    num_stage= 4,
    flip1_cfg= [False, False, False, False],
    flip3_cfg= [True, False, False, False],
    stage_len_dict = dict(step1= 600,step2= 600,step3= 400),
    stage_len_dict2 = dict(step1= 200,step2= 500,step3= 300),
    max_epochs=300)

optimizer = dict(
    type='Adam',
    lr=0.0001, betas=(0.9, 0.999), weight_decay=5e-4)

