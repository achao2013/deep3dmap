## GAN2Shape

### Data Preperation
```sh
scripts/gan2shape/download.sh
```

### Env Preperation
**compile StyleGAN2**
```sh
cd pnpmodules/stylegan2/stylegan2-pytorch/op
python setup.py install
```

### Run 
**Example2**: training on Celeba images:
```sh
sh scripts/gan2shape/run_celeba.sh configs/gan2shape/celeba.py GPU_NUMS
```

## NeuralRecon

### Data Preperation for ScanNet
Download and extract ScanNet by following the instructions provided at http://www.scan-net.org/.
<details>
  <summary>[Expected directory structure of ScanNet (click to expand)]</summary>
    
  You can obtain the train/val/test split information from [here](https://github.com/ScanNet/ScanNet/tree/master/Tasks/Benchmark).
  ```
  DATAROOT
  └───scannet
  │   └───scans
  │   |   └───scene0000_00
  │   |       └───color
  │   |       │   │   0.jpg
  │   |       │   │   1.jpg
  │   |       │   │   ...
  │   |       │   ...
  │   └───scans_test
  │   |   └───scene0707_00
  │   |       └───color
  │   |       │   │   0.jpg
  │   |       │   │   1.jpg
  │   |       │   │   ...
  │   |       │   ...
  |   └───scannetv2_test.txt
  |   └───scannetv2_train.txt
  |   └───scannetv2_val.txt
  ```
</details>

  Next run the data preparation script which parses the raw data format into the processed pickle format.
  This script also generates the ground truth TSDFs using TSDF Fusion.  
<details>
    <summary>[Data preparation script]</summary>

**Change data_path accordingly.**

```sh
sh scripts/neural_recon/gen_tsdf.sh
```
</details>

### Training on ScanNet

```sh
sh scripts/neural_recon/run_train_scannet.sh configs/neural_recon/scannet.py GPU_NUMS
```


## PRNet

### Data Preperation for 300WLP

```sh
python tools/data_gen/prnet.py -i /path/to/300WLP -o /path/to/300WLP-256
```

### Training on 300WLP

```sh
sh scripts/prnet/run_train_prnet.sh configs/prnet/prnet_300wlp.py GPU_NUMS
```


## img2mesh

### Data Preperation for Multipie

```sh
python tools/data_gen/multipie_get_lmk.py
python tools/data_gen/multipie_orgnizedata.py
```


### Train on Multipie

```sh
sh scripts/pt3d_demos/run_train_imgs2face.sh configs/pt3d_demos/imgs2face_multipie.py GPU_NUMS
```


## lerf

### data prepare

download [dataset](https://drive.google.com/drive/folders/1vh0mSl7v29yaGsxleadcj-LCZOE_WEWB)

### train
```sh
python tools/ns/train.py --method-name lerf --data /path/to/data_folder
```