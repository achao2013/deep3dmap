import os
import glob
import json
import torch
import numpy as np
from PIL import Image
from torch.utils.data import Dataset
from .builder import DATASETS

class DTU(Dataset):
    def __init__(self, name, split, data_dir, img_wh, pipeline, sort_key=None):
        super(DTU, self).__init__()
        self.name=name
        self.split = split
        self.data_dir = data_dir
        self.img_wh = img_wh
        self.pipeline = pipeline
        self.sort_key = sort_key

        self.filenames = self.get_filenames(self.data_dir)
        assert len(self.filenames) > 0, 'File dir is empty'
        self.img_wh_original = Image.open(self.filenames[0]).size
        assert self.img_wh_original[1] * self.img_wh[0] == self.img_wh_original[0] * self.img_wh[1], \
            f'You must set @img_wh to have the same aspect ratio as ' \
            f'({self.img_wh_original[0]}, {self.img_wh_original[1]}) !'

        self.imgs = self.load_imgs(self.filenames)
        self.intrinsics, self.poses = self.get_camera_params()

    def __len__(self):
        return len(self.filenames)

    def __getitem__(self, idx):
        img = self.imgs[idx]
        if self.pipeline is not None:
            img = self.pipeline(img)
        return img, idx

    def load_imgs(self, filenames):
        imgs = []
        for p in filenames:
            img = Image.open(p)

            
            imgs.append(img)

        return imgs

    def get_filenames(self, data_dir):
        filenames = glob.glob(f'{data_dir}/*_3_*.png')  # choose images with a same light condition

        if self.sort_key is not None:
            filenames.sort(key=self.sort_key)
        else:
            filenames.sort()

        # choose every 8 images as evaluation images, the rest as training images
        val_indices = list(np.arange(7, len(filenames), 8))
        if self.split == 'train':
            filenames = [filenames[x] for x in np.arange(0, len(filenames)) if x not in val_indices]
        elif self.split == 'val':
            filenames = [filenames[idx] for idx in val_indices]

        return filenames

    def get_camera_params(self):
        prefix = '/'.join(self.data_dir.split('/')[:-2] + ['Cameras', 'train'])
        id_list = [os.path.join(prefix, str(int(name.split('/')[-1][5:8]) - 1).zfill(8) + '_cam.txt') for name in
                   self.filenames]

        intrinsics, poses = [], []
        for id in id_list:
            with open(id) as f:
                text = f.read().splitlines()

                pose_text = text[text.index('extrinsic') + 1:text.index('extrinsic') + 5]
                pose_text = torch.tensor([[float(b) for b in a.strip().split(' ')] for a in pose_text])
                pose_text = torch.inverse(pose_text)

                intrinsic_text = text[text.index('intrinsic') + 1:text.index('intrinsic') + 4]
                intrinsic_text = torch.tensor([[float(b) for b in a.strip().split(' ')] for a in intrinsic_text])
                intrinsic_text[:2, :] *= 4.0  # rescale with image size

                poses.append(pose_text[None, :3, :4])
                intrinsics.append(intrinsic_text[None])

        poses = torch.cat(poses)  # [N, 3, 4]
        intrinsics = torch.cat(intrinsics, 0)

        intrinsics = intrinsics.mean(dim=0)  # assume intrinsics of all cameras are the same
        poses[:, :, 3] /= 200.0

        scale = torch.tensor([self.img_wh[0] / self.img_wh_original[0], self.img_wh[1] / self.img_wh_original[1]])
        intrinsics[:2] *= scale[:, None]

        return intrinsics, poses
