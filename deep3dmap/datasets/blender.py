import os
import glob
import json
import torch
import numpy as np
from PIL import Image
from torch.utils.data import Dataset
from .builder import DATASETS

@DATASETS.register_module()
class Blender(Dataset):
    def __init__(self, name, split, data_dir, img_wh, pipeline, sort_key=None):
        super(Blender, self).__init__()
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

    def get_filenames(self, root):
        filenames = glob.glob(f'{root}/{self.split}/*.png')

        if self.sort_key is not None:
            filenames.sort(key=self.sort_key)
        else:
            filenames.sort()

        if self.split == 'val':  # only validate 8 images
            filenames = filenames[:8]

        return filenames

    def get_camera_params(self):
        file_path = os.path.join(self.data_dir, f'transforms_{self.split}.json')

        with open(file_path, 'r') as f:
            meta = json.load(f)

        poses = []
        for frame in meta['frames']:
            pose = torch.tensor(frame['transform_matrix'])[:3, :4]
            poses.append(pose[None])

        poses = torch.cat(poses)  # [N, 3, 4]s

        cx, cy = [x // 2 for x in self.img_wh_original]
        focal = 0.5 * 800 / np.tan(0.5 * meta['camera_angle_x'])  # original focal length

        intrinsics = torch.tensor([
            [focal, 0, cx], [0, focal, cy], [0, 0, 1.]
        ])

        scale = torch.tensor([self.img_wh[0] / self.img_wh_original[0], self.img_wh[1] / self.img_wh_original[1]])
        intrinsics[:2] *= scale[:, None]

        return intrinsics, poses