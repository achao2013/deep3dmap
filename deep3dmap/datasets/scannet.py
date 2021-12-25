import torch
import numpy as np
import os
import cv2
import pickle
from PIL import Image
from torch.utils.data import Dataset
from .builder import DATASETS
from .pipelines import Compose

@DATASETS.register_module()
class ScanNetDataset(Dataset):
    CLASSES=None
    def __init__(self, datapath, mode, pipeline, nviews, n_scales):
        super(ScanNetDataset, self).__init__()
        self.datapath = datapath
        self.mode = mode
        self.n_views = nviews
        self.pipeline = pipeline
        self.tsdf_file = 'all_tsdf_{}'.format(self.n_views)

        assert self.mode in ["train", "val", "test", "train_debug", "val_debug"]
        self.metas = self.build_list()
        if mode == 'test':
            self.source_path = 'scans_test'
        else:
            self.source_path = 'scans'

        self.n_scales = n_scales
        self.epoch = None
        self.tsdf_cashe = {}
        self.max_cashe = 100

        # processing pipeline
        self.pipeline = Compose(pipeline)
        self.flag = np.zeros(len(self), dtype=np.uint8)

    def set_epoch(self, epoch):
        self.epoch=epoch
        self.tsdf_cashe = {}

    def build_list(self):
        with open(os.path.join(self.datapath, self.tsdf_file, 'fragments_{}.pkl'.format(self.mode)), 'rb') as f:
            metas = pickle.load(f)
        return metas

    def __len__(self):
        return len(self.metas)

    def read_cam_file(self, filepath, vid):
        intrinsics = np.loadtxt(os.path.join(filepath, 'intrinsic', 'intrinsic_color.txt'), delimiter=' ')[:3, :3]
        intrinsics = intrinsics.astype(np.float32)
        extrinsics = np.loadtxt(os.path.join(filepath, 'pose', '{}.txt'.format(str(vid))))
        return intrinsics, extrinsics

    def read_img(self, filepath):
        img = Image.open(filepath)
        return img

    def read_depth(self, filepath):
        # Read depth image and camera pose
        depth_im = cv2.imread(filepath, -1).astype(
            np.float32)
        depth_im /= 1000.  # depth is saved in 16-bit PNG in millimeters
        depth_im[depth_im > 3.0] = 0
        return depth_im

    def read_scene_volumes(self, data_path, scene):
        if scene not in self.tsdf_cashe.keys():
            if len(self.tsdf_cashe) > self.max_cashe:
                self.tsdf_cashe = {}
            full_tsdf_list = []
            for l in range(self.n_scales + 1):
                # load full tsdf volume
                full_tsdf = np.load(os.path.join(data_path, scene, 'full_tsdf_layer{}.npz'.format(l)),
                                    allow_pickle=True)
                full_tsdf_list.append(full_tsdf.f.arr_0)
            self.tsdf_cashe[scene] = full_tsdf_list
        return self.tsdf_cashe[scene]

    def __getitem__(self, idx):
        meta = self.metas[idx]

        imgs = []
        depth = []
        extrinsics_list = []
        intrinsics_list = []

        tsdf_list = self.read_scene_volumes(os.path.join(self.datapath, self.tsdf_file), meta['scene'])

        for i, vid in enumerate(meta['image_ids']):
            # load images
            imgs.append(
                self.read_img(
                    os.path.join(self.datapath, self.source_path, meta['scene'], 'color', '{}.jpg'.format(vid))))

            depth.append(
                self.read_depth(
                    os.path.join(self.datapath, self.source_path, meta['scene'], 'depth', '{}.png'.format(vid)))
            )

            # load intrinsics and extrinsics
            intrinsics, extrinsics = self.read_cam_file(os.path.join(self.datapath, self.source_path, meta['scene']),
                                                        vid)

            intrinsics_list.append(intrinsics)
            extrinsics_list.append(extrinsics)

        intrinsics = np.stack(intrinsics_list)
        extrinsics = np.stack(extrinsics_list)

        items = {
            'imgs': imgs,
            'depth': depth,
            'intrinsics': intrinsics,
            'extrinsics': extrinsics,
            'tsdf_list_full': tsdf_list,
            'vol_origin': meta['vol_origin'],
            'scene': meta['scene'],
            'fragment': meta['scene'] + '_' + str(meta['fragment_id']),
            'epoch': [self.epoch],
        }
        print('input keys:',items.keys())
        print('fragment:',items['fragment'])
        if self.pipeline is not None:
            items = self.pipeline(items)
        return items



def collate_fn(list_data):
    cam_pose, depth_im, _ = list_data
    # Concatenate all lists
    return cam_pose, depth_im, _


class ScanNetSceneDataset(Dataset):
    """Pytorch Dataset for a single scene. getitem loads individual frames"""

    def __init__(self, n_imgs, scene, data_path, max_depth, id_list=None):
        """
        Args:
        """
        self.n_imgs = n_imgs
        self.scene = scene
        self.data_path = data_path
        self.max_depth = max_depth
        if id_list is None:
            self.id_list = [i for i in range(n_imgs)]
        else:
            self.id_list = id_list

    def __len__(self):
        return self.n_imgs

    def __getitem__(self, id):
        """
        Returns:
            dict of meta data and images for a single frame
        """
        id = self.id_list[id]
        cam_pose = np.loadtxt(os.path.join(self.data_path, self.scene, "pose", str(id) + ".txt"), delimiter=' ')

        # Read depth image and camera pose
        depth_im = cv2.imread(os.path.join(self.data_path, self.scene, "depth", str(id) + ".png"), -1).astype(
            np.float32)
        depth_im /= 1000.  # depth is saved in 16-bit PNG in millimeters
        depth_im[depth_im > self.max_depth] = 0

        # Read RGB image
        color_image = cv2.cvtColor(cv2.imread(os.path.join(self.data_path, self.scene, "color", str(id) + ".jpg")),
                                   cv2.COLOR_BGR2RGB)
        color_image = cv2.resize(color_image, (depth_im.shape[1], depth_im.shape[0]), interpolation=cv2.INTER_AREA)

        return cam_pose, depth_im, color_image
