# Copyright (c) achao2013. All rights reserved.
import torch
import numpy as np
import os
import cv2
import pickle
from PIL import Image
from torch.utils.data import Dataset
from .builder import DATASETS
from .pipelines import Compose

import sys

sys.path.append('.')
import argparse
import json
import os

import numpy as np
from deep3dmap.core.renderer.rerender_pr import PyRenderer
import trimesh

from deep3dmap.core.evaluation.depth_eval import eval_depth
from deep3dmap.core.evaluation.mesh_eval import eval_fscore
from deep3dmap.core.evaluation.metrics_utils import parse_metrics_neucon
import open3d as o3d
import ray
from ray.exceptions import GetTimeoutError

torch.multiprocessing.set_sharing_strategy('file_system')








@DATASETS.register_module()
class ScanNetDataset(Dataset):
    CLASSES=None
    def __init__(self, datapath, mode, pipeline, nviews, n_scales, epoch=0, test_mode=False):
        super(ScanNetDataset, self).__init__()
        self.datapath = datapath
        self.mode = mode
        self.test_mode = test_mode
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
        self.epoch = epoch
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

        if self.pipeline is not None:
            items = self.pipeline(items)
        return items

    def evaluate(self, outputs, metric, data_path, save_path, gt_path, max_depth, 
                    num_workers, loader_num_workers, n_proc, n_gpu, **kwargs):
        def process(scene, total_scenes_index, total_scenes_count, 
                    data_path, save_path, gt_path, max_depth, loader_num_workers):
            
            width, height = 640, 480

            test_framid = os.listdir(os.path.join(data_path, scene, 'color'))
            n_imgs = len(test_framid)
            intrinsic_dir = os.path.join(data_path, scene, 'intrinsic', 'intrinsic_depth.txt')
            cam_intr = np.loadtxt(intrinsic_dir, delimiter=' ')[:3, :3]
            dataset = ScanNetSceneDataset(n_imgs, scene, data_path, max_depth)

            dataloader = torch.utils.data.DataLoader(dataset, batch_size=None, collate_fn=collate_fn,
                                                    batch_sampler=None, num_workers=loader_num_workers)

            voxel_size = 4

            # re-fuse to remove hole filling since filled holes are penalized in
            # mesh metrics
            # tsdf_fusion = TSDFFusion(vol_dim, float(voxel_size)/100, origin, color=False)

            #volume = o3d.pipelines.integration.ScalableTSDFVolume(
            volume = o3d.integration.ScalableTSDFVolume(
                voxel_length=float(voxel_size) / 100,
                sdf_trunc=3 * float(voxel_size) / 100,
                color_type=o3d.integration.TSDFVolumeColorType.RGB8)
            #    color_type=o3d.pipelines.integration.TSDFVolumeColorType.RGB8)

            mesh_file = os.path.join(save_path, '%s.ply' % scene.replace('/', '-'))
            try:
                mesh = trimesh.load(mesh_file, process=False)
            except:
                return scene, None

            # mesh renderer
            renderer = PyRenderer()
            mesh_opengl = renderer.mesh_opengl(mesh)

            for i, (cam_pose, depth_trgt, _) in enumerate(dataloader):
                print(total_scenes_index, total_scenes_count, scene, i, len(dataloader))
                if cam_pose[0][0] == np.inf or cam_pose[0][0] == -np.inf or cam_pose[0][0] == np.nan:
                    continue

                _, depth_pred = renderer(height, width, cam_intr, cam_pose, mesh_opengl)

                temp = eval_depth(depth_pred, depth_trgt)
                if i == 0:
                    metrics_depth = temp
                else:
                    metrics_depth = {key: value + temp[key]
                                    for key, value in metrics_depth.items()}

                # placeholder
                color_im = np.repeat(depth_pred[:, :, np.newaxis] * 255, 3, axis=2).astype(np.uint8)
                depth_pred = o3d.geometry.Image(depth_pred)
                color_im = o3d.geometry.Image(color_im)
                rgbd = o3d.geometry.RGBDImage.create_from_color_and_depth(color_im, depth_pred, depth_scale=1.0,
                                                                        depth_trunc=5.0,
                                                                        convert_rgb_to_intensity=False)

                volume.integrate(
                    rgbd,
                    o3d.camera.PinholeCameraIntrinsic(width=width, height=height, fx=cam_intr[0, 0], fy=cam_intr[1, 1],
                                                    cx=cam_intr[0, 2],
                                                    cy=cam_intr[1, 2]), np.linalg.inv(cam_pose))

            metrics_depth = {key: value / len(dataloader)
                            for key, value in metrics_depth.items()}

            # save trimed mesh
            file_mesh_trim = os.path.join(save_path, '%s_trim_single.ply' % scene.replace('/', '-'))
            o3d.io.write_triangle_mesh(file_mesh_trim, volume.extract_triangle_mesh())

            # eval trimed mesh
            file_mesh_trgt = os.path.join(gt_path, scene, scene + '_vh_clean_2.ply')
            metrics_mesh = eval_fscore(file_mesh_trim, file_mesh_trgt)

            metrics = {**metrics_depth, **metrics_mesh}

            rslt_file = os.path.join(save_path, '%s_metrics.json' % scene.replace('/', '-'))
            json.dump(metrics, open(rslt_file, 'w'))

            return scene, metrics


        @ray.remote(num_cpus=num_workers + 1, num_gpus=(1 / n_proc))
        def process_with_single_worker(info_files, data_path, save_path, gt_path, max_depth, loader_num_workers):
            metrics = {}
            for i, info_file in enumerate(info_files):
                scene, temp = process(info_file, i, len(info_files),
                            data_path, save_path, gt_path, max_depth, loader_num_workers)
                if temp is not None:
                    metrics[scene] = temp
            return metrics


        def split_list(_list, n):
            assert len(_list) >= n
            ret = [[] for _ in range(n)]
            for idx, item in enumerate(_list):
                ret[idx % n].append(item)
            return ret

        all_proc = n_proc * n_gpu

        ray.init(num_cpus=all_proc * (num_workers + 1), num_gpus=n_gpu)

        info_files = sorted(os.listdir(data_path))

        info_files = split_list(info_files, all_proc)

        ray_worker_ids = []
        for w_idx in range(all_proc):
            ray_worker_ids.append(process_with_single_worker.remote(info_files[w_idx],
                                     data_path, save_path, gt_path, max_depth, loader_num_workers))

        try:
            results = ray.get(ray_worker_ids, timeout=14400)
        except GetTimeoutError:
            print("`get` timed out.")

        metrics = {}
        for r in results:
            metrics.update(r)

        rslt_file = os.path.join(save_path, 'metrics.json')
        json.dump(metrics, open(rslt_file, 'w'))

        # display results
        parse_metrics_neucon(rslt_file)
        print('parse_metrics_neucon end')
        return metrics


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
