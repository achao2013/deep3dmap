# Copyright 2022 achao2013.


from PIL import Image, ImageOps
import numpy as np
from deep3dmap.core.utils.neucon_utils import coordinates
import transforms3d
import torch
import deep3dmap
from deep3dmap.core.tsdf.tsdf_volume import TSDFVolumeTorch
from deep3dmap.datasets.pipelines.formating import to_tensor
from ..builder import PIPELINES

class Compose(object):
    """ Apply a list of transforms sequentially"""

    def __init__(self, transforms):
        self.transforms = transforms

    def __call__(self, data):
        for transform in self.transforms:
            data = transform(data)
        return data

@PIPELINES.register_module()
class SeqToTensor(object):
    """ Convert to torch tensors"""
    def __init__(self, imgslike_keys=['imgs'], common_keys=['extrinsics'], iter_keys=['tsdf_list_full']):
        self.imgslike_keys=imgslike_keys
        self.common_keys=common_keys
        self.iter_keys=iter_keys
    def __call__(self, data):
        for key in self.imgslike_keys:
            assert len(data[key])>=1
            if data[key][0].ndim==3 and data[key][0].shape[2]>1:
                data[key] = torch.Tensor(np.stack(data[key]).transpose([0, 3, 1, 2]))
            else:
                data[key] = torch.Tensor(np.stack(data[key]))
        for key in self.common_keys:
            data[key] = torch.Tensor(data[key])
        
        
        for key in self.iter_keys:            
            for i in range(len(data[key])):
                if not torch.is_tensor(data[key][i]):
                    data[key][i] = torch.Tensor(data[key][i])
        return data

@PIPELINES.register_module()
class SeqIntrinsicsPoseToProjection(object):
    """ Convert intrinsics and extrinsics matrices to a single projection matrix"""

    def __init__(self, n_views, stride=1, scale=3, in_intrinsics_key='intrinsics', in_extrinsics_key='extrinsics',
                    out_world2camera_key='world_to_aligned_camera', out_matrix_key='proj_matrices'):
        self.nviews = n_views
        self.stride = stride
        self.scale = scale
        self.in_intrinsics_key = in_intrinsics_key
        self.in_extrinsics_key = in_extrinsics_key
        self.out_world2camera_key = out_world2camera_key
        self.out_matrix_key = out_matrix_key
        

    def rotate_view_to_align_xyplane(self, Tr_camera_to_world):
        # world space normal [0, 0, 1]  camera space normal [0, -1, 0]
        z_c = np.dot(np.linalg.inv(Tr_camera_to_world), np.array([0, 0, 1, 0]))[: 3]
        axis = np.cross(z_c, np.array([0, -1, 0]))
        axis = axis / np.linalg.norm(axis)
        theta = np.arccos(-z_c[1] / (np.linalg.norm(z_c)))
        quat = transforms3d.quaternions.axangle2quat(axis, theta)
        rotation_matrix = transforms3d.quaternions.quat2mat(quat)
        return rotation_matrix

    def __call__(self, data):
        middle_pose = data[self.in_extrinsics_key][self.nviews // 2]
        rotation_matrix = self.rotate_view_to_align_xyplane(middle_pose)
        rotation_matrix4x4 = np.eye(4)
        rotation_matrix4x4[:3, :3] = rotation_matrix
        data[self.out_world2camera_key] = torch.from_numpy(rotation_matrix4x4).float() @ middle_pose.inverse()

        proj_matrices = []
        for intrinsics, extrinsics in zip(data[self.in_intrinsics_key], data[self.in_extrinsics_key]):
            view_proj_matrics = []
            for i in range(self.scale):
                # from (camera to world) to (world to camera)
                proj_mat = torch.inverse(extrinsics.data.cpu())
                scale_intrinsics = intrinsics / self.stride / 2 ** i
                scale_intrinsics[-1, -1] = 1
                proj_mat[:3, :4] = scale_intrinsics @ proj_mat[:3, :4]
                view_proj_matrics.append(proj_mat)
            view_proj_matrics = torch.stack(view_proj_matrics)
            proj_matrices.append(view_proj_matrics)
        data[self.out_matrix_key] = torch.stack(proj_matrices)
        data.pop(self.in_intrinsics_key)
        data.pop(self.in_extrinsics_key)
        return data


def pad_scannet(img, intrinsics):
    """ Scannet images are 1296x968 but 1296x972 is 4x3
    so we pad vertically 4 pixels to make it 4x3
    """

    w, h = img.size
    if w == 1296 and h == 968:
        img = ImageOps.expand(img, border=(0, 2))
        intrinsics[1, 2] += 2
    return img, intrinsics

@PIPELINES.register_module()
class SeqResizeImage968x1296(object):
    """ Resize everything to given size.

    Intrinsics are assumed to refer to image prior to resize.
    After resize everything (ex: depth) should have the same intrinsics
    """

    def __init__(self, size, imgs_key='imgs', intrinsics_key='intrinsics'):
        self.size = size
        self.imgs_key=imgs_key
        self.intrinsics_key=intrinsics_key

    def __call__(self, data):
        for i, im in enumerate(data[self.imgs_key]):
            
            im, intrinsics = pad_scannet(im, data[self.intrinsics_key][i])
            w, h = im.size
            im = im.resize(self.size, Image.BILINEAR)
            intrinsics[0, :] /= (w / self.size[0])
            intrinsics[1, :] /= (h / self.size[1])

            data[self.imgs_key][i] = np.array(im, dtype=np.float32)
            data[self.intrinsics_key][i] = intrinsics

        return data

    def __repr__(self):
        return self.__class__.__name__ + '(size={0})'.format(self.size)

@PIPELINES.register_module()
class SeqUnbindImages(object):
    """ Resize everything to given size.

    split data in different dim
    """


    def __call__(self, data, dim=1):
        
        data['imgs'] = torch.unbind(data['imgs'], dim)

        return data

    def __repr__(self):
        return self.__class__.__name__

@PIPELINES.register_module()
class SeqNormalizeImages():
    """Normalize images.
    Please refer to `mmdet.datasets.pipelines.transforms.py:Normalize` for
    detailed docstring.
    """

    def __init__(self, mean, std, keys=['imgs'], to_rgb=True):
        self.mean = to_tensor(np.array(mean, dtype=np.float32)).view(1, -1, 1, 1)
        self.std = to_tensor(np.array(std, dtype=np.float32)).view(1, -1, 1, 1)
        self.to_rgb = to_rgb
        self.keys=keys

    def __call__(self, data):
        """Call function.
        For each dict in results, call the call function of `Normalize` to
        normalize image.
        Args:
            results (list[dict]): List of dict that from
                :obj:`mmtrack.CocoVideoDataset`.
        Returns:
            list[dict]: List of dict that contains normalized results,
            'img_norm_cfg' key is added into result dict.
        """
        for key in self.keys:
            for i, im in enumerate(data[key]):
                im=(im-self.mean.type_as(im))/self.std.type_as(im)
                data[key][i]=im
        return data

@PIPELINES.register_module()
class SeqRandomTransformSpace(object):
    """ Apply a random 3x4 linear transform to the world coordinate system.
        This affects pose as well as TSDFs.
    """

    def __init__(self, voxel_dim, voxel_size, random_rotation=True, random_translation=True,
                 paddingXY=1.5, paddingZ=.25, origin=[0, 0, 0], max_epoch=999, max_depth=3.0,
                 in_origin_key='vol_origin', in_epoch_key='epoch', in_tsdf_key='tsdf_list_full', 
                 in_extrinsics_key='extrinsics', in_intrinsics_key='intrinsics', in_imgs_key='imgs', in_depth_key='depth', 
                 out_origin_partial_key='vol_origin_partial', out_tsdf_key='tsdf_list', out_occ_key='occ_list'):
        """
        Args:
            voxel_dim: tuple of 3 ints (nx,ny,nz) specifying
                the size of the output volume
            voxel_size: floats specifying the size of a voxel
            random_rotation: wheater or not to apply a random rotation
            random_translation: wheater or not to apply a random translation
            paddingXY: amount to allow croping beyond maximum extent of TSDF
            paddingZ: amount to allow croping beyond maximum extent of TSDF
            origin: origin of the voxel volume (xyz position of voxel (0,0,0))
            max_epoch: maximum epoch
            max_depth: maximum depth
        """
        self.in_origin_key=in_origin_key
        self.in_epoch_key=in_epoch_key
        self.in_tsdf_key=in_tsdf_key
        self.in_extrinsics_key=in_extrinsics_key
        self.in_intrinsics_key=in_intrinsics_key
        self.in_imgs_key=in_imgs_key
        self.in_depth_key=in_depth_key
        self.out_origin_partial_key=out_origin_partial_key
        self.out_tsdf_key=out_tsdf_key
        self.out_occ_key=out_occ_key

        self.voxel_dim = voxel_dim
        self.origin = origin
        self.voxel_size = voxel_size
        self.random_rotation = random_rotation
        self.random_translation = random_translation
        self.max_depth = max_depth
        self.padding_start = torch.Tensor([paddingXY, paddingXY, paddingZ])
        # no need to pad above (bias towards floor in volume)
        self.padding_end = torch.Tensor([paddingXY, paddingXY, 0])

        # each epoch has the same transformation
        self.random_r = torch.rand(max_epoch)
        self.random_t = torch.rand((max_epoch, 3))

    def __call__(self, data):
        origin = torch.Tensor(data[self.in_origin_key])
        if (not self.random_rotation) and (not self.random_translation):
            T = torch.eye(4)
        else:
            # construct rotaion matrix about z axis
            if self.random_rotation:
                r = self.random_r[data[self.in_epoch_key][0]] * 2 * np.pi
            else:
                r = 0
            # first construct it in 2d so we can rotate bounding corners in the plane
            R = torch.tensor([[np.cos(r), -np.sin(r)],
                              [np.sin(r), np.cos(r)]], dtype=torch.float32)

            # get corners of bounding volume
            voxel_dim_old = torch.tensor(data[self.in_tsdf_key][0].shape) * self.voxel_size
            xmin, ymin, zmin = origin
            xmax, ymax, zmax = origin + voxel_dim_old

            corners2d = torch.tensor([[xmin, xmin, xmax, xmax],
                                      [ymin, ymax, ymin, ymax]], dtype=torch.float32)

            # rotate corners in plane
            corners2d = R @ corners2d

            # get new bounding volume (add padding for data augmentation)
            xmin = corners2d[0].min()
            xmax = corners2d[0].max()
            ymin = corners2d[1].min()
            ymax = corners2d[1].max()
            zmin = zmin
            zmax = zmax

            # randomly sample a crop
            voxel_dim = list(data[self.in_tsdf_key][0].shape)
            start = torch.Tensor([xmin, ymin, zmin]) - self.padding_start
            end = (-torch.Tensor(voxel_dim) * self.voxel_size +
                   torch.Tensor([xmax, ymax, zmax]) + self.padding_end)
            if self.random_translation:
                t = self.random_t[data[self.in_epoch_key][0]]
            else:
                t = .5
            t = t * start + (1 - t) * end - origin

            T = torch.eye(4)

            T[:2, :2] = R
            T[:3, 3] = -t

        for i in range(len(data[self.in_extrinsics_key])):
            data[self.in_extrinsics_key][i] = T @ data[self.in_extrinsics_key][i]

        data[self.in_origin_key] = torch.tensor(self.origin, dtype=torch.float, device=T.device)

        data = self.transform(data, T.inverse(), old_origin=origin)

        return data

    def transform(self, data, transform=None, old_origin=None,
                  align_corners=False):
        """ Applies a 3x4 linear transformation to the TSDF.

        Each voxel is moved according to the transformation and a new volume
        is constructed with the result.

        Args:
            data: items from data loader
            transform: 4x4 linear transform
            old_origin: origin of the voxel volume (xyz position of voxel (0, 0, 0))
                default (None) is the same as the input
            align_corners:

        Returns:
            Items with new TSDF and occupancy in the transformed coordinates
        """

        # ----------computing visual frustum hull------------
        bnds = torch.zeros((3, 2))
        bnds[:, 0] = np.inf
        bnds[:, 1] = -np.inf

        for i in range(data[self.in_imgs_key].shape[0]):
            size = data[self.in_imgs_key][i].shape[1:]
            cam_intr = data[self.in_intrinsics_key][i]
            cam_pose = data[self.in_extrinsics_key][i]
            view_frust_pts = get_view_frustum(self.max_depth, size, cam_intr, cam_pose)
            bnds[:, 0] = torch.min(bnds[:, 0], torch.min(view_frust_pts, dim=1)[0])
            bnds[:, 1] = torch.max(bnds[:, 1], torch.max(view_frust_pts, dim=1)[0])

        # -------adjust volume bounds-------
        num_layers = 3
        center = (torch.tensor(((bnds[0, 1] + bnds[0, 0]) / 2, (bnds[1, 1] + bnds[1, 0]) / 2, -0.2)) - data[
            self.in_origin_key]) / self.voxel_size
        center[:2] = torch.round(center[:2] / 2 ** num_layers) * 2 ** num_layers
        center[2] = torch.floor(center[2] / 2 ** num_layers) * 2 ** num_layers
        origin = torch.zeros_like(center)
        origin[:2] = center[:2] - torch.tensor(self.voxel_dim[:2]) // 2
        origin[2] = center[2]
        vol_origin_partial = origin * self.voxel_size + data[self.in_origin_key]

        data[self.out_origin_partial_key] = vol_origin_partial

        # ------get partial tsdf and occupancy ground truth--------
        if self.in_tsdf_key in data.keys():
            # -------------grid coordinates------------------
            old_origin = old_origin.view(1, 3)

            x, y, z = self.voxel_dim
            coords = coordinates(self.voxel_dim, device=old_origin.device)
            world = coords.type(torch.float) * self.voxel_size + vol_origin_partial.view(3, 1)
            world = torch.cat((world, torch.ones_like(world[:1])), dim=0)
            world = transform[:3, :] @ world
            coords = (world - old_origin.T) / self.voxel_size

            data[self.out_tsdf_key] = []
            data[self.out_occ_key] = []

            for l, tsdf_s in enumerate(data[self.in_tsdf_key]):
                # ------get partial tsdf and occ-------
                vol_dim_s = torch.tensor(self.voxel_dim) // 2 ** l
                tsdf_vol = TSDFVolumeTorch(vol_dim_s, vol_origin_partial,
                                           voxel_size=self.voxel_size * 2 ** l, margin=3)
                for i in range(data[self.in_imgs_key].shape[0]):
                    depth_im = data[self.in_depth_key][i]
                    cam_intr = data[self.in_intrinsics_key][i]
                    cam_pose = data[self.in_extrinsics_key][i]

                    tsdf_vol.integrate(depth_im, cam_intr, cam_pose, obs_weight=1.)

                tsdf_vol, weight_vol = tsdf_vol.get_volume()
                occ_vol = torch.zeros_like(tsdf_vol).bool()
                occ_vol[(tsdf_vol < 0.999) & (tsdf_vol > -0.999) & (weight_vol > 1)] = True

                # grid sample expects coords in [-1,1]
                coords_world_s = coords.view(3, x, y, z)[:, ::2 ** l, ::2 ** l, ::2 ** l] / 2 ** l
                dim_s = list(coords_world_s.shape[1:])
                coords_world_s = coords_world_s.view(3, -1)

                old_voxel_dim = list(tsdf_s.shape)

                coords_world_s = 2 * coords_world_s / (torch.Tensor(old_voxel_dim) - 1).view(3, 1) - 1
                coords_world_s = coords_world_s[[2, 1, 0]].T.view([1] + dim_s + [3])

                # bilinear interpolation near surface,
                # no interpolation along -1,1 boundry
                tsdf_vol = torch.nn.functional.grid_sample(
                    tsdf_s.view([1, 1] + old_voxel_dim),
                    coords_world_s, mode='nearest', align_corners=align_corners
                ).squeeze()
                tsdf_vol_bilin = torch.nn.functional.grid_sample(
                    tsdf_s.view([1, 1] + old_voxel_dim), coords_world_s, mode='bilinear',
                    align_corners=align_corners
                ).squeeze()
                mask = tsdf_vol.abs() < 1
                tsdf_vol[mask] = tsdf_vol_bilin[mask]

                # padding_mode='ones' does not exist for grid_sample so replace
                # elements that were on the boarder with 1.
                # voxels beyond full volume (prior to croping) should be marked as empty
                mask = (coords_world_s.abs() >= 1).squeeze(0).any(3)
                tsdf_vol[mask] = 1

                data[self.out_tsdf_key].append(tsdf_vol)
                data[self.out_occ_key].append(occ_vol)
            data.pop(self.in_tsdf_key)
            #data.pop(self.in_depth_key)
        #data.pop('epoch')
        return data

    def __repr__(self):
        return self.__class__.__name__


def rigid_transform(xyz, transform):
    """Applies a rigid transform to an (N, 3) pointcloud.
    """
    xyz_h = torch.cat([xyz, torch.ones((len(xyz), 1))], dim=1)
    xyz_t_h = (transform @ xyz_h.T).T
    return xyz_t_h[:, :3]


def get_view_frustum(max_depth, size, cam_intr, cam_pose):
    """Get corners of 3D camera view frustum of depth image
    """
    im_h, im_w = size
    im_h = int(im_h)
    im_w = int(im_w)
    view_frust_pts = torch.stack([
        (torch.tensor([0, 0, 0, im_w, im_w]) - cam_intr[0, 2]) * torch.tensor(
            [0, max_depth, max_depth, max_depth, max_depth]) /
        cam_intr[0, 0],
        (torch.tensor([0, 0, im_h, 0, im_h]) - cam_intr[1, 2]) * torch.tensor(
            [0, max_depth, max_depth, max_depth, max_depth]) /
        cam_intr[1, 1],
        torch.tensor([0, max_depth, max_depth, max_depth, max_depth])
    ])
    view_frust_pts = rigid_transform(view_frust_pts.T, cam_pose).T
    return view_frust_pts
