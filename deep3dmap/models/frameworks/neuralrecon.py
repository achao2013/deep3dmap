import torch
import torch.nn as nn

from deep3dmap.models.backbones.mnas_multi import MnasMulti
from ..neucon_network import NeuConNet
from ..modulars.gru_fusion import GRUFusion
from deep3dmap.core.utils.neucon_utils import tocuda
from ..builder import MODELS
from deep3dmap.models.frameworks import BaseFramework

@MODELS.register_module()
class NeuralRecon(BaseFramework):
    '''
    NeuralRecon main class.
    '''

    def __init__(self, model_cfgs, train_cfg=None, test_cfg=None):
        super(NeuralRecon, self).__init__()
        self.model_cfg = model_cfgs
        alpha = float(self.model_cfg.BACKBONE2D.ARC.split('-')[-1])
        # other hparams
        self.n_scales = len(self.model_cfg.THRESHOLDS) - 1

        # networks
        self.backbone2d = MnasMulti(alpha)
        self.neucon_net = NeuConNet(model_cfgs)
        # for fusing to global volume
        self.fuse_to_global = GRUFusion(model_cfgs, direct_substitute=True)

        
    def extract_feat(self, img):
        """Directly extract features from the backbone+neck."""
        x = self.backbone2d(img)
        if self.with_neck:
            x = self.neck(x)
        return x
    def extract_feats(self, imgs):
        """Extract features from multiple images.

        Args:
            imgs (list[torch.Tensor]): A list of images. The images are
                augmented from the same image but in different ways.

        Returns:
            list[torch.Tensor]: Features of different images
        """
        assert isinstance(imgs, list) or isinstance(imgs, tuple)
        return [self.extract_feat(img) for img in imgs]
    def forward_train(self, inputs, save_mesh=False):
        '''

        :param inputs: dict: {
            'imgs':                    (Tensor), images,
                                    (batch size, number of views, C, H, W)
            'vol_origin':              (Tensor), origin of the full voxel volume (xyz position of voxel (0, 0, 0)),
                                    (batch size, 3)
            'vol_origin_partial':      (Tensor), origin of the partial voxel volume (xyz position of voxel (0, 0, 0)),
                                    (batch size, 3)
            'world_to_aligned_camera': (Tensor), matrices: transform from world coords to aligned camera coords,
                                    (batch size, number of views, 4, 4)
            'proj_matrices':           (Tensor), projection matrix,
                                    (batch size, number of views, number of scales, 4, 4)
            when we have ground truth:
            'tsdf_list':               (List), tsdf ground truth for each level,
                                    [(batch size, DIM_X, DIM_Y, DIM_Z)]
            'occ_list':                (List), occupancy ground truth for each level,
                                    [(batch size, DIM_X, DIM_Y, DIM_Z)]
            others: unused in network
        }
        :param save_mesh: a bool to indicate whether or not to save the reconstructed mesh of current sample
        :return: outputs: dict: {
            'coords':                  (Tensor), coordinates of voxels,
                                    (number of voxels, 4) (4 : batch ind, x, y, z)
            'tsdf':                    (Tensor), TSDF of voxels,
                                    (number of voxels, 1)
            When it comes to save results:
            'origin':                  (List), origin of the predicted partial volume,
                                    [3]
            'scene_tsdf':              (List), predicted tsdf volume,
                                    [(nx, ny, nz)]
        }
                 loss_dict: dict: {
            'tsdf_occ_loss_X':         (Tensor), multi level loss
            'total_loss':              (Tensor), total loss
        }
        '''
        
        outputs = {}
        inputs = tocuda(inputs)
        imgs = torch.unbind(inputs['imgs'], 1)
        #print('imgs:',imgs)

        # image feature extraction
        # in: images; out: feature maps
        features = self.extract_feats(imgs)
        print("features:",len(features),len(features[0][0]),features[0][0].shape)

        # coarse-to-fine decoder: SparseConv and GRU Fusion.
        # in: image feature; out: sparse coords and tsdf
        outputs, loss_dict = self.neucon_net(features, inputs, outputs)

        # fuse to global volume.
        if not self.training and 'coords' in outputs.keys():
            outputs = self.fuse_to_global(outputs['coords'], outputs['tsdf'], inputs, self.n_scales, outputs, save_mesh)

        # gather loss.
        print_loss = 'Loss: '
        for k, v in loss_dict.items():
            print_loss += f'{k}: {v} '

        weighted_loss = 0

        for i, (k, v) in enumerate(loss_dict.items()):
            weighted_loss += v * self.model_cfg.LW[i]

        loss_dict.update({'total_loss': weighted_loss})
        return outputs, loss_dict
    
    def simple_test(self, img, img_metas, **kwargs):
        pass

    
    def aug_test(self, imgs, img_metas, **kwargs):
        """Test function with test time augmentation."""
        pass