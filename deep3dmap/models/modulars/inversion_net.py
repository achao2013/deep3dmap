from vit_pytorch import ViT
import torch
import torch.nn as nn
from ..builder import BACKBONES

@BACKBONES.register_module()
class InversionNet(nn.Module):
    def __init__(self, imsize, pose_mode):
        super(InversionNet, self).__init__()
        self.imsize = imsize
        self.pose_mode = pose_mode

        if pose_mode == '3d':
            final_dims = 3  # [N, 3]
        elif pose_mode == '6d':
            final_dims = 9  # [N, 9]
        else:
            raise NotImplementedError

        self.main = ViT(
            image_size=self.imsize,
            patch_size=self.imsize // 16,
            num_classes=final_dims,
            dim=256,
            depth=6,
            heads=16,
            mlp_dim=256
        )

    def forward(self, img):
        em = self.main(img)

        return em