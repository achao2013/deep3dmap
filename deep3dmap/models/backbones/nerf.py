import torch
import torch.nn as nn
from collections import defaultdict
from einops import rearrange
from ..builder import BACKBONES
from deep3dmap.models.modulars.embeddings import HighDimEmbedding

@BACKBONES.register_module()
class NeRF(nn.Module):
    def __init__(self, xyz_freq=10, dir_freq=4, fc_depth=8, fc_dim=256, skips=(4,)):
        super(NeRF, self).__init__()
        self.fc_depth = fc_depth
        self.fc_dim = fc_dim
        self.skips = skips

        self.embedding_xyz = HighDimEmbedding(3, xyz_freq)
        self.embedding_dir = HighDimEmbedding(3, dir_freq)
        self.in_channels_xyz = self.embedding_xyz.out_channels
        self.in_channels_dir = self.embedding_dir.out_channels

        # xyz encoding layers
        for i in range(fc_depth):
            if i == 0:
                layer = nn.Linear(self.in_channels_xyz, fc_dim)
            elif i in skips:
                layer = nn.Linear(fc_dim + self.in_channels_xyz, fc_dim)
            else:
                layer = nn.Linear(fc_dim, fc_dim)
            layer = nn.Sequential(layer, nn.ReLU(True))
            setattr(self, f"xyz_encoding_{i + 1}", layer)
        self.xyz_encoding_final = nn.Linear(fc_dim, fc_dim)

        # output layers
        self.sigma = nn.Linear(fc_dim, 1)

        # direction encoding layers
        self.rgb = nn.Sequential(
            nn.Linear(fc_dim + self.in_channels_dir, fc_dim // 2),
            nn.ReLU(True),
            nn.Linear(fc_dim // 2, 3),
            nn.Sigmoid()
        )

    def forward(self, x, sigma_only=False):
        if not sigma_only:
            input_xyz, input_dir = torch.split(x, [3, 3], dim=-1)

            input_xyz = self.embedding_xyz(input_xyz)
            input_dir = self.embedding_dir(input_dir)
        else:
            input_xyz = x

            input_xyz = self.embedding_xyz(input_xyz)

        xyz_ = input_xyz
        for i in range(self.fc_depth):
            if i in self.skips:
                xyz_ = torch.cat([input_xyz, xyz_], -1)
            xyz_ = getattr(self, f"xyz_encoding_{i + 1}")(xyz_)

        sigma = self.sigma(xyz_)
        if sigma_only:
            return sigma

        xyz_encoding_final = self.xyz_encoding_final(xyz_)

        dir_encoding = torch.cat([xyz_encoding_final, input_dir], -1)

        rgb = self.rgb(dir_encoding)

        out = torch.cat([rgb, sigma], -1)

        return out
