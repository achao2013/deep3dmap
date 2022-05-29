import torch.nn as nn
import torch
from deep3dmap.core.renderer.utils import look_at_rotation, r6d2mat, pose_to_d9

#position encoder
class HighDimEmbedding(nn.Module):
    def __init__(self, in_channels, N_freqs, logscale=True):
        """
        Defines a function that embeds x to (x, sin(2^k x), cos(2^k x), ...)
        in_channels: number of input channels (3 for both xyz and direction)
        """
        super(HighDimEmbedding, self).__init__()
        self.N_freqs = N_freqs
        self.in_channels = in_channels
        self.funcs = [torch.sin, torch.cos]
        self.out_channels = in_channels * (len(self.funcs) * N_freqs + 1)

        if logscale:
            self.freq_bands = 2 ** torch.linspace(0, N_freqs - 1, N_freqs)
        else:
            self.freq_bands = torch.linspace(1, 2 ** (N_freqs - 1), N_freqs)

    def forward(self, x, dim=-1):
        """
        Embeds x to (x, sin(2^k x), cos(2^k x), ...)
        Different from the paper, "x" is also in the output
        See https://github.com/bmild/nerf/issues/12

        Inputs:
            x: (B, self.in_channels)

        Outputs:
            out: (B, self.out_channels)
        """
        out = [x]
        for freq in self.freq_bands:
            for func in self.funcs:
                out += [func(freq * x)]

        return torch.cat(out, dim)




class PoseParameters(nn.Module):
    def __init__(self, length, pose_mode, data):
        super(PoseParameters, self).__init__()
        self.length = length
        self.pose_mode = pose_mode
        self.data = data
        self.up = (0., 0, 1)
        # [N, 9]: (x, y, z, r1, r2) or [N, 3]: (x, y, z)
        self.poses_embed = nn.Parameter(self.init_poses_embed())

    def init_poses_embed(self):
        if self.pose_mode == '3d':
            poses_embed = torch.tensor([[0., 0, 1]]).repeat(self.length, 1)  # [N, 3]
        elif self.pose_mode == '6d':
            t = torch.tensor([[0., 0, 1]]).repeat(self.length, 1)  # [N, 3]
            R = look_at_rotation(t, up=self.up)  # [N, 3, 3]
            poses = torch.cat((R, t[..., None]), -1)
            poses_embed = pose_to_d9(poses)
        else:
            raise NotImplementedError

        return poses_embed

    @property
    def poses(self):
        if self.pose_mode == '3d':
            t = self.poses_embed[:, :3]  # [N, 3]
            R = look_at_rotation(t, device=t.device)  # [N, 3, 3]
        elif self.pose_mode == '6d':
            t = self.poses_embed[:, :3]  # [N, 3]
            r = self.poses_embed[:, 3:]
            R = r6d2mat(r)[:, :3, :3]  # [N, 3, 3]
        else:
            raise NotImplementedError

        poses = torch.cat((R, t[..., None]), -1)  # [N, 3, 4]

        return poses

    def forward(self, pose_indices=None):
        if pose_indices is None:
            return self.poses
        return self.poses[pose_indices]