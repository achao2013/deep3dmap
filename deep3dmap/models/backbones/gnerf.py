import torch
import torch.nn as nn
from collections import defaultdict
from einops import rearrange

from deep3dmap.core.renderer.renderer_nfvr import sample_pdf, inference
from deep3dmap.models.backbones import NeRF

class GNeRF(nn.Module):
    def __init__(self, ray_sampler, xyz_freq=10, dir_freq=4, fc_depth=8, fc_dim=256, skips=(4,),
                 N_samples=64, N_importance=64, chunk=1024 * 32, white_back=False):
        super(GNeRF, self).__init__()
        self.ray_sampler = ray_sampler
        self.chunk = chunk
        self.N_samples = N_samples
        self.N_importance = N_importance
        self.white_back = white_back
        self.noise_std = 1.0

        self.nerf = NeRF(xyz_freq=xyz_freq, dir_freq=dir_freq, fc_depth=fc_depth, fc_dim=fc_dim, skips=skips)

    def forward(self, coords, img_wh, poses=None):
        nbatch, h, w, _ = coords.shape
        device = coords.device

        noise_std = self.noise_std if self.training else 0.0
        perturb = 1.0 if self.training else 0.0

        poses = self.ray_sampler.random_poses(nbatch, device) if poses is None else poses

        rays = self.ray_sampler.get_rays(coords, poses, img_wh, device)
        rays = rearrange(rays, 'n h w c -> (n h w) c')

        results = {'coarse': defaultdict(list), 'fine': defaultdict(list)}
        for i in range(0, rays.shape[0], self.chunk):
            rendered_ray_chunks = self.render_rays(rays=rays[i:i + self.chunk], perturb=perturb, noise_std=noise_std)

            for k_1, v_1 in rendered_ray_chunks.items():
                for k_2, v_2 in v_1.items():
                    results[k_1][k_2] += [v_2]

        for k_1, v_1 in results.items():
            for k_2, v_2 in v_1.items():
                v_2 = torch.cat(v_2, 0)
                v_2 = rearrange(v_2, '(n h w) c -> n c h w', n=nbatch, h=h, w=w)
                results[k_1][k_2] = v_2 * 2.0 - 1.0

        if self.training:
            return results['coarse']['rgb'], results['fine']['rgb'], poses
        else:
            return results['fine']

    def render_rays(self, rays, use_disp=False, perturb=0.0, noise_std=1.0):
        N_rays = rays.shape[0]
        device = rays.device
        rays_o, rays_d = rays[:, 0:3], rays[:, 3:6]  # both [N_rays, 3]
        near, far = rays[:, 6:7], rays[:, 7:8]  # both [N_rays, 1]

        rets = {'coarse': {}, 'fine': {}}
        for i, type in enumerate(rets.keys()):
            if type == 'coarse':
                z_steps = torch.linspace(0, 1, self.N_samples, device=device)  # [N_samples]

                if not use_disp:  # use linear sampling in depth space
                    z_vals = near * (1 - z_steps) + far * z_steps
                else:  # use linear sampling in disparity space
                    z_vals = 1 / (1 / near * (1 - z_steps) + 1 / far * z_steps)

                z_vals = z_vals.expand(N_rays, self.N_samples)  # [N_rays, N_samples]
            else:
                z_vals_mid = 0.5 * (z_vals[:, :-1] + z_vals[:, 1:])  # (N_rays, N_samples-1) interval mid points
                new_z_vals = sample_pdf(z_vals_mid, weights[:, 1:-1], self.N_importance, det=(perturb == 0)).detach()
                # detach so that grad doesn't propogate to weights_coarse from here
                z_vals, _ = torch.sort(torch.cat([z_vals, new_z_vals], -1), -1)  # [N_rays, N_samples + N_importance]

            xyz_sampled = rays_o.unsqueeze(1) + rays_d.unsqueeze(1) * z_vals.unsqueeze(2)  # [(N_rays, N_samples, 3]

            rgb, depth, weights = inference(self.nerf, xyz_sampled, rays_d, z_vals,
                                            far, self.white_back, self.chunk, noise_std,
                                            weights_only=False)

            rets[type].update({
                'rgb': rgb,
                'depth': depth.detach()[:, None],
                'opacity': weights.sum(1).detach()[:, None]
            })

        return rets

    def decrease_noise(self, it):
        end_it = 5000
        if it < end_it:
            self.noise_std = 1.0 - float(it) / end_it