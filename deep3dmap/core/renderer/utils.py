import torch
import torch.nn.functional as F

def mm_normalize(x, min=0, max=1):
    x_min = x.min()
    x_max = x.max()
    x_range = x_max - x_min
    x_z = (x - x_min) / x_range
    x_out = x_z * (max - min) + min
    return x_out


def rand_range(size, min, max):
    return torch.rand(size)*(max-min)+min


def rand_posneg_range(size, min, max):
    i = (torch.rand(size) > 0.5).type(torch.float)*2.-1.
    return i*rand_range(size, min, max)


def get_grid(b, H, W, normalize=True):
    if normalize:
        h_range = torch.linspace(-1,1,H)
        w_range = torch.linspace(-1,1,W)
    else:
        h_range = torch.arange(0,H)
        w_range = torch.arange(0,W)
    grid = torch.stack(torch.meshgrid([h_range, w_range]), -1).repeat(b,1,1,1).flip(3).float() # flip h,w to x,y
    return grid


def get_rotation_matrix(tx, ty, tz):
    m_x = torch.zeros((len(tx), 3, 3)).to(tx.device)
    m_y = torch.zeros((len(tx), 3, 3)).to(tx.device)
    m_z = torch.zeros((len(tx), 3, 3)).to(tx.device)

    m_x[:, 1, 1], m_x[:, 1, 2] = tx.cos(), -tx.sin()
    m_x[:, 2, 1], m_x[:, 2, 2] = tx.sin(), tx.cos()
    m_x[:, 0, 0] = 1

    m_y[:, 0, 0], m_y[:, 0, 2] = ty.cos(), ty.sin()
    m_y[:, 2, 0], m_y[:, 2, 2] = -ty.sin(), ty.cos()
    m_y[:, 1, 1] = 1

    m_z[:, 0, 0], m_z[:, 0, 1] = tz.cos(), -tz.sin()
    m_z[:, 1, 0], m_z[:, 1, 1] = tz.sin(), tz.cos()
    m_z[:, 2, 2] = 1
    return torch.matmul(m_z, torch.matmul(m_y, m_x))


def get_transform_matrices(view):
    b = view.size(0)
    if view.size(1) == 6:
        rx = view[:,0]
        ry = view[:,1]
        rz = view[:,2]
        trans_xyz = view[:,3:].reshape(b,1,3)
    elif view.size(1) == 5:
        rx = view[:,0]
        ry = view[:,1]
        rz = view[:,2]
        delta_xy = view[:,3:].reshape(b,1,2)
        trans_xyz = torch.cat([delta_xy, torch.zeros(b,1,1).to(view.device)], 2)
    elif view.size(1) == 3:
        rx = view[:,0]
        ry = view[:,1]
        rz = view[:,2]
        trans_xyz = torch.zeros(b,1,3).to(view.device)
    rot_mat = get_rotation_matrix(rx, ry, rz)
    return rot_mat, trans_xyz


def get_face_idx(b, h, w):
    idx_map = torch.arange(h*w).reshape(h,w)
    faces1 = torch.stack([idx_map[:h-1,:w-1], idx_map[1:,:w-1], idx_map[:h-1,1:]], -1).reshape(-1,3)
    faces2 = torch.stack([idx_map[:h-1,1:], idx_map[1:,:w-1], idx_map[1:,1:]], -1).reshape(-1,3)
    return torch.cat([faces1,faces2], 0).repeat(b,1,1).int()


def vcolor_to_texture_cube(vcolors):
    # input bxcxnx3
    b, c, n, f = vcolors.shape
    coeffs = torch.FloatTensor(
        [[ 0.5,  0.5,  0.5],
         [ 0. ,  0. ,  1. ],
         [ 0. ,  1. ,  0. ],
         [-0.5,  0.5,  0.5],
         [ 1. ,  0. ,  0. ],
         [ 0.5, -0.5,  0.5],
         [ 0.5,  0.5, -0.5],
         [ 0. ,  0. ,  0. ]]).to(vcolors.device)
    return coeffs.matmul(vcolors.permute(0,2,3,1)).reshape(b,n,2,2,2,c)


def get_textures_from_im(im, tx_size=1):
    b, c, h, w = im.shape
    if tx_size == 1:
        textures = torch.cat([im[:,:,:h-1,:w-1].reshape(b,c,-1), im[:,:,1:,1:].reshape(b,c,-1)], 2)
        textures = textures.transpose(2,1).reshape(b,-1,1,1,1,c)
    elif tx_size == 2:
        textures1 = torch.stack([im[:,:,:h-1,:w-1], im[:,:,:h-1,1:], im[:,:,1:,:w-1]], -1).reshape(b,c,-1,3)
        textures2 = torch.stack([im[:,:,1:,:w-1], im[:,:,:h-1,1:], im[:,:,1:,1:]], -1).reshape(b,c,-1,3)
        textures = vcolor_to_texture_cube(torch.cat([textures1, textures2], 2)) # bxnx2x2x2xc
    else:
        raise NotImplementedError("Currently support texture size of 1 or 2 only.")
    return textures


def pose_to_d9(pose: torch.Tensor) -> torch.Tensor:
    nbatch = pose.shape[0]
    R = pose[:, :3, :3]  # [N, 3, 3]
    t = pose[:, :3, -1]  # [N, 3]

    r6 = R[:, :2, :3].reshape(nbatch, -1)  # [N, 6]

    d9 = torch.cat((t, r6), -1)  # [N, 9]

    return d9


def r6d2mat(d6: torch.Tensor) -> torch.Tensor:
    """
    Converts 6D rotation representation by Zhou et al. [1] to rotation matrix
    using Gram--Schmidt orthogonalisation per Section B of [1].
    Args:
        d6: 6D rotation representation, of size (*, 6)

    Returns:
        batch of rotation matrices of size (*, 3, 3)

    [1] Zhou, Y., Barnes, C., Lu, J., Yang, J., & Li, H.
    On the Continuity of Rotation Representations in Neural Networks.
    IEEE Conference on Computer Vision and Pattern Recognition, 2019.
    Retrieved from http://arxiv.org/abs/1812.07035
    """

    a1, a2 = d6[..., :3], d6[..., 3:]
    b1 = F.normalize(a1, dim=-1)
    b2 = a2 - (b1 * a2).sum(-1, keepdim=True) * b1
    b2 = F.normalize(b2, dim=-1)
    b3 = torch.cross(b1, b2, dim=-1)
    return torch.stack((b1, b2, b3), dim=-2)


def look_at_rotation(camera_position, at=(0, 0, 0), up=(0., 0, 1), device: str = "cpu") -> torch.Tensor:
    # Format input and broadcast
    nbatch = camera_position.shape[0]
    camera_position = camera_position.to(device)
    if not torch.is_tensor(at):
        at = torch.tensor(at, dtype=torch.float32, device=device)
    at = at.expand(nbatch, 3)
    if not torch.is_tensor(up):
        up = torch.tensor(up, dtype=torch.float32, device=device)
    up = up.expand(nbatch, 3)

    for t, n in zip([camera_position, at, up], ["camera_position", "at", "up"]):
        if t.shape[-1] != 3:
            msg = "Expected arg %s to have shape (N, 3); got %r"
            raise ValueError(msg % (n, t.shape))
    z_axis = F.normalize(camera_position - at, eps=1e-5)
    x_axis = F.normalize(torch.cross(up, z_axis, dim=1), eps=1e-5)
    y_axis = F.normalize(torch.cross(z_axis, x_axis, dim=1), eps=1e-5)
    is_close = torch.isclose(x_axis, torch.tensor(0.0), atol=5e-3).all(dim=1, keepdim=True)
    if is_close.any():
        # print(f'warning: up vector {up[0].detach()} is close to x_axis {z_axis[0].detach()}')
        replacement = F.normalize(torch.cross(y_axis, z_axis, dim=1), eps=1e-5)
        x_axis = torch.where(is_close, replacement, x_axis)
    R = torch.cat((x_axis[:, None, :], y_axis[:, None, :], z_axis[:, None, :]), dim=1)
    return R.transpose(1, 2)
