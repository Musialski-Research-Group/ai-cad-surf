import torch
import torch.nn.functional as F
from torch import nn
import numpy as np
from typing import Callable, List, Tuple, Union


def conv3x3(in_channels, out_channels, stride=1, 
            padding=1, bias=True, groups=1):    
    return nn.Conv2d(
        in_channels,
        out_channels,
        kernel_size=3,
        stride=stride,
        padding=padding,
        bias=bias,
        groups=groups)


def upconv2x2(in_channels, out_channels, mode='transpose'):
    if mode == 'transpose':
        return nn.ConvTranspose2d(
            in_channels,
            out_channels,
            kernel_size=2,
            stride=2)
    else:
        # out_channels is always going to be the same
        # as in_channels
        return nn.Sequential(
            nn.Upsample(mode='bilinear', scale_factor=2),
            conv1x1(in_channels, out_channels))


def conv1x1(in_channels, out_channels, groups=1):
    return nn.Conv2d(
        in_channels,
        out_channels,
        kernel_size=1,
        groups=groups,
        stride=1)


def grid_sample_2d(input, grid):
    # grid (B, Ho, Wo, 2)
    # input (B, C, H, W)
    # output (B, C, Ho, Wo)
    B, Ho, Wo, _ = grid.shape
    P = Ho * Wo
    _, C, H, W = input.shape

    # grid to index
    reso = torch.tensor([2.0 / (W-1), 2.0 / (H-1)]).view(1, 1, 1, 2).to(device=grid.device)
    # B, Ho, Wo, 2
    grid = (grid+1.0)/reso
    w1h1 = torch.floor(grid).long()
    w2h2 = w1h1 + 1
    assert(w1h1.min() >= 0 and w2h2[..., 0].max() <= W and w2h2[..., 1].max() <= H)

    input = F.pad(input, (0, 1, 0, 1), mode='replicate')

    # B, 4*P: Q11, Q12, Q21, Q22
    index = torch.stack([w1h1[...,1]*(W+1)+w1h1[...,0], (w1h1[...,1]+1)*(W+1)+w1h1[...,0],
                            w1h1[...,1]*(W+1)+w1h1[...,0]+1, (w1h1[...,1]+1)*(W+1)+w1h1[...,0]+1], dim=1).view(B, -1)
    # 4 of (B, C, P) (f11, f12, f21, f22)
    input_4samples = torch.gather(input.view(B, C, -1), -1, index.unsqueeze(1).expand(-1, C, -1)).split(P, dim=-1)

    # B, Ho, Wo, 2 -> B, P
    diff_x2x, diff_y2y = torch.unbind((w2h2 - grid).view(B, -1, 2), dim=-1)
    diff_xx1, diff_yy1 = torch.unbind((grid - w1h1).view(B, -1, 2), dim=-1)

    f11, f12, f21, f22 = input_4samples
    # B,1,P,2 @ B,C,P,2,2
    result = torch.stack([diff_x2x, diff_xx1],dim=-1).reshape(B,1,P,1,2).expand(-1,C,-1,-1,-1).reshape(-1,1,2) @ torch.stack([f11, f12, f21, f22],dim=-1).reshape(B,C,P,2,2).reshape(-1,2,2) @ torch.stack([diff_y2y, diff_yy1],dim=-1).reshape(B,1,P,2,1).expand(-1,C,-1,-1,-1).reshape(-1,2,1)
    result = result.view(B,C,Ho,Wo)

    return result


def grid_sample_3d(input, grid):
    """
    grid (B, Do, Ho, Wo, 3)
    input (B, C, D, H, W)
    output (B, C, Do, Ho, Wo)
    """
    B, Do, Ho, Wo, _ = grid.shape
    P = Ho * Wo * Do
    _, C, D, H, W = input.shape

    # ref = F.grid_sample(input, grid, padding_mode="border", align_corners=True)
    # grid to index B,1,1,1,3
    reso = torch.tensor([2.0/ (D-1), 2.0 / (W-1), 2.0 / (H-1)]).view(1, 1, 1, 1, 3).to(device=grid.device)
    # B, Ho, Wo, 2
    grid = (grid+1.0)/reso
    x000 = torch.floor(grid).long()

    input = F.pad(input, (0, 1, 0, 1, 0, 1), mode='replicate')

    # B, 8*P: Q000, Q001, Q010, Q011, Q100, Q101, Q110, Q111
    index = torch.stack([(x000[...,1]+x000[...,2]*(H+1))*(W+1)+x000[...,0],
                         (x000[...,1]+(x000[...,2]+1)*(H+1))*(W+1)+x000[...,0],
                         (x000[...,1]+1+x000[...,2]*(H+1))*(W+1)+x000[...,0],
                         (x000[...,1]+1+(x000[...,2]+1)*(H+1))*(W+1)+x000[...,0],
                         (x000[...,1]+x000[...,2]*(H+1))*(W+1)+x000[...,0]+1,
                         (x000[...,1]+(x000[...,2]+1)*(H+1))*(W+1)+x000[...,0]+1,
                         (x000[...,1]+1+x000[...,2]*(H+1))*(W+1)+x000[...,0]+1,
                         (x000[...,1]+1+(x000[...,2]+1)*(H+1))*(W+1)+x000[...,0]+1,
                         ], dim=1).view(B, -1)
    # 2 of (B, C, 4P) corresponding f000, f001, f010, f011, f100, f101, f110, f111
    f0xx, f1xx = torch.gather(input.view(B, C, -1), -1, index.unsqueeze(1).expand(-1, C, -1)).split(P*4, dim=-1)

    # B, Ho, Wo, 3 -> B, P
    xd, yd, zd = torch.unbind((grid - x000).view(B, -1, 3), dim=-1)

    # B, C, 4P
    fxx = f0xx * (1-xd).repeat(1, 4).unsqueeze(1) + f1xx * xd.repeat(1,4).unsqueeze(1)

    # f00, f01: B, C, 2P
    f0x, f1x = fxx.split(2*P, dim=-1)
    fx = f0x * (1-yd).repeat(1, 2).unsqueeze(1) + f1x*yd.repeat(1,2).unsqueeze(1)

    f0, f1 = fx.split(P, dim=-1)
    result = f0*(1-zd).unsqueeze(1) + f1*zd.unsqueeze(1)

    result = result.view(B,C,Do,Ho,Wo)
    # assert(torch.allclose(ref, result))
    return result



def get_knn(vs: torch.Tensor, k: int, batch_idx: torch.Tensor) -> torch.Tensor:
    mat_square = torch.matmul(vs, vs.transpose(2, 1))
    diag = torch.diagonal(mat_square, dim1=1, dim2=2)
    diag = diag.unsqueeze(2).expand(mat_square.shape)
    dist_mat = (diag + diag.transpose(2, 1) - 2 * mat_square)
    _, index = dist_mat.topk(k + 1, dim=2, largest=False, sorted=True)
    index = index[:, :, 1:].view(-1, k) + batch_idx[:, None] * vs.shape[1]
    return index.flatten()


def extract_angles(vs: torch.Tensor, distance_k: torch.Tensor, vs_k: torch.Tensor) -> Union[
    Tuple[torch.Tensor, ...], List[torch.Tensor]]:
    proj = torch.einsum('nd,nkd->nk', vs, vs_k)
    cos_angles = torch.clamp(proj / distance_k, -1., 1.)
    proj = vs_k - vs[:, None, :] * proj[:, :, None]
    # moving same axis points
    ma = torch.abs(proj).sum(2) == 0
    num_points_to_replace = ma.sum().item()
    if num_points_to_replace:
        proj[ma] = torch.rand(num_points_to_replace,
                              vs.shape[1], device=ma.device)
    proj = proj / torch.norm(proj, p=2, dim=2)[:, :, None]
    angles = torch.acos(cos_angles)
    return angles, proj


def min_angles(dirs: torch.Tensor, up: torch.Tensor) -> torch.Tensor:
    ref = dirs[:, 0]
    all_cos = torch.einsum('nd,nkd->nk', ref, dirs)
    all_sin = torch.cross(ref.unsqueeze(
        1).expand(-1, dirs.shape[1], -1), dirs, dim=2)
    all_sin = torch.einsum('nd,nkd->nk', up, all_sin)
    all_angles = torch.atan2(all_sin, all_cos)
    all_angles[:, 0] = 0
    all_angles[all_angles < 0] = all_angles[all_angles < 0] + 2 * np.pi
    all_angles, inds = all_angles.sort(dim=1)
    inds = torch.argsort(inds, dim=1)
    all_angles_0 = 2 * np.pi - all_angles[:, -1]
    all_angles[:, 1:] = all_angles[:, 1:] - all_angles[:, :-1]
    all_angles[:, 0] = all_angles_0
    all_angles = torch.gather(all_angles, 1, inds)
    return all_angles


def extract_rotation_invariant_features(k: int) -> Tuple[Callable[[Union[torch.Tensor, np.array]], torch.Tensor], int]:
    batch_idx = None
    num_features = k * 3 + 1

    def get_input(xyz: torch.Tensor) -> Union[Tuple[torch.Tensor, ...], List[torch.Tensor]]:
        nonlocal batch_idx
        if batch_idx is None or len(batch_idx) != xyz.shape[0] * xyz.shape[1]:
            batch_idx, _ = torch.meshgrid(
                [torch.arange(xyz.shape[0]), torch.arange(xyz.shape[1])])
            batch_idx = batch_idx.flatten().to(xyz.device)
        return xyz.view(-1, 3), batch_idx

    def extract(base_vs: Union[torch.Tensor, np.array]):
        nonlocal num_features
        with torch.no_grad():
            if type(base_vs) is np.array:
                base_vs = torch.Tensor(base_vs)
            batch_size, num_pts = base_vs.shape[:2]
            vs, batch_idx = get_input(base_vs)
            knn = get_knn(base_vs, k, batch_idx)
            vs_k = vs[knn].view(-1, k, vs.shape[1])
            distance = torch.norm(vs, p=2, dim=1)
            vs_unit = vs / distance[:, None]
            distance_k = distance[knn].view(-1, k)
            angles, proj_unit = extract_angles(vs_unit, distance_k, vs_k)
            proj_min_angle = min_angles(proj_unit, vs_unit)
            fe = torch.cat([distance.unsqueeze(1), distance_k,
                            angles, proj_min_angle], dim=1)
        return fe.view(batch_size, num_pts, num_features)

    return extract, num_features
