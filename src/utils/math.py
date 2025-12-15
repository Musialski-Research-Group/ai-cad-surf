import math
import scipy.spatial as spatial

import torch
import torch.nn.functional as F
from torch.autograd import grad
import numpy as np

from utils.setup import get_cuda_ifavailable


def center_and_scale(points, cp=None, scale=None):
    """Center a point cloud and scale it to unite sphere."""
    if cp is None:
        cp = points.mean(axis=1)
    points = points - cp[:, None, :]
    if scale is None:
        scale = np.linalg.norm(points, axis=-1).max(-1)
    points = points / scale[:, None, None]
    return points, cp, scale


def scale_pc_to_unit_sphere(points, cp=None, s=None):
    if cp is None:
        cp = points.mean(axis=0)
    points = points - cp[None, :]
    if s is None:
        s = np.linalg.norm(points, axis=-1).max(-1)
    points = points / s
    return points, cp, s


def gradient(inputs, outputs, create_graph=True, retain_graph=True):
    d_points = torch.ones_like(outputs, requires_grad=False, device=outputs.device)
    points_grad = grad(
        outputs=outputs,
        inputs=inputs,
        grad_outputs=d_points,
        create_graph=create_graph,
        retain_graph=retain_graph,
        only_inputs=True)[0] # [:, -3:]
    return points_grad


def compute_props(decoder, latent, z_gt, device):
    """Compute derivative properties on a grid."""
    res = z_gt.shape[1]
    x, y, grid_points = get_2d_grid_uniform(resolution=res, range=1.2, device=device)
    grid_points.requires_grad_()
    if latent is not None:
        grid_points_latent = torch.cat([latent.expand(grid_points.shape[0], -1), grid_points], dim=1)
    else:
        grid_points_latent = grid_points
    z = decoder(grid_points_latent)
    z_np = z.detach().cpu().numpy().reshape(x.shape[0], x.shape[0])

    #plot z difference image
    z_diff = np.abs(np.abs(np.reshape(z_np, [res, res])) - np.abs(z_gt)).reshape(x.shape[0], x.shape[0])

    return x, y, z_np, z_diff


def compute_deriv_props(decoder, latent, z_gt, device):
    """Compute derivative properties on a grid."""
    res = z_gt.shape[1]
    x, y, grid_points = get_2d_grid_uniform(resolution=res, range=1.2, device=device)
    grid_points.requires_grad_()
    if latent is not None:
        grid_points_latent = torch.cat([latent.expand(grid_points.shape[0], -1), grid_points], dim=1)
    else:
        grid_points_latent = grid_points
    z = decoder(grid_points_latent)
    z_np = z.detach().cpu().numpy().reshape(x.shape[0], x.shape[0])

    # compute derivatives
    grid_grad = gradient(grid_points, z)
    dx = gradient(grid_points, grid_grad[:, 0], create_graph=False, retain_graph=True)
    dy = gradient(grid_points, grid_grad[:, 1], create_graph=False, retain_graph=False)

    grid_curl = (dx[:, 1] - dy[:, 0]).cpu().detach().numpy().reshape(x.shape[0], x.shape[0])

    # compute eikonal term (gradient magnitude)
    eikonal_term = ((grid_grad.norm(2, dim=-1) - 1) ** 2).cpu().detach().numpy().reshape(x.shape[0], x.shape[0])

    # compute divergence
    grid_div = (dx[:, 0] + dy[:, 1]).detach().cpu().numpy().reshape(x.shape[0], x.shape[0])

    # compute det of hessian with eigenvalues and eigenvectors
    hessian = torch.stack([dx, dy], dim=1)
    eigs = torch.linalg.eigvalsh(hessian).detach().cpu().numpy().reshape(x.shape[0], x.shape[0], 2)
    _, eigs_vecs = torch.linalg.eigh(hessian)
    eigs_vecs = eigs_vecs.detach().cpu().numpy().reshape(x.shape[0], x.shape[0], 2, 2)
    hessian_dot = torch.bmm(hessian, F.normalize(grid_grad, dim=-1)[:, :, None])

    hessian_det = hessian_dot.norm(p=2, dim=-2).cpu().detach().numpy().reshape(x.shape[0], x.shape[0])
    # hessian_det[hessian_det > 200] = 0

    # plot z difference image
    z_diff = np.abs(np.abs(np.reshape(z_np, [res, res])) - np.abs(z_gt)).reshape(x.shape[0], x.shape[0])

    # z_np = z_gt.reshape(x.shape[0], x.shape[0]) # if use, z_np=z_gt, plot the gt SDF.
    grid_grad = grid_grad.cpu().detach().numpy().reshape(x.shape[0], x.shape[0], 2)
    return x, y, z_np, z_diff, eikonal_term, grid_div, grid_curl, hessian_det, eigs, eigs_vecs, grid_grad


def get_2d_grid_uniform(resolution=100, range=1.2, device=None):
    """Generate points on a uniform grid within  a given range."""
    x = np.linspace(-range, range, resolution)
    y = x
    xx, yy = np.meshgrid(x, y)
    grid_points = get_cuda_ifavailable(torch.tensor(np.vstack([xx.ravel(), yy.ravel()]).T, dtype=torch.float),
                                             device=device)
    return x, y, grid_points


def get_3d_grid(resolution=100, bbox=1.2*np.array([[-1, 1], [-1, 1], [-1, 1]]), device=None, eps=0.1, dtype=np.float16):
    """
    Generate points on a uniform grid within a given range.
    Reimplemented from SAL: https://github.com/matanatz/SAL/blob/master/code/utils/plots.py
    and IGR: https://github.com/amosgropp/IGR/blob/master/code/utils/plots.py
    """
    shortest_axis = np.argmin(bbox[:, 1] - bbox[:, 0])
    if (shortest_axis == 0):
        x = np.linspace(bbox[0, 0] - eps,  bbox[0, 1] + eps, resolution)
        length = np.max(x) - np.min(x)
        y = np.arange(bbox[1, 0] - eps, bbox[1, 1] + length / (x.shape[0] - 1) + eps, length / (x.shape[0] - 1))
        z = np.arange(bbox[2, 0] - eps, bbox[2, 1] + length / (x.shape[0] - 1) + eps, length / (x.shape[0] - 1))
    elif (shortest_axis == 1):
        y = np.linspace(bbox[1, 0] - eps,  bbox[1, 1] + eps, resolution)
        length = np.max(y) - np.min(y)
        x = np.arange(bbox[0, 0] - eps, bbox[0, 1] + length / (y.shape[0] - 1) + eps, length / (y.shape[0] - 1))
        z = np.arange(bbox[2, 0] - eps, bbox[2, 1] + length / (y.shape[0] - 1) + eps, length / (y.shape[0] - 1))
    elif (shortest_axis == 2):
        z = np.linspace(bbox[2, 0] - eps, bbox[2, 1] + eps, resolution)
        length = np.max(z) - np.min(z)
        x = np.arange(bbox[0, 0] - eps, bbox[0, 1] + length / (z.shape[0] - 1) + eps, length / (z.shape[0] - 1))
        y = np.arange(bbox[1, 0] - eps, bbox[1, 1] + length / (z.shape[0] - 1) + eps, length / (z.shape[0] - 1))

    xx, yy, zz = np.meshgrid(x.astype(dtype), y.astype(dtype), z.astype(dtype)) #
    # grid_points = get_cuda_ifavailable(torch.tensor(np.vstack([xx.ravel(), yy.ravel(), zz.ravel()]).T, dtype=torch.float16),
    #                                          device=device)
    grid_points = torch.tensor(np.vstack([xx.ravel(), yy.ravel(), zz.ravel()]).T, dtype=torch.float16)
    return {"grid_points": grid_points,
            "shortest_axis_length": length,
            "xyz": [x, y, z],
            "shortest_axis_index": shortest_axis}


def interpolate(t: float, start: float, end: float, mode: str = "linear") -> float:

    t = max(0.0, min(1.0, t))

    if mode == "linear":
        return start + t * (end - start)

    elif mode == "quadratic":
        return start + (end - start) * (t * t)

    elif mode == "cosine":
        cos_t = (1 + math.cos(math.pi * (1 - t))) / 2
        return end + (start - end) * cos_t

    elif mode == "exponential":
        if start <= 0 or end <= 0:
            return start + t * (end - start)
        factor = (end / start) ** t
        return start * factor

    else:
        raise ValueError(f"Unknown mode: {mode!r}. Use 'linear', 'cosine', or 'exponential'.")


def knn_nearest_point_weights(points):
    kd_tree = spatial.KDTree(points)
    
    # query each point for sigma
    dist, _ = kd_tree.query(points, k=51, workers=-1)
    sigmas = dist[:, -1:]
    return sigmas
