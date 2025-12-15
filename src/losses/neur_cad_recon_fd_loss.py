import torch
import torch.nn as nn

import utils
from .term import eikonal_loss, get_fermi, finite_neurcad_loss


class NeurCADReconLossFD(nn.Module):
    #call for class: criterion = LOSSES[config.loss.name](config.loss)
    def __init__(self,config):
        super().__init__()
        self.config = config
        self.weights = config.weights
        self.d_offset = config.d_offset_start
        self.h_step = config.offset_h
        self.finite_difference = config.finite_difference
        self.eikonal_type = config.eikonal_type

        self.use_denominator = config.use_denominator

    def forward(self, output_pred: dict, input_point: dict, **kwargs):
        #input gt:
        # 'points': manifold_points,
        # 'mnfld_n': manifold_normals,
        # 'nonmnfld_points': nonmnfld_points,
        # 'near_points': near_points,     =manifold_points + σ * 𝒩(0, 1) same number of namifold points
        #source: nn/datasets/reconstruction.py __getitem__

        model = kwargs.get('model', None)

        #input points
        mnfld_points=input_point.get('points', None)
        nonmnfld_points=input_point.get('nonmnfld_points', None)
        near_points=input_point.get('near_points', None)

        #record grad to calculate dloss/dpoint
        mnfld_points.requires_grad_()
        nonmnfld_points.requires_grad_()
        near_points.requires_grad_()

        # Network predictions
        non_manifold_pred = output_pred["nonmanifold_pnts_pred"]  # f(x_off)
        manifold_pred = output_pred["manifold_pnts_pred"]  # f(x_surf)
        near_points_pred = output_pred["near_points_pred"]

        near_points_grad = utils.gradient(near_points, output_pred['near_points_pred'])

        #get sdf and gradient
        manifold_grad = utils.gradient(mnfld_points, manifold_pred)
        non_manifold_grad = utils.gradient(nonmnfld_points, non_manifold_pred)
        #loss
        device = mnfld_points.device

        div_loss = torch.zeros((), device=device)
        morse_loss = torch.zeros((), device=device)
        offset_loss = torch.zeros((), device=device)
        curv_term = torch.zeros((), device=device)
        normal_term = torch.zeros((), device=device)
        min_surf_loss = torch.zeros((), device=device)

        # manifold loss
        sdf_term=(torch.abs(manifold_pred)).mean()

        # non-manifold loss
        alpha=100
        inter_term=(torch.exp(-alpha * torch.abs(non_manifold_pred))).mean()

        # eikonal loss
        eikonal_term=eikonal_loss(non_manifold_grad, manifold_grad, self.eikonal_type)

        # morse loss
        u, v, x_w = get_fermi(model, near_points, near_points_grad, self.d_offset)
        morse_loss = finite_neurcad_loss(x_w, near_points_pred, u, v, self.h_step, model, self.use_denominator, near_points_grad).mean()

        total_loss=self.weights[0] * sdf_term + \
               self.weights[1] * inter_term + \
               self.weights[3] * eikonal_term + \
               self.weights[5] * morse_loss

        return {
            "loss": total_loss,
            'sdf_term': sdf_term,
            'inter_term': inter_term,
            'latent_reg_term': 0,
            'eikonal_term': eikonal_term,
            'normals_loss': normal_term,
            'div_loss': div_loss,
            'curv_loss': curv_term.mean(),
            'min_surf_loss': min_surf_loss,
            'morse_term': morse_loss,
            'offset_term': offset_loss
        }, manifold_grad



    def update(self, current_iteration, n_iterations, div_params=None, eik_params=None):
        return