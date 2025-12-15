import torch
import torch.nn as nn

import utils
from .term import eikonal_loss, latent_rg_loss


class NeuralSingularHessianLoss(nn.Module):
    def __init__(self, config):
        super().__init__()
        if config.weights is None:
            config.weights = [3e3, 1e2, 1e2, 5e1, 1e2, 1e1]
        self.weights = config.weights  # sdf, intern, normal, eikonal, div
        self.loss_type = config.type
        self.div_decay = config.div_decay
        self.div_type = config.div_type
        self.use_morse = True if config.weight_for_morse else False
        self.bidirectional_morse = config.bidirectional_morse

    def forward(self, output_pred: dict, gt: dict):
        mnfld_points = gt.get('points')
        nonmnfld_points = gt.get('nonmnfld_points')
        mnfld_points.requires_grad_()
        nonmnfld_points.requires_grad_()
        mnfld_n_gt = gt.get('mnfld_n_gt', None)
        near_points = gt.get('near_points', None)

        dims = mnfld_points.shape[-1]
        device = mnfld_points.device

        #########################################
        # Compute required terms
        #########################################

        non_manifold_pred = output_pred["nonmanifold_pnts_pred"]
        manifold_pred = output_pred["manifold_pnts_pred"]
        latent_reg = output_pred["latent_reg"]

        div_loss = torch.tensor([0.0], device=mnfld_points.device)
        morse_loss = torch.tensor([0.0], device=mnfld_points.device)
        curv_term = torch.tensor([0.0], device=mnfld_points.device)
        latent_reg_term = torch.tensor([0.0], device=mnfld_points.device)
        normal_term = torch.tensor([0.0], device=mnfld_points.device)

        # compute gradients for div (divergence), curl and curv (curvature)
        if manifold_pred is not None:
            mnfld_grad = utils.gradient(mnfld_points, manifold_pred)
        else:
            mnfld_grad = None

        nonmnfld_grad = utils.gradient(nonmnfld_points, non_manifold_pred)

        morse_nonmnfld_points = None
        morse_nonmnfld_grad = None
        if self.use_morse and near_points is not None:
            morse_nonmnfld_points = near_points
            morse_nonmnfld_grad = utils.gradient(near_points, output_pred['near_points_pred'])
        elif self.use_morse and near_points is None:
            morse_nonmnfld_points = nonmnfld_points
            morse_nonmnfld_grad = nonmnfld_grad

        if self.use_morse:
            nonmnfld_dx = utils.gradient(morse_nonmnfld_points, morse_nonmnfld_grad[:, :, 0])
            nonmnfld_dy = utils.gradient(morse_nonmnfld_points, morse_nonmnfld_grad[:, :, 1])

            mnfld_dx = utils.gradient(mnfld_points, mnfld_grad[:, :, 0])
            mnfld_dy = utils.gradient(mnfld_points, mnfld_grad[:, :, 1])
            if dims == 3:
                nonmnfld_dz = utils.gradient(morse_nonmnfld_points, morse_nonmnfld_grad[:, :, 2])
                nonmnfld_hessian_term = torch.stack((nonmnfld_dx, nonmnfld_dy, nonmnfld_dz), dim=-1)

                mnfld_dz = utils.gradient(mnfld_points, mnfld_grad[:, :, 2])
                mnfld_hessian_term = torch.stack((mnfld_dx, mnfld_dy, mnfld_dz), dim=-1)
            else:
                nonmnfld_hessian_term = torch.stack((nonmnfld_dx, nonmnfld_dy), dim=-1)
                mnfld_hessian_term = torch.stack((mnfld_dx, mnfld_dy), dim=-1)

            nonmnfld_det = torch.det(nonmnfld_hessian_term)
            mnfld_det = torch.det(mnfld_hessian_term)

            morse_mnfld = torch.tensor([0.0], device=mnfld_points.device)
            morse_nonmnfld = torch.tensor([0.0], device=mnfld_points.device)
            if self.div_type == 'l2':
                morse_nonmnfld = nonmnfld_det.square().mean()
                if self.bidirectional_morse:
                    morse_mnfld = mnfld_det.square().mean()
            elif self.div_type == 'l1':
                morse_nonmnfld = nonmnfld_det.abs().mean()
                # morse_nonmnfld = morse_nonmnfld_grad.norm(dim=-1).square().mean()
                # morse_nonmnfld = nonmnfld_hessian_term.norm(dim=[-1, -2]).square().mean()
                # nonmnfld_divergence = nonmnfld_dx[:, :, 0] + nonmnfld_dy[:, :, 1] + nonmnfld_dz[:, :, 2]
                # morse_nonmnfld = torch.clamp(torch.abs(nonmnfld_divergence), 0.1, 50).mean()
                if self.bidirectional_morse:
                    morse_mnfld = mnfld_det.abs().mean()

            morse_loss = 0.5 * (morse_nonmnfld + morse_mnfld)

        # latent regulariation for multiple shape learning
        latent_reg_term = latent_rg_loss(latent_reg, device)

        # normal term
        if mnfld_n_gt is not None:
            if 'igr' in self.loss_type:
                normal_term = ((mnfld_grad - mnfld_n_gt).abs()).norm(2, dim=1).mean()
            else:
                normal_term = (
                        1 - torch.abs(torch.nn.functional.cosine_similarity(mnfld_grad, mnfld_n_gt, dim=-1))).mean()

        # signed distance function term
        sdf_term = torch.abs(manifold_pred).mean()
        # sdf_term = (torch.abs(manifold_pred) * torch.exp(manifold_pred.abs())).mean()

        # eikonal term
        # eikonal_term = eikonal_loss(nonmnfld_grad, mnfld_grad=mnfld_grad, eikonal_type='abs')
        # Sometimes > relax may leading to bad results, use another type relax
        # eikonal_term = relax_eikonal_loss(None, mnfld_grad=mnfld_grad, udf=self.udf)
        eikonal_term = eikonal_loss(None, mnfld_grad=mnfld_grad, eikonal_type='abs')

        # inter term
        inter_term = torch.exp(-1e2 * torch.abs(non_manifold_pred)).mean()
        #########################################
        # Losses
        #########################################

        # losses used in the paper
        if self.loss_type == 'siren':  # SIREN loss
            loss = self.weights[0] * sdf_term + self.weights[1] * inter_term + \
                   self.weights[2] * normal_term + self.weights[3] * eikonal_term
        elif self.loss_type == 'siren_w_morse':
            loss = self.weights[0] * sdf_term + self.weights[1] * inter_term + \
                   self.weights[2] * normal_term + self.weights[3] * eikonal_term + \
                   self.weights[4] * morse_loss
        elif self.loss_type == 'siren_wo_n':  # SIREN loss without normal constraint
            self.weights[2] = 0
            loss = self.weights[0] * sdf_term + self.weights[1] * inter_term + self.weights[3] * eikonal_term
        elif self.loss_type == 'siren_wo_n_w_morse':
            self.weights[2] = 0
            loss = self.weights[0] * sdf_term + self.weights[1] * inter_term + self.weights[3] * eikonal_term + \
                   self.weights[5] * morse_loss
        elif self.loss_type == 'siren_wo_n_wo_e_wo_morse':
            loss = self.weights[0] * sdf_term + self.weights[1] * inter_term
        elif self.loss_type == 'igr':  # IGR loss
            self.weights[1] = 0
            loss = self.weights[0] * sdf_term + self.weights[2] * normal_term + self.weights[3] * eikonal_term
        elif self.loss_type == 'igr_wo_n':  # IGR without normals loss
            self.weights[1] = 0
            self.weights[2] = 0
            loss = self.weights[0] * sdf_term + self.weights[3] * eikonal_term
        elif self.loss_type == 'igr_wo_n_w_morse':
            self.weights[1] = 0
            self.weights[2] = 0
            loss = self.weights[0] * sdf_term + self.weights[3] * eikonal_term + self.weights[5] * morse_loss
        elif self.loss_type == 'siren_w_div':  # SIREN loss with divergence term
            loss = self.weights[0] * sdf_term + self.weights[1] * inter_term + \
                   self.weights[2] * normal_term + self.weights[3] * eikonal_term + \
                   self.weights[4] * div_loss
        elif self.loss_type == 'siren_wo_e_w_morse':
            self.weights[3] = 0
            self.weights[4] = 0
            loss = self.weights[0] * sdf_term + self.weights[1] * inter_term + \
                   self.weights[2] * normal_term + self.weights[5] * morse_loss
        elif self.loss_type == 'siren_wo_e_wo_n_w_morse':
            self.weights[2] = 0
            self.weights[3] = 0
            self.weights[4] = 0
            loss = self.weights[0] * sdf_term + self.weights[1] * inter_term + self.weights[5] * morse_loss
        elif self.loss_type == 'siren_wo_n_w_div':  # SIREN loss without normals and with divergence constraint
            loss = self.weights[0] * sdf_term + self.weights[1] * inter_term + self.weights[3] * eikonal_term + \
                   self.weights[4] * div_loss
        else:
            print(self.loss_type)
            raise Warning("unrecognized loss type")

        # If multiple surface reconstruction, then latent and latent_reg are defined so reg_term need to be used
        if latent_reg is not None:
            loss += self.weights[6] * latent_reg_term

        return {"loss": loss, 'sdf_term': sdf_term, 'inter_term': inter_term, 'latent_reg_term': latent_reg_term,
                'eikonal_term': eikonal_term, 'normals_loss': normal_term, 'div_loss': div_loss,
                'curv_loss': curv_term.mean(), 'morse_term': morse_loss}, mnfld_grad


    def update(self, current_iteration, n_iterations, params=None):
        # `params`` should be (start_weight, *optional middle, end_weight) where optional middle is of the form [percent, value]*
        # Thus (1e2, 0.5, 1e2 0.7 0.0, 0.0) means that the weight at [0, 0.5, 0.75, 1] of the training process, the weight should
        #   be [1e2,1e2,0.0,0.0]. Between these points, the weights change as per the div_decay parameter, e.g. linearly, quintic, step etc.
        #   Thus the weight stays at 1e2 from 0-0.5, decay from 1e2 to 0.0 from 0.5-0.75, and then stays at 0.0 from 0.75-1.

        if not hasattr(self, 'decay_params_list'):
            assert len(params) >= 2, params
            assert len(params[1:-1]) % 2 == 0
            self.decay_params_list = list(zip([params[0], *params[1:-1][1::2], params[-1]], [0, *params[1:-1][::2], 1]))

        curr = current_iteration / n_iterations
        we, e = min([tup for tup in self.decay_params_list if tup[1] >= curr], key=lambda tup: tup[1])
        w0, s = max([tup for tup in self.decay_params_list if tup[1] <= curr], key=lambda tup: tup[1])

        # Divergence term anealing functions
        if self.div_decay == 'linear':  # linearly decrease weight from iter s to iter e
            if current_iteration < s * n_iterations:
                self.weights[5] = w0
            elif current_iteration >= s * n_iterations and current_iteration < e * n_iterations:
                self.weights[5] = w0 + (we - w0) * (current_iteration / n_iterations - s) / (e - s)
            else:
                self.weights[5] = we
        elif self.div_decay == 'quintic':  # linearly decrease weight from iter s to iter e
            if current_iteration < s * n_iterations:
                self.weights[5] = w0
            elif current_iteration >= s * n_iterations and current_iteration < e * n_iterations:
                self.weights[5] = w0 + (we - w0) * (1 - (1 - (current_iteration / n_iterations - s) / (e - s)) ** 5)
            else:
                self.weights[5] = we
        elif self.div_decay == 'step':  # change weight at s
            if current_iteration < s * n_iterations:
                self.weights[5] = w0
            else:
                self.weights[5] = we
        elif self.div_decay == 'none':
            pass
        else:
            raise Warning("unsupported div decay value")
