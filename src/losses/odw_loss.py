import torch
import torch.nn as nn

import utils
from .term import eikonal_loss, first_order_morse_loss


class OffDiagonalWeingartenLoss(nn.Module):
    def __init__(self, config):
        super().__init__()

        # Loss weights: [sdf, intern, normal, eikonal, div, odw, offset]
        self.weights = config.weights

        self.eikonal_type = config.eikonal_type

        self.offset_dir = config.morse_offset_dir  # 'positive', 'negative', 'both'
        self.d_offset = config.offset_d


        self.use_offset_decay = config.use_offset_decay
        if self.use_offset_decay:
            self.d_offset_start = config.d_offset_start
            self.d_offset_end = config.d_offset_end

        self.h_offset = config.offset_h

        self.finite_difference = config.finite_difference
        
        self.use_near_points = config.use_near_points

        #anneal
        self.div_decay = config.div_decay
        self.decay_params= config.decay_params
        self.div_type= config.div_type

    def forward(self, output_pred: dict, gt: dict, **kwargs):

        model = kwargs.get('model', None)
        iteration = kwargs.get('iteration', None)
        n_iterations = kwargs.get('n_iterations', None)

        mnfld_points = gt.get('points')  # on-surface samples
        nonmnfld_points = gt.get('nonmnfld_points')  # off/near-surface
        near_points = gt.get('near_points', None)  # thin shell for Morse

        device = mnfld_points.device

        mnfld_points.requires_grad_()
        nonmnfld_points.requires_grad_()
        near_points.requires_grad_()

        # Network predictions
        non_manifold_pred = output_pred["nonmanifold_pnts_pred"]  # f(x_off)
        manifold_pred = output_pred["manifold_pnts_pred"]  # f(x_surf)
        # latent_reg        = output_pred["latent_reg"]

        div_loss = torch.zeros((), device=device)
        morse_loss = torch.zeros((), device=device)
        offset_loss = torch.zeros((), device=device)
        curv_term = torch.zeros((), device=device)
        normal_term = torch.zeros((), device=device)
        min_surf_loss = torch.zeros((), device=device)

        mnfld_grad = utils.gradient(mnfld_points, manifold_pred)
        nonmnfld_grad = utils.gradient(nonmnfld_points, non_manifold_pred)

        if near_points is not None:
            morse_nonmnfld_points = near_points
            morse_nonmnfld_grad = utils.gradient(near_points, output_pred['near_points_pred'])
        else:
            morse_nonmnfld_points = nonmnfld_points
            morse_nonmnfld_grad = nonmnfld_grad

        if self.use_offset_decay:
            d_offset = utils.interpolate(iteration / n_iterations, self.d_offset_start, self.d_offset_end, 'quadratic')
        else:
            d_offset = self.d_offset


        if self.use_near_points:
            morse_loss = first_order_morse_loss(
                model=model,
                x_surf=morse_nonmnfld_points,
                d_offset=d_offset,
                h_step=self.h_offset,
                d_direction=self.offset_dir,
                central=True,
                manifold_pred=manifold_pred,
                mnfld_grad=morse_nonmnfld_grad,
                finite_difference=self.finite_difference,
                use_near_points=self.use_near_points
            )
        else:
            morse_loss = first_order_morse_loss(
                model=model,
                x_surf=mnfld_points,
                d_offset=d_offset,
                h_step=self.h_offset,
                d_direction=self.offset_dir,
                central=True,
                manifold_pred=manifold_pred,
                mnfld_grad=mnfld_grad,
                finite_difference=self.finite_difference,
                use_near_points=self.use_near_points
            )

        # signed distance function term (DM-Term)
        sdf_term = torch.abs(manifold_pred).mean()

        # inter term (NDM-Term), default alpha=100
        alpha = 1e2
        inter_term = torch.exp(-alpha * torch.abs(non_manifold_pred)).mean()

        # eikonal term
        eikonal_term = eikonal_loss(nonmnfld_grad=morse_nonmnfld_grad, mnfld_grad=mnfld_grad, eikonal_type=self.eikonal_type)

        loss = self.weights[0] * sdf_term + \
               self.weights[1] * inter_term + \
               self.weights[3] * eikonal_term + \
               self.weights[5] * morse_loss

        return {
            "loss": loss,
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
        }, mnfld_grad

    def update(self, current_iteration, n_iterations, params=None):
        # `params`` should be (start_weight, *optional middle, end_weight) where optional middle is of the form [percent, value]*
        # Thus (1e2, 0.5, 1e2 0.7 0.0, 0.0) means that the weight at [0, 0.5, 0.7, 1] of the training process, the weight should
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
