import torch
import torch.nn as nn

import utils 
from .term import eikonal_loss, latent_rg_loss


class DiGSLoss(nn.Module):
    def __init__(self, config):
        super().__init__()
        # Store loss weights for different terms: [sdf, inter, normal, eikonal, divergence]
        self.weights = config.weights
        # Set the type of loss (which will determine which terms are used)
        self.loss_type = config.type
        # Set divergence weight annealing strategy
        self.div_decay = config.div_decay
        # Set divergence loss type: l1, l2, or ground truth based (gt_l1/gt_l2)
        self.div_type = config.div_type
        # Clamp value for divergence and curvature terms
        self.div_clamp = config.div_clamp
        # Determine if curvature terms should be used based on the loss type string
        self.use_curvs = True if 'curv' in self.loss_type else False
        # Determine if divergence terms should be used based on the loss type string
        self.use_div = True if 'div' in self.loss_type else False
        # Store the model for Hessian computation


    def forward(self, output_pred: dict, gt: dict):
        mnfld_points = gt.get('points')
        nonmnfld_points = gt.get('nonmnfld_points')
        mnfld_points.requires_grad_()
        nonmnfld_points.requires_grad_()

        mnfld_n_gt = gt.get('mnfld_n', None)
        # nonmnfld_dist = gt.get('nonmnfld_dist', None)
        curvatures = gt.get('curvatures', None)
        # network_weights = gt.get('network_weights', None)
        nonmnfld_div_gt = gt.get('nonmnfld_div_gt', None)
        mnfld_div_gt = gt.get('mnfld_div_gt', None)

        # Extract the dimensionality and device from the manifold points
        dims = mnfld_points.shape[-1]
        device = mnfld_points.device

        #########################################
        # Compute required terms
        #########################################
        # Extract predictions and latent information from the output dictionary
        non_manifold_pred = output_pred["nonmanifold_pnts_pred"]
        manifold_pred = output_pred["manifold_pnts_pred"]
        latent_reg = output_pred["latent_reg"]
        latent = output_pred.get("latent", None)

        # Initialize loss components to zero (on the correct device)
        div_loss = torch.tensor([0.0], device=mnfld_points.device)
        curv_term = torch.tensor([0.0], device=mnfld_points.device)
        latent_reg_term = torch.tensor([0.0], device=mnfld_points.device)

        # Compute gradients for manifold points if predictions exist, else set to None
        if manifold_pred is not None:
            mnfld_grad = utils.gradient(mnfld_points, manifold_pred)
        else:
            mnfld_grad = None

        # Compute gradients for non-manifold points
        nonmnfld_grad = utils.gradient(nonmnfld_points, non_manifold_pred)

        # # XXPMXX: Compute the Hessian for each sample in the manifold points
        # if manifold_pred is not None: 
        #     mnfld_hessian = vectorized_hessian(self.model, mnfld_points, nonmnfld_points)

        # ---------------------------
        # Compute Curvature Term
        # ---------------------------
        if self.use_curvs:
            if curvatures is None:
                raise Warning(" loss type requires curvatuers but none were provided.")
            # Compute partial derivatives (gradients) along each dimension for manifold points
            mnfld_dx = utils.gradient(mnfld_points, mnfld_grad[:, :, 0])
            mnfld_dy = utils.gradient(mnfld_points, mnfld_grad[:, :, 1])
            if dims == 3:
                mnfld_dz = utils.gradient(mnfld_points, mnfld_grad[:, :, 2])
                # Approximate divergence (sum of partial derivatives) in 3D
                mnfld_divergence = mnfld_dx[:, :, 0] + mnfld_dy[:, :, 1] + mnfld_dz[:, :, 2]
            else:
                # For 2D points, sum only x and y gradients
                mnfld_divergence = mnfld_dx[:, :, 0] + mnfld_dy[:, :, 1]

            # Compute curvature loss using either L2 or L1 formulation, clamped to avoid extreme values
            if self.div_type == 'l2':
                gt_mean_curvature = torch.square(torch.sum(curvatures, dim=-1))
                curv_term = torch.clamp((torch.square(mnfld_divergence) - gt_mean_curvature), 0.1, self.div_clamp)
            elif self.div_type == 'l1':
                gt_mean_curvature = torch.abs(torch.sum(curvatures, dim=-1))
                curv_term = torch.clamp((torch.abs(mnfld_divergence) - gt_mean_curvature), 0.1, self.div_clamp)
                # Alternative formulation (commented out):
                # curv_term = (torch.abs(mnfld_divergence) - gt_mean_curvature)

        # ---------------------------
        # Compute Divergence Term
        # ---------------------------
        if self.use_div:
            # Compute partial derivatives for non-manifold points
            nonmnfld_dx = utils.gradient(nonmnfld_points, nonmnfld_grad[:, :, 0])
            nonmnfld_dy = utils.gradient(nonmnfld_points, nonmnfld_grad[:, :, 1])
            if dims == 3:
                nonmnfld_dz = utils.gradient(nonmnfld_points, nonmnfld_grad[:, :, 2])
                # Sum gradients to approximate divergence in 3D
                nonmnfld_divergence = nonmnfld_dx[:, :, 0] + nonmnfld_dy[:, :, 1] + nonmnfld_dz[:, :, 2]
            else:
                # For 2D points, sum x and y derivatives
                nonmnfld_divergence = nonmnfld_dx[:, :, 0] + nonmnfld_dy[:, :, 1]
            # Handle any NaN values in the computed divergence by setting them to zero
            nonmnfld_divergence[nonmnfld_divergence.isnan()] = 0

            # Select divergence loss based on the specified divergence type
            if self.div_type == 'l2':
                nonmnfld_divergence_term = torch.clamp(torch.square(nonmnfld_divergence), 0.1, self.div_clamp)
            elif self.div_type == 'l1':
                nonmnfld_divergence_term = torch.clamp(torch.abs(nonmnfld_divergence), 0.1, self.div_clamp)
            elif self.div_type == 'gt_l2':
                nonmnfld_divergence_term = torch.square(nonmnfld_divergence - nonmnfld_div_gt) + \
                                           torch.square(mnfld_divergence - mnfld_div_gt)
            elif self.div_type == 'gt_l1':
                nonmnfld_divergence_term = torch.abs(nonmnfld_divergence.abs() - nonmnfld_div_gt.abs()) + \
                                           torch.abs(mnfld_divergence.abs() - mnfld_div_gt.abs())
            else:
                raise Warning("unsupported divergence type. only suuports l1 and l2")

            # Average the divergence term over all points
            div_loss = nonmnfld_divergence_term.mean() #+ mnfld_divergence_term.mean()

        # ---------------------------
        # Compute Eikonal Loss
        # ---------------------------
        # Enforce the gradient norm to be 1 for both manifold and non-manifold predictions
        eikonal_term = eikonal_loss(nonmnfld_grad, mnfld_grad=mnfld_grad, eikonal_type='abs')

        # ---------------------------
        # Compute Latent Regularization Loss
        # ---------------------------
        latent_reg_term = latent_rg_loss(latent_reg, device)
        
        # ---------------------------
        # Compute Normal Loss Term
        # ---------------------------
        if mnfld_n_gt is not None:
            if 'igr' in self.loss_type:
                # For IGR loss, use L2 norm difference between predicted and ground-truth normals
                normal_term = ((mnfld_grad - mnfld_n_gt).abs()).norm(2, dim=1).mean()
            else:
                # Otherwise, use cosine similarity (transformed into a loss)
                normal_term = (1 - torch.abs(torch.nn.functional.cosine_similarity(mnfld_grad, mnfld_n_gt, dim=-1))).mean()

        # ---------------------------
        # Compute Signed Distance Function (SDF) Loss Term
        # ---------------------------
        # Use the absolute value of the manifold prediction as the SDF loss term
        sdf_term = torch.abs(manifold_pred).mean()

        # ---------------------------
        # Compute "Inter" Term
        # ---------------------------
        # Penalizes non-manifold predictions by applying an exponential decay function
        inter_term = torch.exp(-1e2 * torch.abs(non_manifold_pred)).mean()
        # Alternative formulation (commented out):
        # inter_term = torch.exp(-1e0 * torch.abs(non_manifold_pred)).mean()


        #########################################
        # Losses
        #########################################
        # Combine the computed terms into a final loss value based on the chosen loss type.
        if self.loss_type == 'siren': # SIREN loss
            loss = self.weights[0]*sdf_term + self.weights[1] * inter_term + \
                   self.weights[2]*normal_term + self.weights[3]*eikonal_term
        elif self.loss_type == 'siren_wo_n': # SIREN loss without normal constraint
            self.weights[2] = 0
            loss = self.weights[0]*sdf_term + self.weights[1] * inter_term + self.weights[3]*eikonal_term
        elif self.loss_type == 'igr': # IGR loss
            self.weights[1] = 0
            loss = self.weights[0]*sdf_term + self.weights[2]*normal_term + self.weights[3]*eikonal_term
        elif self.loss_type == 'igr_wo_n': # IGR without normals loss
            self.weights[1] = 0
            self.weights[2] = 0
            loss = self.weights[0]*sdf_term + self.weights[3]*eikonal_term
        elif self.loss_type == 'siren_w_div': # SIREN loss with divergence term
            loss = self.weights[0]*sdf_term + self.weights[1] * inter_term + \
                   self.weights[2]*normal_term + self.weights[3]*eikonal_term + \
                   self.weights[4] * div_loss
        elif self.loss_type == 'siren_wo_n_w_div':  # SIREN loss without normals and with divergence constraint
            loss = self.weights[0]*sdf_term + self.weights[1] * inter_term + self.weights[3]*eikonal_term + \
                   self.weights[4] * div_loss
        elif self.loss_type == 'siren_w_curv':
            loss = self.weights[0]*sdf_term + self.weights[1] * inter_term + \
                   self.weights[2]*normal_term + self.weights[3]*eikonal_term + self.weights[4]*curv_term.mean()
        elif self.loss_type == 'siren_w_div_w_curv':
            loss = self.weights[0]*sdf_term + self.weights[1] * inter_term + \
                   self.weights[2]*normal_term + self.weights[3]*eikonal_term + self.weights[4]*curv_term.mean() \
                   + self.weights[4]*div_loss
        elif self.loss_type == 'siren_wo_n_w_div_w_curv':
            loss = self.weights[0]*sdf_term + self.weights[1] * inter_term + self.weights[3]*eikonal_term + \
                   self.weights[4] * (div_loss + curv_term.mean())
        else:
            raise Warning("unrecognized loss type")
        
        # If using multiple surface reconstruction, add latent regularization loss if latent and latent_reg are provided
        if latent is not None and latent_reg is not None:
            loss += self.weights[5] * latent_reg_term
        
        # Return a dictionary of all computed loss components and the manifold gradients for further analysis or debugging
        return {"loss": loss, 'sdf_term': sdf_term, 'inter_term': inter_term, 'latent_reg_term': latent_reg_term,
                'eikonal_term': eikonal_term, 'normals_loss': normal_term, 'div_loss': div_loss, 
                'curv_loss': curv_term.mean()}, mnfld_grad


    def update(self, current_iteration, n_iterations, params=None):
        # `params` should be (start_weight, *optional middle, end_weight) where optional middle is of the form [percent, value]*
        # For example, (1e2, 0.5, 1e2, 0.7, 0.0, 0.0) means:
        #   At 0% of training: weight = 1e2
        #   From 0-50%: weight remains 1e2
        #   From 50-75%: weight decays from 1e2 to 0.0
        #   From 75-100%: weight remains 0.0
        # The weights change as per the div_decay parameter (e.g., linearly, quintic, step, etc.)
        
        # Initialize the decay parameters list if not already set
        if not hasattr(self, 'decay_params_list'):
            assert len(params) >= 2, params
            # TODO: check if this check does anything
            # print(params)
            assert len(params[1:-1]) % 2 == 0
            # Pair weight values with corresponding progress percentages
            self.decay_params_list = list(zip([params[0], *params[1:-1][1::2], params[-1]],
                                               [0, *params[1:-1][::2], 1]))

        # Compute current progress as a fraction (0 to 1)
        curr = current_iteration / n_iterations
        # Determine the next decay point (with progress >= current) and the previous decay point (with progress <= current)
        we, e = min([tup for tup in self.decay_params_list if tup[1] >= curr], key=lambda tup: tup[1])
        w0, s = max([tup for tup in self.decay_params_list if tup[1] <= curr], key=lambda tup: tup[1])

        # Adjust divergence weight according to the selected annealing strategy
        if self.div_decay == 'linear': # Linearly decrease weight from iteration s to e
            if current_iteration < s * n_iterations:
                self.weights[4] = w0
            elif current_iteration >= s * n_iterations and current_iteration < e * n_iterations:
                self.weights[4] = w0 + (we - w0) * (current_iteration / n_iterations - s) / (e - s)
            else:
                self.weights[4] = we
        elif self.div_decay == 'quintic': # Use a quintic function for annealing between s and e
            if current_iteration < s * n_iterations:
                self.weights[4] = w0
            elif current_iteration >= s * n_iterations and current_iteration < e * n_iterations:
                self.weights[4] = w0 + (we - w0) * (1 - (1 - (current_iteration / n_iterations - s) / (e - s)) ** 5)
            else:
                self.weights[4] = we
        elif self.div_decay == 'step': # Change weight abruptly at iteration s
            if current_iteration < s * n_iterations:
                self.weights[4] = w0
            else:
                self.weights[4] = we
        elif self.div_decay == 'none':
            # No decay applied; keep the weight unchanged
            pass
        else:
            raise Warning("unsupported div decay value")
