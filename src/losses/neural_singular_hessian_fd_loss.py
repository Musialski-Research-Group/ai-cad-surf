import torch
import torch.nn as nn
import torch.nn.functional as F

import utils
from .term import eikonal_loss, latent_rg_loss


class NeuralSingularHessianLossFD(nn.Module):
    def __init__(
            self,
            config,
            model=None
    ):
        super().__init__()
        if config.weights is None:
            config.weights = [3e3, 1e2, 1e2, 5e1, 1e2, 1e1]
        self.weights = config.weights
        self.loss_type = config.type
        self.div_decay = config.div_decay
        self.div_type = config.div_type
        self.use_morse = True if config.weight_for_morse else False
        self.bidirectional_morse = config.bidirectional_morse

        self.finite_difference = config.finite_difference

        #self.finite_difference = getattr(config, 'finite_difference', True)
        self.h_step = getattr(config, 'h_step', 0.05)
        self.model = model

    def finite_hessian_det(self, model, points, grads):
        """
        使用有限差分计算Hessian矩阵的行列式
        参数:
            points: (B, N, 3) 输入点
            grads: (B, N, 3) 对应点的梯度
        返回:
            det: (B, N) Hessian行列式值
        """
        B, N, _ = points.shape
        device = points.device

        # 1. 计算单位法向量
        n = F.normalize(grads, dim=-1, eps=1e-9)  # (B, N, 3)

        # 2. 生成随机正交切向量
        rand = torch.randn_like(n)
        u = F.normalize(rand - torch.sum(rand * n, dim=-1, keepdim=True) * n, dim=-1)  # (B, N, 3)
        v = F.normalize(torch.cross(n, u, dim=-1), dim=-1)  # (B, N, 3)

        h = self.h_step

        # 3. 构建采样点 (9个点/输入点)
        # 基础点
        base_points = points.repeat(1, 9, 1)  # (B, 9*N, 3)

        # 创建偏移向量
        offsets = torch.zeros((9, 3), device=device)
        # 中心点 (0偏移) - 索引0
        # +h*u - 索引1
        offsets[1] = torch.tensor([1, 0, 0])
        # -h*u - 索引2
        offsets[2] = torch.tensor([-1, 0, 0])
        # +h*v - 索引3
        offsets[3] = torch.tensor([0, 1, 0])
        # -h*v - 索引4
        offsets[4] = torch.tensor([0, -1, 0])
        # +h*u+h*v - 索引5
        offsets[5] = torch.tensor([1, 1, 0])
        # +h*u-h*v - 索引6
        offsets[6] = torch.tensor([1, -1, 0])
        # -h*u+h*v - 索引7
        offsets[7] = torch.tensor([-1, 1, 0])
        # -h*u-h*v - 索引8
        offsets[8] = torch.tensor([-1, -1, 0])

        # 扩展偏移以匹配输入维度
        offsets = offsets.view(1, 9, 1, 3)  # (1, 9, 1, 3)
        u_exp = u.unsqueeze(1)  # (B, 1, N, 3)
        v_exp = v.unsqueeze(1)  # (B, 1, N, 3)

        # 计算实际偏移点
        offset_vectors = h * (offsets[..., 0:1] * u_exp +
                              offsets[..., 1:2] * v_exp)  # (B, 9, N, 3)

        # 重塑为(B, 9*N, 3)
        offset_vectors = offset_vectors.view(B, 9 * N, 3)
        sample_points = base_points + offset_vectors
        sample_points = sample_points.squeeze(0)
        # 4. 计算所有采样点的SDF值
        #with torch.no_grad():
        sdf_vals = model(sample_points).view(B, 9, N)  # (B, 9, N)

        # 5. 提取各位置的SDF值
        f00 = sdf_vals[:, 0]  # 中心点
        fu = sdf_vals[:, 1]  # +h*u
        fu_m = sdf_vals[:, 2]  # -h*u
        fv = sdf_vals[:, 3]  # +h*v
        fv_m = sdf_vals[:, 4]  # -h*v
        fuv = sdf_vals[:, 5]  # +h*u+h*v
        fu_vm = sdf_vals[:, 6]  # +h*u-h*v
        fum_v = sdf_vals[:, 7]  # -h*u+h*v
        fum_vm = sdf_vals[:, 8]  # -h*u-h*v

        # 6. 计算二阶导数近似
        h2 = h * h
        H_uu = (fu + fu_m - 2 * f00) / h2
        H_vv = (fv + fv_m - 2 * f00) / h2
        H_uv = (fuv - fu_vm - fum_v + fum_vm) / (4 * h2)

        # 7. 计算Hessian行列式
        det = H_uu * H_vv - H_uv * H_uv
        return det

    def forward(self, output_pred: dict, gt: dict,**kwargs):
        model = kwargs.get('model', None)

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

        div_loss = torch.tensor([0.0], device=device)
        morse_loss = torch.tensor([0.0], device=device)
        curv_term = torch.tensor([0.0], device=device)
        latent_reg_term = torch.tensor([0.0], device=device)
        normal_term = torch.tensor([0.0], device=device)

        # compute gradients
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
            # 使用有限差分方法计算Hessian行列式
            if self.finite_difference: #and self.model is not None:    #if self.finite_difference and self.model is not None:
                # 非流形点上的有限差分
                nonmnfld_det = self.finite_hessian_det(
                    model,
                    morse_nonmnfld_points,
                    morse_nonmnfld_grad
                ).squeeze(0)

                #print('nonmnfld_det',nonmnfld_det)

                morse_nonmnfld = torch.tensor(0.0, device=device)
                if self.div_type == 'l2':
                    morse_nonmnfld = nonmnfld_det.square().mean()
                elif self.div_type == 'l1':
                    morse_nonmnfld = nonmnfld_det.abs().mean()

                # 流形点上的有限差分（如果启用双向）
                morse_mnfld = torch.tensor(0.0, device=device)
                # if self.bidirectional_morse and mnfld_grad is not None:    #bidirectional_morse default False
                #     mnfld_det = self.finite_hessian_det(
                #         model,
                #         mnfld_points,
                #         mnfld_grad
                #     ).squeeze(0)
                #
                #     if self.div_type == 'l2':
                #         morse_mnfld = mnfld_det.square().mean()
                #     elif self.div_type == 'l1':
                #         morse_mnfld = mnfld_det.abs().mean()

                morse_loss = 0.5 * (morse_nonmnfld + morse_mnfld)

            else:
                # 原始自动微分方法
                nonmnfld_dx = utils.gradient(morse_nonmnfld_points, morse_nonmnfld_grad[:, :, 0])
                nonmnfld_dy = utils.gradient(morse_nonmnfld_points, morse_nonmnfld_grad[:, :, 1])
                if dims == 3:
                    nonmnfld_dz = utils.gradient(morse_nonmnfld_points, morse_nonmnfld_grad[:, :, 2])
                    nonmnfld_hessian_term = torch.stack((nonmnfld_dx, nonmnfld_dy, nonmnfld_dz), dim=-1)
                    #mnfld_dz = utils.gradient(mnfld_points, mnfld_grad[:, :, 2])
                    #mnfld_hessian_term = torch.stack((mnfld_dx, mnfld_dy, mnfld_dz), dim=-1)
                else:
                    nonmnfld_hessian_term = torch.stack((nonmnfld_dx, nonmnfld_dy), dim=-1)

                nonmnfld_det = torch.det(nonmnfld_hessian_term)

                morse_mnfld = torch.tensor([0.0], device=device)
                morse_nonmnfld = torch.tensor([0.0], device=device)

                #print('nonmnfld_det_analytical', nonmnfld_det)
                if self.div_type == 'l2':
                    morse_nonmnfld = nonmnfld_det.square().mean()
                elif self.div_type == 'l1':
                    morse_nonmnfld = nonmnfld_det.abs().mean()

                morse_loss = 0.5 * (morse_nonmnfld + morse_mnfld)

        # 其余部分保持不变...
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

        # eikonal term
        eikonal_term = eikonal_loss(None, mnfld_grad=mnfld_grad, eikonal_type='abs')

        # inter term
        inter_term = torch.exp(-1e2 * torch.abs(non_manifold_pred)).mean()

        # 损失组合部分保持不变...
        # ... [原有的损失组合代码] ...

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
