# Yizhak Ben-Shabat (Itzik) <sitzikbs@gmail.com>
# Chamin Hewa Koneputugodage <chamin.hewa@anu.edu.au>
# This code is the implementation of the DiGS loss functions
# It was partly based on SIREN implementation and architecture but with several significant modifications.
# for the original SIREN version see: https://github.com/vsitzmann/siren

import math

import torch
import torch.nn.functional as F

def eikonal_loss(nonmnfld_grad, mnfld_grad, eikonal_type='abs'):
    """
    Computes the eikonal loss that penalises when ||grad(f)|| != 1 for points on and off the manifold
    Expected input shape: (batch_size, num_points, dim=3) for both gradients
    This loss encourages the gradient norm to be close to 1 (a property of signed distance functions)
    """
    if nonmnfld_grad is not None and mnfld_grad is not None:
        # Concatenate gradients from both non-manifold and manifold points along the points dimension
        all_grads = torch.cat([nonmnfld_grad, mnfld_grad], dim=-2)
    elif nonmnfld_grad is not None:
        all_grads = nonmnfld_grad
    elif mnfld_grad is not None:
        all_grads = mnfld_grad

    # Depending on the eikonal_type, compute the loss using absolute difference or squared difference
    if eikonal_type == 'abs':
        eikonal_term = ((all_grads.norm(2, dim=2) - 1).abs()).mean()
    elif eikonal_type == 'square':
        eikonal_term = ((all_grads.norm(2, dim=2) - 1).square()).mean()
    else:
        raise ValueError(f"Unknown eikonal_type: '{eikonal_type}'. Must be 'abs' or 'square'.")

    return eikonal_term


def latent_rg_loss(latent_reg, device):
    """
    Computes the latent regularization loss for multi-shape learning (e.g., in a VAE context)
    If latent_reg is provided, return its mean; otherwise, return a zero tensor on the correct device.
    """
    if latent_reg is not None:
        reg_loss = latent_reg.mean()
    else:
        reg_loss = torch.tensor([0.0], device=device)

    return reg_loss


def DT(t):
    """
    Compute DT(t) = ((64π - 80)/π^4) t^4 - ((64π - 88)/π^3) t^3
                    + ((16π - 29)/π^2) t^2 + (3/π) t

    This will work if `t` is a float, a NumPy array, or a PyTorch tensor
    """
    # Pre‐compute the four constant coefficients
    c4 = (64 * math.pi - 80) / (math.pi**4)
    c3 = (64 * math.pi - 88) / (math.pi**3)
    c2 = (16 * math.pi - 29) / (math.pi**2)
    c1 = 3.0 / math.pi

    return c4 * t**4 - c3 * t**3 + c2 * t**2 + c1 * t


def gaussian_curvature(nonmnfld_hessian_term, morse_nonmnfld_grad, use_DT = False):
    device = morse_nonmnfld_grad.device
    nonmnfld_hessian_term = torch.cat((nonmnfld_hessian_term, morse_nonmnfld_grad[:, :, :, None]), dim=-1)
    zero_grad = torch.zeros(
        (morse_nonmnfld_grad.shape[0], morse_nonmnfld_grad.shape[1], 1, 1),
        device=device)
    zero_grad = torch.cat((morse_nonmnfld_grad[:, :, None, :], zero_grad), dim=-1)
    nonmnfld_hessian_term = torch.cat((nonmnfld_hessian_term, zero_grad), dim=-2)
    morse_nonmnfld = (-1. / (morse_nonmnfld_grad.norm(dim=-1) ** 2 + 1e-12)) * torch.det(
        nonmnfld_hessian_term)

    morse_nonmnfld = morse_nonmnfld.abs()

    if use_DT:
        morse_nonmnfld = DT(morse_nonmnfld)

    morse_loss = morse_nonmnfld.mean()

    return morse_loss



def offset_loss(model, surface_pts, gradients, delta=0.05,offset_type='abs'):
    """
    Compute ±δ level‐set loss using precomputed gradients.
    """
    n_surf = surface_pts.shape[0]
    grad_surf = gradients[:n_surf]

    normals = grad_surf / (grad_surf.norm(dim=-1, keepdim=True) + 1e-8)

    xp = surface_pts + delta * normals
    xm = surface_pts - delta * normals

    fp = model(xp)
    fm = model(xm)

    if offset_type == 'abs':
        return ((fp - delta).abs() + (fm + delta).abs()).mean()
    elif offset_type == 'square':
        return (((fp - delta).square() + (fm + delta).square())).mean()
    else:
        raise ValueError(f"Unknown offset_type: '{offset_type}'. Must be 'abs' or 'square'.")


def offset_loss_shells(
        model,
        surface_pts: torch.Tensor,
        gradients:   torch.Tensor,
        delta=0.05,
        offset_type: str = "abs",
):
    """
    Offset-shell loss.
    If *delta* is a **scalar**: identical to the old behaviour (single shell).
    If *delta* is a **sequence / tensor**: evaluates a loss for *each* δ and
    returns their arithmetic mean.

    Parameters
    ----------
    model         : Callable; implicit decoder f(x)
    surface_pts   : (N,D) tensor of zero-level samples
    gradients     : (N,D) pre-computed ∇f at surface_pts
    delta         : float or list/tuple/1-D tensor of floats
    offset_type   : 'abs' (L1) or 'square' (L2)

    Returns
    -------
    torch.Scalar  – averaged offset loss over all requested δ shells.
    """

    # hack: overreide the offsets for now
    deltas = [0.005, 0.01, 0.02, 0.04]   # assuming the model operates in a unit-cube scale

    # unify delta to a 1-D tensor on the same device
    if torch.is_tensor(delta):
        deltas = delta.to(surface_pts)
    elif isinstance(delta, (list, tuple)):
        deltas = torch.tensor(delta, device=surface_pts.device, dtype=surface_pts.dtype)
    else:  # single scalar
        deltas = torch.tensor([float(delta)], device=surface_pts.device, dtype=surface_pts.dtype)

    # surface normals (already detached from computation graph)
    normals = gradients / (gradients.norm(dim=-1, keepdim=True) + 1e-8)

    # accumulate per-shell losses
    loss_accum = 0.0
    for d in deltas:
        xp = surface_pts + d * normals
        xm = surface_pts - d * normals

        fp = model(xp)
        fm = model(xm)

        if offset_type == "abs":
            loss_shell = (fp - d).abs() + (fm + d).abs()
        elif offset_type == "square":
            loss_shell = (fp - d).square() + (fm + d).square()
        else:
            raise ValueError("offset_type must be 'abs' or 'square'")

        loss_accum = loss_accum + loss_shell.mean()

    # average across all δ values
    return loss_accum / deltas.numel()


def sample_shell(
        model,
        x_surf: torch.Tensor,      # (B,3) on-surface points (requires_grad=False)
        d_offset: float,             # shell radius ‖d‖ along the normal
        d_direction: str = "both",   # positive, negative, both
        manifold_pred: torch.Tensor = None,
        mnfld_grad: torch.Tensor = None,
        no_offset = False,
):
    # -------------------------------------------------------------------------
    #  sample_shell – draw shell points at a normal offset d
    #  NOTE:  *d_offset*  controls the distance along the normal,
    #         *h_step*    (handled in first_order_morse_loss) controls the
    #                     finite-difference spacing in the tangent plane.
    # -------------------------------------------------------------------------
    x_s = x_surf.detach().requires_grad_(True)

    # ------------- compute unit normals n = ∇f / ‖∇f‖ --------------------
    n = F.normalize(mnfld_grad, dim=-1, eps=1e-9)
    n = n.detach()

    # ------------- random orthonormal tangents u, v ----------------------
    rand = torch.randn_like(n)
    u = F.normalize(rand - (rand * n).sum(-1, keepdim=True) * n, dim=-1)
    v = F.normalize(torch.cross(n, u, dim=-1), dim=-1)

    if no_offset:
        x_w = x_s
    else:
        # ------------- shell point  x + d n  with  d ∈ (0,d_offset] ----------
        if d_direction == "positive":
            d = torch.empty(x_surf.shape[0], 1, device=x_s.device).uniform_(0.0, d_offset)
        elif d_direction == "negative":
            d = -torch.empty(x_surf.shape[0], 1, device=x_s.device).uniform_(0.0, d_offset)
        elif d_direction == "both":
            d = torch.empty(x_surf.shape[0], 1, device=x_s.device).uniform_(-d_offset, d_offset)
        else:
            raise ValueError(f"Invalid d_direction: {d_direction}. Choose from 'positive', 'negative', or 'both'.")
        x_w = x_s + d * n


    return x_w.detach(), u.detach(), v.detach()


def mixed_second_derivative(model, x, u, v):
    """
    x : (M,3) points that already have requires_grad=True
    u,v : (M,3) orthonormal tangent directions
    returns Duv  shape (M,)
    """
    # 1. ∇f
    f = model(x).sum()
    g = torch.autograd.grad(f, x, create_graph=True)[0]

    # 2. Hessian-vector product H·v  (one extra backward pass)
    gv = (g * v).sum(-1)
    Hv = torch.autograd.grad(gv, x, torch.ones_like(gv), retain_graph=True, create_graph=True)[0]

    # 3. uᵀ·(H·v)
    Duv = (Hv * u).sum(-1)
    return Duv


def first_order_morse_loss(
        model,
        x_surf: torch.Tensor,
        d_offset: float = 0.1,         # shell distance  d
        h_step: float = 0.05,           # tangent step   h
        central: bool = False,
        d_direction: str = "both",   # positive, negative, both
        manifold_pred: torch.Tensor = None,
        mnfld_grad: torch.Tensor = None,
        finite_difference: bool = True,
        use_near_points: bool = False
):
# -------------------------------------------------------------------------
#  first_order_morse_loss – mixed-offset Gaussian-curvature 
#
#  Parameters
#  ----------
#  model        : neural SDF  f:ℝ³→ℝ
#  x_surf       : (B,3) on-surface points (requires_grad=False)
#  d_offset     : normal-direction shell radius  d
#  h_step       : tangent-plane step size        h  (if None → h=d_offset
#                                                 for backward compatibility)
#  central: central (±h) stencil if True
# -------------------------------------------------------------------------
#
#            morse_loss = first_order_morse_loss(
#                 model=model,
#                 x_surf=morse_nonmnfld_points,
#                 d_offset=d_offset,
#                 h_step=self.h_offset,
#                 d_direction=self.odw_offset_dir,
#                 central=True,
#                 manifold_pred=manifold_pred,
#                 mnfld_grad=morse_nonmnfld_grad,
#                 finite_difference=self.finite_difference,
#                 use_near_points=self.use_near_points
#             )
    # ---------- sample shell Ω_t (normal offset only uses d_offset) -------
    x_w, u, v = sample_shell(
        model,
        x_surf,
        d_offset,
        d_direction=d_direction,
        manifold_pred=manifold_pred,
        mnfld_grad=mnfld_grad,
        no_offset=use_near_points
    )

    if finite_difference:
        def f(pts: torch.Tensor):
            return model(pts).view(-1)
        # ---------- one-sided mixed stencil using step size h -----------------
        f00 = f(x_w)
        fu  = f(x_w + h_step * u)
        fv  = f(x_w + h_step * v)
        fuv = f(x_w + h_step * (u + v))

        Duv = (fuv - fu - fv + f00) / (h_step * h_step)
        loss = Duv.abs()

        # ---------- optional central (±h) correction --------------------------
        if F:
            fu_m  = f(x_w - h_step * u)
            fv_m  = f(x_w - h_step * v)
            fuv_m = f(x_w - h_step * (u + v))
            Duv_m = (fuv_m - fu_m - fv_m + f00) / (h_step * h_step)
            loss = 0.5 * (Duv + Duv_m).abs()
    else:
        with torch.enable_grad():
            x_w = x_w.detach().requires_grad_(True)  # keep the graph
            Duv = mixed_second_derivative(model, x_w, u, v)
            loss = Duv.abs()

    return loss.mean()


def get_fermi(model, points, grad, offset_d):
    x=points.detach().requires_grad_(True)
    n = F.normalize(grad, dim=-1, eps=1e-9)
    n= n.detach()
    rand = torch.randn_like(n)

    x_u = F.normalize(rand - (rand * n).sum(-1, keepdim=True) * n, dim=-1)
    x_v=F.normalize(torch.cross(n, x_u, dim=-1), dim=-1)

    x_w=x
    # return near points and a perpendicular coordinate on the point
    return x_u.detach(),x_v.detach(),x_w.detach()


def finite_neurcad_loss(x,y,u,v,h_step,model,use_denominator,grad):

    def f(pts: torch.Tensor):
        return model(pts).view(-1)

    f0 = f(x)

    f_uu = (f(x + h_step * u) + f(x - h_step * u) - 2 * f0) / (h_step ** 2)
    f_vv = (f(x + h_step * v) + f(x - h_step * v) - 2 * f0) / (h_step ** 2)
    f_uv = (f(x + h_step * (u + v))
            - f(x + h_step * (u - v))
            - f(x + h_step * (-u + v))
            + f(x - h_step * (u + v))) / (4 * h_step ** 2)

    det_hessian = f_uu * f_vv - f_uv ** 2

    if use_denominator:
        grad_norm = torch.norm(grad, dim=-1)
        denominator = grad_norm ** 4 + 1e-8
        curvature = det_hessian / denominator
    else:
        curvature = det_hessian

    Duv=curvature

    return Duv.abs()
