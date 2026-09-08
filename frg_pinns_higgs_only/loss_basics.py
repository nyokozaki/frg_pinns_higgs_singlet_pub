import torch
import numpy as np
import torch.autograd as autograd

from perturbation.config_params import finite_T
from thermal_functions import u_thermal_finiteT, u_CW_zeroT
from my_networks import u_tree_exact
from FRG_residual import R_FRG_in

import hyperparams as hp
device = hp.device

# ============================================================
# Consistency Loss (Network direct outputs vs Autograd)
# ============================================================

def loss_consistency(model, t_in, rho_in):
    u, ur, urr = model(t_in, rho_in)

    # consistency of the first derivatives
    u_rho_in = autograd.grad(u, rho_in, grad_outputs=torch.ones_like(u), create_graph=True, retain_graph=True)[0]

    u_rho_autograd = model.A_RHO * u_rho_in

    loss_1st = torch.mean((u_rho_autograd - ur)**2)

    # consistency of the second derivatives (differentiate ur once more with autograd)
    ur_rho_in = autograd.grad(ur, rho_in, grad_outputs=torch.ones_like(ur), create_graph=True, retain_graph=True)[0]

    urr_autograd = model.A_RHO * ur_rho_in

    loss_2nd = torch.mean((urr_autograd - urr)**2)

    return loss_1st + loss_2nd



myflag = None

def sample_cheb_like(n, device):
    u = torch.empty(n, device=device).uniform_(0, np.pi)
    return torch.cos(u).unsqueeze(1)

def loss_bc0(model, n_bc):
    t_in = torch.full((n_bc, 1), 1.0, device=device)

    rho_in = (
        torch.empty(n_bc, 1, device=device)
        .uniform_(-1.0, 1.0)
        .requires_grad_(True)
    )

    # get all first/second derivatives predicted directly by the model
    u, du_drho, urr = model(t_in, rho_in)

    t_phys = model.unscale_t(t_in)
    rho_phys = model.unscale_rho(rho_in)

    global myflag
    if myflag is None:
        print("loss_bc0 (finite T) imposed at physical t =", t_phys[0, 0].item())
        myflag = 1.0

    # compute the seed (target) potential
    rho_phys_req = rho_phys.detach().clone().requires_grad_(True)
    t_phys_fixed = t_phys.detach().clone()

    uv_cw = hp.uv_cw

    if finite_T:
        u_seed_req = (u_tree_exact(t_phys_fixed, rho_phys_req)
        + u_thermal_finiteT(t_phys_fixed, rho_phys_req)
        + u_CW_zeroT(t_phys_fixed, rho_phys_req)*uv_cw
        )

    else:
        u_seed_req = (u_tree_exact(t_phys_fixed, rho_phys_req)
                      +u_CW_zeroT(t_phys_fixed, rho_phys_req)*uv_cw
                      )

    # --- compute the first derivatives of the target ---
    (du_seed_drho,) = torch.autograd.grad(
        u_seed_req,
        (rho_phys_req,),
        grad_outputs=torch.ones_like(u_seed_req),
        create_graph=True,
        retain_graph=True,
    )

    # loss of the first derivatives
    loss_rho = ((du_seed_drho.detach() - du_drho) ** 2).mean()

    return loss_rho



def loss_bc(model, model_prev, n_bc):
    rho_in = (
        torch.empty(n_bc, 1, device=device)
        .uniform_(-1.0, 1.0)
        .requires_grad_(True)
    )

    t_cur = torch.full((n_bc, 1), 1.0, device=device)
    t_prev = torch.full((n_bc, 1), -1.0, device=device)

    u_cur, ducur_drho, urr_cur = model(t_cur, rho_in)
    u_prev, duprev_drho, urr_prev = model_prev(t_prev, rho_in)

    loss_rho = ((ducur_drho - duprev_drho)**2).mean()

    loss2nd = ((urr_cur - urr_prev)**2).mean()

    w2nd=hp.w_bc2_diff2

    return loss_rho + w2nd*loss2nd

def loss_pde(model, n_res, loop=1.0, epoch_ratio=0.0):
    n_uni = int(0.4 * n_res)
    n_cheb = n_res - n_uni

    t_uni = torch.empty(n_uni, 1, device=device).uniform_(-1.0, 1.0)
    t_cheb = sample_cheb_like(n_cheb, device)
    t_in = torch.cat([t_uni, t_cheb], dim=0)

    rho_uni = torch.empty(n_uni, 1, device=device).uniform_(-1.0, 1.0)
    rho_cheb = sample_cheb_like(n_cheb, device)
    rho_in = torch.cat([rho_uni, rho_cheb], dim=0)


    t_in.requires_grad_(True)
    rho_in.requires_grad_(True)

    R = R_FRG_in(model, t_in, rho_in, loop, epoch_ratio)
    L_pde_val = torch.mean(R**2)

    # also compute the consistency loss at the PDE evaluation points
    L_consist_val = loss_consistency(model, t_in, rho_in)

    return L_pde_val, L_consist_val


def loss_interface_overlap(model_left, model_right, rho_ov_min, rho_ov_max, t_min, t_max, n_interface=1024):
    rho_phys = torch.empty(n_interface, 1, device=device).uniform_(rho_ov_min, rho_ov_max)
    t_phys = torch.empty(n_interface, 1, device=device).uniform_(t_min, t_max)

    t_left_in = model_left.scale_t(t_phys)
    rho_left_in = model_left.scale_rho(rho_phys)
    _, ur_left, urr_left = model_left(t_left_in, rho_left_in)


    t_right_in = model_right.scale_t(t_phys)
    rho_right_in = model_right.scale_rho(rho_phys)
    _, ur_right, urr_right = model_right(t_right_in, rho_right_in)

    loss_val1 = ((ur_left - ur_right) ** 2).mean()

    return loss_val1
