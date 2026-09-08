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

def loss_consistency(model, t_in, rho_in, sigma_in):
    u, ur, us, urr, uss, urs = model(t_in, rho_in, sigma_in)

    ws1 = hp.ws1
    ws2 = hp.ws2
    wrsmix = hp.wrsmix

    # consistency of the first derivatives
    u_rho_in = autograd.grad(u, rho_in, grad_outputs=torch.ones_like(u), create_graph=True, retain_graph=True)[0]
    u_sigma_in = autograd.grad(u, sigma_in, grad_outputs=torch.ones_like(u), create_graph=True, retain_graph=True)[0]
    
    u_rho_autograd = model.A_RHO * u_rho_in
    u_sigma_autograd = model.A_SIGMA * u_sigma_in
    
    loss_1st = torch.mean((u_rho_autograd - ur)**2) + ws1*torch.mean((u_sigma_autograd - us)**2)

    # consistency of the second derivatives (differentiate ur, us once more with autograd)
    ur_rho_in = autograd.grad(ur, rho_in, grad_outputs=torch.ones_like(ur), create_graph=True, retain_graph=True)[0]
    us_sigma_in = autograd.grad(us, sigma_in, grad_outputs=torch.ones_like(us), create_graph=True, retain_graph=True)[0]
    us_rho_in = autograd.grad(us, rho_in, grad_outputs=torch.ones_like(us), create_graph=True, retain_graph=True)[0]
    ur_sigma_in = autograd.grad(ur, sigma_in, grad_outputs=torch.ones_like(ur), create_graph=True, retain_graph=True)[0]
    
    
    urr_autograd = model.A_RHO * ur_rho_in
    uss_autograd = model.A_SIGMA * us_sigma_in
    urs_autograd = model.A_RHO * us_rho_in
    usr_autograd = model.A_SIGMA * ur_sigma_in
    
    loss_2nd = (torch.mean((urr_autograd - urr)**2) 
    + ws2*torch.mean((uss_autograd - uss)**2) 
    + wrsmix*torch.mean((urs_autograd - urs)**2)
    + wrsmix*torch.mean((usr_autograd - urs)**2))

    return loss_1st + loss_2nd



myflag = None

'''
def sample_cheb_like(n, device):
    N = 200
    k = torch.arange(N + 1, device=device)
    cheb_nodes = torch.cos(np.pi * k / N)
    idx = torch.randint(0, N + 1, (n,), device=device)
    return cheb_nodes[idx].unsqueeze(1)
'''

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
    sigma_in = (
        torch.empty(n_bc, 1, device=device)
        .uniform_(-1.0, 1.0)
        .requires_grad_(True)
    )

    '''
    n_uni = int(0.5 * n_bc)
    n_cheb = n_bc - n_uni

    rho_uni = torch.empty(n_uni, 1, device=device).uniform_(-1.0, 1.0)
    rho_cheb = sample_cheb_like(n_cheb, device)
    rho_in = torch.cat([rho_uni, rho_cheb], dim=0)

    sigma_uni = torch.empty(n_uni, 1, device=device).uniform_(-1.0, 1.0)
    sigma_cheb = sample_cheb_like(n_cheb, device)
    sigma_in = torch.cat([sigma_uni, sigma_cheb], dim=0)
    '''

    # get all first/second derivatives predicted directly by the model
    u, du_drho, du_dsigma, urr, uss, urs = model(t_in, rho_in, sigma_in)

    t_phys = model.unscale_t(t_in)
    rho_phys = model.unscale_rho(rho_in)
    sigma_phys = model.unscale_sigma(sigma_in)

    global myflag
    if myflag is None:
        print("loss_bc0 (finite T) imposed at physical t =", t_phys[0, 0].item())
        myflag = 1.0

    # compute the seed (target) potential
    rho_phys_req = rho_phys.detach().clone().requires_grad_(True)
    sigma_phys_req = sigma_phys.detach().clone().requires_grad_(True)
    t_phys_fixed = t_phys.detach().clone()

    uv_cw = hp.uv_cw

    if finite_T:
        u_seed_req = (u_tree_exact(t_phys_fixed, rho_phys_req, sigma_phys_req) 
        + u_thermal_finiteT(t_phys_fixed, rho_phys_req, sigma_phys_req)
        + u_CW_zeroT(t_phys_fixed, rho_phys_req, sigma_phys_req)*uv_cw
        )
        
    else:
        u_seed_req = (u_tree_exact(t_phys_fixed, rho_phys_req, sigma_phys_req)
                      +u_CW_zeroT(t_phys_fixed, rho_phys_req, sigma_phys_req)*uv_cw
                      )
        
    # --- compute the first derivatives of the target ---
    du_seed_drho, du_seed_dsigma = torch.autograd.grad(
        u_seed_req,
        (rho_phys_req, sigma_phys_req),
        grad_outputs=torch.ones_like(u_seed_req),
        create_graph=True,  # set True in order to compute second derivatives
        retain_graph=True,
    )

    # loss of the first derivatives
    loss_rho = ((du_seed_drho.detach() - du_drho) ** 2).mean()
    loss_sigma = ((du_seed_dsigma.detach() - du_dsigma) ** 2).mean()


    '''
    # --- compute the second derivatives of the target ---
    du_seed_drhorho = torch.autograd.grad(
        du_seed_drho, rho_phys_req,
        grad_outputs=torch.ones_like(du_seed_drho),
        create_graph=False, retain_graph=True
    )[0]

    du_seed_dsigmasigma = torch.autograd.grad(
        du_seed_dsigma, sigma_phys_req,
        grad_outputs=torch.ones_like(du_seed_dsigma),
        create_graph=False, retain_graph=True
    )[0]

    du_seed_drhosigma = torch.autograd.grad(
        du_seed_drho, sigma_phys_req,
        grad_outputs=torch.ones_like(du_seed_drho),
        create_graph=False, retain_graph=False
    )[0]

    # ==========================================
    # compute the soft mask (restricted to the region rho_phys_req, sigma_phys_req > x0)
    # ==========================================
    mask_temp = 100.
    origin_cut = 0.1
    mask_rho = torch.sigmoid(mask_temp * (rho_phys_req - origin_cut))
    mask_sigma = torch.sigmoid(mask_temp * (sigma_phys_req - origin_cut))
    soft_mask = mask_rho * mask_sigma

    def masked_mean(loss_tensor):
        """Weighted average by the mask; evaluate the loss only over the region of valid samples."""
        eps = 1e-12
        return (loss_tensor * soft_mask).sum() / (soft_mask.sum() + eps)


    # loss of the second derivatives (automatically includes the finite-temperature thermal-loop correction)
    loss_urr = masked_mean((du_seed_drhorho.detach() - urr)**2)
    loss_uss = masked_mean((du_seed_dsigmasigma.detach() - uss)**2)
    loss_urs = masked_mean((du_seed_drhosigma.detach() - urs)**2)

    #print(du_seed_drhorho[0][0])

    w2nd=hp.w_bc1_diff2
    return loss_rho + loss_sigma + w2nd*(loss_urr + loss_uss + loss_urs)
    '''

    return loss_rho + loss_sigma



def loss_bc(model, model_prev, n_bc):
    '''
    n_uni = int(0.5 * n_bc)
    n_cheb = n_bc - n_uni

    rho_uni = torch.empty(n_uni, 1, device=device).uniform_(-1.0, 1.0)
    rho_cheb = sample_cheb_like(n_cheb, device)
    rho_in = torch.cat([rho_uni, rho_cheb], dim=0)

    sigma_uni = torch.empty(n_uni, 1, device=device).uniform_(-1.0, 1.0)
    sigma_cheb = sample_cheb_like(n_cheb, device)
    sigma_in = torch.cat([sigma_uni, sigma_cheb], dim=0)
    '''
    rho_in = (
        torch.empty(n_bc, 1, device=device)
        .uniform_(-1.0, 1.0)
        .requires_grad_(True)
    )
    sigma_in = (
        torch.empty(n_bc, 1, device=device)
        .uniform_(-1.0, 1.0)
        .requires_grad_(True)
    )

    t_cur = torch.full((n_bc, 1), 1.0, device=device)
    t_prev = torch.full((n_bc, 1), -1.0, device=device)

    u_cur, ducur_drho, ducur_dsigma, urr_cur, uss_cur, urs_cur = model(t_cur, rho_in, sigma_in)
    u_prev, duprev_drho, duprev_dsigma, urr_prev, uss_prev, urs_prev = model_prev(t_prev, rho_in, sigma_in)

    loss_rho = ((ducur_drho - duprev_drho)**2).mean()
    loss_sigma = ((ducur_dsigma - duprev_dsigma)**2).mean()

    loss2nd = ((urr_cur - urr_prev)**2 + (uss_cur - uss_prev)**2 + (urs_cur - urs_prev)**2).mean()

    w2nd=hp.w_bc2_diff2

    return loss_rho + loss_sigma + w2nd*loss2nd

def loss_pde(model, n_res, loop=1.0, epoch_ratio=0.0):
    n_uni = int(0.4 * n_res)
    n_cheb = n_res - n_uni

    rho_in = torch.empty(n_res, 1, device=device).uniform_(-1.0, 1.0)
    sigma_in = torch.empty(n_res, 1, device=device).uniform_(-1.0, 1.0)

    t_uni = torch.empty(n_uni, 1, device=device).uniform_(-1.0, 1.0)
    t_cheb = sample_cheb_like(n_cheb, device)
    t_in = torch.cat([t_uni, t_cheb], dim=0)

    rho_uni = torch.empty(n_uni, 1, device=device).uniform_(-1.0, 1.0)
    rho_cheb = sample_cheb_like(n_cheb, device)
    rho_in = torch.cat([rho_uni, rho_cheb], dim=0)

    sigma_uni = torch.empty(n_uni, 1, device=device).uniform_(-1.0, 1.0)
    sigma_cheb = sample_cheb_like(n_cheb, device)
    sigma_in = torch.cat([sigma_uni, sigma_cheb], dim=0)


    t_in.requires_grad_(True)
    rho_in.requires_grad_(True)
    sigma_in.requires_grad_(True)
    
    R = R_FRG_in(model, t_in, rho_in, sigma_in, loop, epoch_ratio)
    L_pde_val = torch.mean(R**2)
    
    '''
    rho_edge_in = torch.full_like(t_in, -1.0, requires_grad=True)
    sigma_edge_in = torch.full_like(t_in, -1.0, requires_grad=True)

    R_edge = R_FRG_in(model, t_in, rho_edge_in, sigma_edge_in, loop)

    R_diff = R - R_edge.detach()
    L_pde_val = torch.mean(R_diff**2)
    '''
    
    # also compute the consistency loss at the PDE evaluation points
    L_consist_val = loss_consistency(model, t_in, rho_in, sigma_in)
    
    return L_pde_val, L_consist_val


def loss_interface_overlap(model_left, model_right, rho_ov_min, rho_ov_max, t_min, t_max, sigma_min, sigma_max, n_interface=1024):
    rho_phys = torch.empty(n_interface, 1, device=device).uniform_(rho_ov_min, rho_ov_max)
    t_phys = torch.empty(n_interface, 1, device=device).uniform_(t_min, t_max)
    sigma_phys = torch.empty(n_interface, 1, device=device).uniform_(sigma_min, sigma_max)

    t_left_in = model_left.scale_t(t_phys)
    rho_left_in = model_left.scale_rho(rho_phys)
    sigma_left_in = model_left.scale_sigma(sigma_phys)
    _, ur_left, us_left, urr_left, uss_left, urs_left = model_left(t_left_in, rho_left_in, sigma_left_in)


    t_right_in = model_right.scale_t(t_phys)
    rho_right_in = model_right.scale_rho(rho_phys)
    sigma_right_in = model_right.scale_sigma(sigma_phys)    
    _, ur_right, us_right, urr_right, uss_right, urs_right = model_right(t_right_in, rho_right_in, sigma_right_in)

    loss_val1 = ((ur_left - ur_right) ** 2).mean()
    loss_val2 = ((us_left - us_right) ** 2).mean()

    '''
    w_mod = 1e-3
    loss_val3 = ((urr_left - urr_right) ** 2).mean()
    loss_val4 = ((uss_left - uss_right) ** 2).mean()
    loss_val5 = ((urs_left - urs_right) ** 2).mean()

    return loss_val1 + loss_val2 + (loss_val3 + loss_val4 + w_mod*loss_val5)
    '''
    return loss_val1 + loss_val2
