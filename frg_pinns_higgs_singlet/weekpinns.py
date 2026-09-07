import torch
import numpy as np
import torch.autograd as autograd

from FRG_residual import R_FRG_in, get_eta_rho, get_eta_sigma
# ============================================================
# Weak-PINNs (VPINNs) helpers
# ============================================================

def eval_legendre(n, x):
    if n == 0:
        return torch.ones_like(x)
    elif n == 1:
        return x
    p0 = torch.ones_like(x)
    p1 = x
    for i in range(1, n):
        p2 = ((2.0 * i + 1.0) * x * p1 - i * p0) / (i + 1.0)
        p0 = p1
        p1 = p2
    return p1

def setup_weak_pinn(deg, t_min, t_max, rho_min, rho_max, sigma_min, sigma_max, configs, device):
    t_np, w_np = np.polynomial.legendre.leggauss(deg)
    t_tilde = torch.tensor(t_np, dtype=torch.float32, device=device)
    w_tilde = torch.tensor(w_np, dtype=torch.float32, device=device)

    T_tilde, Rho_tilde, Sig_tilde = torch.meshgrid(t_tilde, t_tilde, t_tilde, indexing='ij')
    Wt, Wrho, Wsig = torch.meshgrid(w_tilde, w_tilde, w_tilde, indexing='ij')

    J_t = (t_max - t_min) / 2.0
    J_rho = (rho_max - rho_min) / 2.0
    J_sig = (sigma_max - sigma_min) / 2.0
    W_3d = Wt * Wrho * Wsig * J_t * J_rho * J_sig

    t_phys = (0.5 * (T_tilde + 1.0) * (t_max - t_min) + t_min).requires_grad_(True)
    rho_phys = (0.5 * (Rho_tilde + 1.0) * (rho_max - rho_min) + rho_min).requires_grad_(True)
    sigma_phys = (0.5 * (Sig_tilde + 1.0) * (sigma_max - sigma_min) + sigma_min).requires_grad_(True)

    T_req = 2.0 * (t_phys - t_min) / (t_max - t_min) - 1.0
    R_req = 2.0 * (rho_phys - rho_min) / (rho_max - rho_min) - 1.0
    S_req = 2.0 * (sigma_phys - sigma_min) / (sigma_max - sigma_min) - 1.0

    env_3d = (1.0 - T_req**2) * (1.0 - R_req**2) * (1.0 - S_req**2)

    Fa_list, dFa_dt_list, dFa_drho_list, dFa_dsigma_list = [], [], [], []
    
    for (nt, nr, ns) in configs:
        Pt = eval_legendre(nt, T_req)
        Pr = eval_legendre(nr, R_req)
        Ps = eval_legendre(ns, S_req)
        F_a = env_3d * Pt * Pr * Ps

        dFa_dt = torch.autograd.grad(F_a.sum(), t_phys, create_graph=False, retain_graph=True)[0]
        dFa_drho = torch.autograd.grad(F_a.sum(), rho_phys, create_graph=False, retain_graph=True)[0]
        dFa_dsigma = torch.autograd.grad(F_a.sum(), sigma_phys, create_graph=False, retain_graph=True)[0]

        Fa_list.append(F_a.detach().reshape(-1, 1))
        dFa_dt_list.append(dFa_dt.detach().reshape(-1, 1))
        dFa_drho_list.append(dFa_drho.detach().reshape(-1, 1))
        dFa_dsigma_list.append(dFa_dsigma.detach().reshape(-1, 1))

    t_in = T_tilde.reshape(-1, 1)
    rho_in = Rho_tilde.reshape(-1, 1)
    sigma_in = Sig_tilde.reshape(-1, 1)
    W_3d = W_3d.reshape(-1, 1)

    return t_in, rho_in, sigma_in, W_3d, Fa_list, dFa_dt_list, dFa_drho_list, dFa_dsigma_list

def loss_weak(model, t_in_w, rho_in_w, sigma_in_w, W_3d, Fa_list, dFa_dt_list, dFa_drho_list, dFa_dsigma_list, loop=1.0, epoch_ratio=0.0):
    t_in = t_in_w.detach().clone().requires_grad_(True)
    rho_in = rho_in_w.detach().clone().requires_grad_(True)
    sigma_in = sigma_in_w.detach().clone().requires_grad_(True)

    R = R_FRG_in(model, t_in, rho_in, sigma_in, loop, epoch_ratio)

    u, u_rho, u_sigma, _, _, _ = model(t_in, rho_in, sigma_in)
    u_tin = torch.autograd.grad(u, t_in, grad_outputs=torch.ones_like(u), create_graph=True)[0]

    u_t = model.A_T * u_tin

    t_phys = model.unscale_t(t_in)
    rho_phys = model.unscale_rho(rho_in)
    sigma_phys = model.unscale_sigma(sigma_in)

    with torch.no_grad():
        eta_rho = get_eta_rho(t_phys)
        eta_sigma = get_eta_sigma(t_phys)

    LPAp_val = 1.0 

    LHS_NN = (
        u_t
        + 4.0 * u
        - (2.0 + 0.5 * eta_rho * LPAp_val) * rho_phys * u_rho
        - (2.0 + 0.5 * eta_sigma * LPAp_val) * sigma_phys * u_sigma
    )

    RHS_eval = LHS_NN - R

    loss_w = 0.0
    for Fa, dFa_dt, dFa_drho, dFa_dsigma in zip(Fa_list, dFa_dt_list, dFa_drho_list, dFa_dsigma_list):
        
        LHS_IBP_integrand = u * (
            - dFa_dt
            + 4.0 * Fa
            + (2.0 + 0.5 * eta_rho * LPAp_val) * (Fa + rho_phys * dFa_drho)
            + (2.0 + 0.5 * eta_sigma * LPAp_val) * (Fa + sigma_phys * dFa_dsigma)
        )

        weak_residual = torch.sum((LHS_IBP_integrand - RHS_eval * Fa) * W_3d)
        
        loss_w += weak_residual**2

    return loss_w
