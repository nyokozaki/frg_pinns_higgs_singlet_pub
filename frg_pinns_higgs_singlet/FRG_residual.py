import torch
import numpy as np
import torch.autograd as autograd

from perturbation.config_params import tau_uv, finite_T
from running_couplings import get_running_couplings

# ============================================================
# FRG residual
# ============================================================

kloop = 1.0 / (12.0 * torch.pi**2)
NGS = 3.0
LPAp = 1.0
#epoch_ratio = 0.0

######################################################################

def get_eta_rho(t):
    with torch.no_grad():
        g1, g2, yt, _, _ = get_running_couplings(t)
    gamma = (1.0 / (16.0 * torch.pi**2)) * (
        3.0 * yt**2
        - 3.0 * g1**2 / 4.0
        - 9.0 * g2**2 / 4.0
    )
    return 2.0 * gamma

def get_eta_sigma(t):
    with torch.no_grad():
        _, _, _, lamS_t, lamHS_t = get_running_couplings(t)
    # gamma_S from PyR@TE3 for this model (matches frg_pinn.tex Sec. 2.3):
    #   gamma_S = (16 pi^2)^-2 (lambda_HS^2 + 3 lambda_S^2),  eta_S = 2 gamma_S.
    # Runs before 2026-09 used (18 lamS^2 + 2 lamHS^2) -- the 1-loop beta_lamS
    # coefficient reused as a proxy; eta_S ~ 1e-4 either way and its effect on the
    # flow is ~1e-5, so existing trained snapshots are unaffected.
    gamma = (1.0 / (16.0 * torch.pi**2))**2 * (3.0 * lamS_t**2 + 1.0 * lamHS_t**2)
    return 2.0 * gamma


def coth(x, eps=1e-10):
    return torch.cosh(x) / (torch.sinh(x) + eps)

def R_FRG_in(model, t_in, rho_in, sigma_in, loop=1.0, epoch_ratio=0.0):

    u, u_rho, u_sigma, u_rhorho, u_sigmasigma, u_rhosigma = model(t_in, rho_in, sigma_in)

    # 時間微分だけは autograd を使用
    u_tin = autograd.grad(
        u, t_in,
        grad_outputs=torch.ones_like(u),
        create_graph=True,
        retain_graph=True,
    )[0]

    A_T = model.A_T
    u_t = A_T * u_tin

    rho_phys = model.unscale_rho(rho_in)
    sigma_phys = model.unscale_sigma(sigma_in)

    t_in_ext = t_in.detach().clone()
    t_phys_ext = model.unscale_t(t_in_ext)

    with torch.no_grad():
        eta_rho = get_eta_rho(t_phys_ext)
        eta_sigma = get_eta_sigma(t_phys_ext)

    rtree = (
        u_t
        + 4.0 * u
        - (2.0 + eta_rho * LPAp) * rho_phys * u_rho
        - (2.0 + eta_sigma * LPAp) * sigma_phys * u_sigma
    )

    eps = 1e-10

    mG2 = u_rho

    M11 = u_rho + 2.0 * rho_phys * u_rhorho
    M22 = u_sigma + 2.0 * sigma_phys * u_sigmasigma
    M12 = 2.0 * torch.sqrt(torch.clamp(rho_phys * sigma_phys, min=eps)) * u_rhosigma

    disk_eps=1e-10
    disc = (M11 - M22)**2 + 4.0 * M12**2
    sqrt_disc = torch.sqrt(torch.clamp(disc, min=disk_eps))

    m1_sq = 0.5 * (M11 + M22 - sqrt_disc)
    m2_sq = 0.5 * (M11 + M22 + sqrt_disc)

    val_mG = 1.0 + mG2
    val_m1 = 1.0 + m1_sq

    neg_mG_mask = val_mG < 0.0
    if torch.any(neg_mG_mask) and epoch_ratio > 0.6:
        idx = torch.nonzero(neg_mG_mask, as_tuple=False)[0]
        i = idx[0]

        bad_t_in = t_in[i, 0].detach().cpu().item()
        bad_rho_in = rho_in[i, 0].detach().cpu().item()
        bad_sigma_in = sigma_in[i, 0].detach().cpu().item()

        bad_t_phys = model.unscale_t(t_in[i:i+1]).detach().cpu().item()
        bad_rho_phys = model.unscale_rho(rho_in[i:i+1]).detach().cpu().item()
        bad_sigma_phys = model.unscale_sigma(sigma_in[i:i+1]).detach().cpu().item()

        bad_val = val_mG[i, 0].detach().cpu().item()
        bad_mG = mG2[i, 0].detach().cpu().item()
        bad_u_rho = u_rho[i, 0].detach().cpu().item()

        msg = (
            "\n[ERROR] 1 + mG2 became negative. Abort.\n"
            f"  scaled   : t_in={bad_t_in:.6f}, rho_in={bad_rho_in:.6f}, sigma_in={bad_sigma_in:.6f}\n"
            f"  physical : t={bad_t_phys:.6f}, rho={bad_rho_phys:.6f}, sigma={bad_sigma_phys:.6f}\n"
            f"  1+mG2    = {bad_val:.12e}  (mG2 = {bad_mG:.12e})\n"
            f"  u_rho    = {bad_u_rho:.12e}\n"
        )
        print(msg)
        raise RuntimeError(msg)

    
    neg_m1_mask = val_m1 < 0.0
    if torch.any(neg_m1_mask) and epoch_ratio > 0.6:
        idx = torch.nonzero(neg_m1_mask, as_tuple=False)[0]
        i = idx[0]

        bad_t_in = t_in[i, 0].detach().cpu().item()
        bad_rho_in = rho_in[i, 0].detach().cpu().item()
        bad_sigma_in = sigma_in[i, 0].detach().cpu().item()

        bad_t_phys = model.unscale_t(t_in[i:i+1]).detach().cpu().item()
        bad_rho_phys = model.unscale_rho(rho_in[i:i+1]).detach().cpu().item()
        bad_sigma_phys = model.unscale_sigma(sigma_in[i:i+1]).detach().cpu().item()

        bad_val = val_m1[i, 0].detach().cpu().item()
        bad_m1 = m1_sq[i, 0].detach().cpu().item()
        bad_M11 = M11[i, 0].detach().cpu().item()
        bad_M22 = M22[i, 0].detach().cpu().item()
        bad_M12 = M12[i, 0].detach().cpu().item()
        bad_disc = disc[i, 0].detach().cpu().item()

        msg = (
            "\n[ERROR] 1 + m1_sq became negative. Abort.\n"
            f"  scaled   : t_in={bad_t_in:.6f}, rho_in={bad_rho_in:.6f}, sigma_in={bad_sigma_in:.6f}\n"
            f"  physical : t={bad_t_phys:.6f}, rho={bad_rho_phys:.6f}, sigma={bad_sigma_phys:.6f}\n"
            f"  1+m1_sq  = {bad_val:.12e}  (m1_sq = {bad_m1:.12e})\n"
            f"  M11      = {bad_M11:.12e}, M22 = {bad_M22:.12e}, M12 = {bad_M12:.12e}\n"
            f"  disc     = {bad_disc:.12e}\n"
        )
        print(msg)
        raise RuntimeError(msg)

    termG_0 = 1.0 / torch.sqrt(torch.clamp(1.0 + mG2, min=eps))
    term1_0 = 1.0 / torch.sqrt(torch.clamp(1.0 + m1_sq, min=eps))
    term2_0 = 1.0 / torch.sqrt(torch.clamp(1.0 + m2_sq, min=eps))

    rho_phys_ext = rho_phys.detach().clone()

    with torch.no_grad():
        g1, g2, yt, _, _ = get_running_couplings(t_phys_ext)
        if finite_T:
            tau_t_ext = tau_uv * torch.exp(-t_phys_ext)

    mW_T2 = 0.5 * g2**2 * rho_phys_ext
    mZ_T2 = 0.5 * (g2**2 + g1**2) * rho_phys_ext
    mt2 = yt**2 * rho_phys_ext

    mW_L2 = mW_T2
    mZ_L2 = mZ_T2
    #mA_L2 = torch.zeros_like(mW_T2)
    
    termWT0 = 1.0 / torch.sqrt(torch.clamp(1.0 + mW_T2, min=eps))
    termWL0 = 1.0 / torch.sqrt(torch.clamp(1.0 + mW_L2, min=eps))
    termZT0 = 1.0 / torch.sqrt(torch.clamp(1.0 + mZ_T2, min=eps))
    termZL0 = 1.0 / torch.sqrt(torch.clamp(1.0 + mZ_L2, min=eps))
    #termAL0 = 1.0 / torch.sqrt(torch.clamp(1.0 + mA_L2, min=eps))
    termT0  = 1.0 / torch.sqrt(torch.clamp(1.0 + mt2, min=eps))

    if finite_T:
        EG = torch.sqrt(torch.clamp(1.0 + mG2, min=eps))
        E1 = torch.sqrt(torch.clamp(1.0 + m1_sq, min=eps))
        E2 = torch.sqrt(torch.clamp(1.0 + m2_sq, min=eps))
        E_WT = torch.sqrt(torch.clamp(1.0 + mW_T2, min=eps))
        E_WL = torch.sqrt(torch.clamp(1.0 + mW_L2, min=eps))
        E_ZT = torch.sqrt(torch.clamp(1.0 + mZ_T2, min=eps))
        E_ZL = torch.sqrt(torch.clamp(1.0 + mZ_L2, min=eps))
        #E_AL = torch.sqrt(torch.clamp(1.0 + mA_L2, min=eps))
        E_T  = torch.sqrt(torch.clamp(1.0 + mt2, min=eps))

        factorG = coth(EG / (2.0 * tau_t_ext), eps=0.0)
        factor1 = coth(E1 / (2.0 * tau_t_ext), eps=0.0)
        factor2 = coth(E2 / (2.0 * tau_t_ext), eps=0.0)
        factor_WT = coth(E_WT / (2.0 * tau_t_ext), eps=0.0)
        factor_WL = coth(E_WL / (2.0 * tau_t_ext), eps=0.0)
        factor_ZT = coth(E_ZT / (2.0 * tau_t_ext), eps=0.0)
        factor_ZL = coth(E_ZL / (2.0 * tau_t_ext), eps=0.0)
        #factor_AL = coth(E_AL / (2.0 * tau_t_ext), eps=0.0)
        factor_T  = torch.tanh(E_T / (2.0 * tau_t_ext))

        termG = termG_0 * factorG
        term1 = term1_0 * factor1
        term2 = term2_0 * factor2
        termWT = termWT0 * factor_WT
        termWL = termWL0 * factor_WL
        termZT = termZT0 * factor_ZT
        termZL = termZL0 * factor_ZL
        #termAL = termAL0 * factor_AL
        termT  = termT0 * factor_T
    else:
        termG, term1, term2 = termG_0, term1_0, term2_0
        termWT, termWL = termWT0, termWL0
        termZT, termZL = termZT0, termZL0
        #termAL = torch.zeros_like(termZL)
        termT  = termT0

    loop_sum = (
        3.0 * termG
        + term1
        + term2
        + 4.0 * termWT
        + 2.0 * termWL
        + 2.0 * termZT
        + 1.0 * termZL
        #+ 1.0 * termAL
        - 12.0 * termT
    )

    rloop = (kloop) * loop_sum
    rloop = torch.clamp(rloop, min=-2.0, max=2.0)

    R = rtree - rloop

    return R
