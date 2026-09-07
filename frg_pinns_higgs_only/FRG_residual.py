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
        g1, g2, yt = get_running_couplings(t)
    gamma = (1.0 / (16.0 * torch.pi**2)) * (
        3.0 * yt**2
        - 3.0 * g1**2 / 4.0
        - 9.0 * g2**2 / 4.0
    )
    return 2.0 * gamma


def coth(x, eps=1e-10):
    return torch.cosh(x) / (torch.sinh(x) + eps)

def R_FRG_in(model, t_in, rho_in, loop=1.0, epoch_ratio=0.0):

    u, u_rho, u_rhorho = model(t_in, rho_in)

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

    t_in_ext = t_in.detach().clone()
    t_phys_ext = model.unscale_t(t_in_ext)

    with torch.no_grad():
        eta_rho = get_eta_rho(t_phys_ext)

    rtree = (
        u_t
        + 4.0 * u
        - (2.0 + eta_rho * LPAp) * rho_phys * u_rho
    )

    eps = 1e-10

    mG2 = u_rho
    mH2 = u_rho + 2.0 * rho_phys * u_rhorho

    val_mG = 1.0 + mG2
    val_mH = 1.0 + mH2

    neg_mG_mask = val_mG < 0.0
    if torch.any(neg_mG_mask) and epoch_ratio > 0.6:
        idx = torch.nonzero(neg_mG_mask, as_tuple=False)[0]
        i = idx[0]

        bad_t_in = t_in[i, 0].detach().cpu().item()
        bad_rho_in = rho_in[i, 0].detach().cpu().item()

        bad_t_phys = model.unscale_t(t_in[i:i+1]).detach().cpu().item()
        bad_rho_phys = model.unscale_rho(rho_in[i:i+1]).detach().cpu().item()

        bad_val = val_mG[i, 0].detach().cpu().item()
        bad_mG = mG2[i, 0].detach().cpu().item()
        bad_u_rho = u_rho[i, 0].detach().cpu().item()

        msg = (
            "\n[ERROR] 1 + mG2 became negative. Abort.\n"
            f"  scaled   : t_in={bad_t_in:.6f}, rho_in={bad_rho_in:.6f}\n"
            f"  physical : t={bad_t_phys:.6f}, rho={bad_rho_phys:.6f}\n"
            f"  1+mG2    = {bad_val:.12e}  (mG2 = {bad_mG:.12e})\n"
            f"  u_rho    = {bad_u_rho:.12e}\n"
        )
        print(msg)
        raise RuntimeError(msg)


    neg_mH_mask = val_mH < 0.0
    if torch.any(neg_mH_mask) and epoch_ratio > 0.6:
        idx = torch.nonzero(neg_mH_mask, as_tuple=False)[0]
        i = idx[0]

        bad_t_in = t_in[i, 0].detach().cpu().item()
        bad_rho_in = rho_in[i, 0].detach().cpu().item()

        bad_t_phys = model.unscale_t(t_in[i:i+1]).detach().cpu().item()
        bad_rho_phys = model.unscale_rho(rho_in[i:i+1]).detach().cpu().item()

        bad_val = val_mH[i, 0].detach().cpu().item()
        bad_mH = mH2[i, 0].detach().cpu().item()

        msg = (
            "\n[ERROR] 1 + mH2 became negative. Abort.\n"
            f"  scaled   : t_in={bad_t_in:.6f}, rho_in={bad_rho_in:.6f}\n"
            f"  physical : t={bad_t_phys:.6f}, rho={bad_rho_phys:.6f}\n"
            f"  1+mH2    = {bad_val:.12e}  (mH2 = {bad_mH:.12e})\n"
        )
        print(msg)
        raise RuntimeError(msg)

    termG_0 = 1.0 / torch.sqrt(torch.clamp(1.0 + mG2, min=eps))
    termH_0 = 1.0 / torch.sqrt(torch.clamp(1.0 + mH2, min=eps))

    rho_phys_ext = rho_phys.detach().clone()

    with torch.no_grad():
        g1, g2, yt = get_running_couplings(t_phys_ext)
        if finite_T:
            tau_t_ext = tau_uv * torch.exp(-t_phys_ext)

    mW_T2 = 0.5 * g2**2 * rho_phys_ext
    mZ_T2 = 0.5 * (g2**2 + g1**2) * rho_phys_ext
    mt2 = yt**2 * rho_phys_ext

    mW_L2 = mW_T2
    mZ_L2 = mZ_T2

    termWT0 = 1.0 / torch.sqrt(torch.clamp(1.0 + mW_T2, min=eps))
    termWL0 = 1.0 / torch.sqrt(torch.clamp(1.0 + mW_L2, min=eps))
    termZT0 = 1.0 / torch.sqrt(torch.clamp(1.0 + mZ_T2, min=eps))
    termZL0 = 1.0 / torch.sqrt(torch.clamp(1.0 + mZ_L2, min=eps))
    termT0  = 1.0 / torch.sqrt(torch.clamp(1.0 + mt2, min=eps))

    if finite_T:
        EG = torch.sqrt(torch.clamp(1.0 + mG2, min=eps))
        EH = torch.sqrt(torch.clamp(1.0 + mH2, min=eps))
        E_WT = torch.sqrt(torch.clamp(1.0 + mW_T2, min=eps))
        E_WL = torch.sqrt(torch.clamp(1.0 + mW_L2, min=eps))
        E_ZT = torch.sqrt(torch.clamp(1.0 + mZ_T2, min=eps))
        E_ZL = torch.sqrt(torch.clamp(1.0 + mZ_L2, min=eps))
        E_T  = torch.sqrt(torch.clamp(1.0 + mt2, min=eps))

        factorG = coth(EG / (2.0 * tau_t_ext), eps=0.0)
        factorH = coth(EH / (2.0 * tau_t_ext), eps=0.0)
        factor_WT = coth(E_WT / (2.0 * tau_t_ext), eps=0.0)
        factor_WL = coth(E_WL / (2.0 * tau_t_ext), eps=0.0)
        factor_ZT = coth(E_ZT / (2.0 * tau_t_ext), eps=0.0)
        factor_ZL = coth(E_ZL / (2.0 * tau_t_ext), eps=0.0)
        factor_T  = torch.tanh(E_T / (2.0 * tau_t_ext))

        termG = termG_0 * factorG
        termH = termH_0 * factorH
        termWT = termWT0 * factor_WT
        termWL = termWL0 * factor_WL
        termZT = termZT0 * factor_ZT
        termZL = termZL0 * factor_ZL
        termT  = termT0 * factor_T
    else:
        termG, termH = termG_0, termH_0
        termWT, termWL = termWT0, termWL0
        termZT, termZL = termZT0, termZL0
        termT  = termT0

    loop_sum = (
        3.0 * termG
        + termH
        + 4.0 * termWT
        + 2.0 * termWL
        + 2.0 * termZT
        + 1.0 * termZL
        - 12.0 * termT
    )

    rloop = (kloop) * loop_sum
    rloop = torch.clamp(rloop, min=-2.0, max=2.0)

    R = rtree - rloop

    return R
