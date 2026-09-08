"""
Seed potential giving the initial condition of the FRG flow (UV edge, t_phys=0).

NumPy port of the following functions from frg_pinns_higgs_singlet:
  - my_networks.u_tree_exact
  - thermal_functions.u_thermal_finiteT (1-loop, bare masses only; ring term disabled)
    (in thermal_functions.py the ring term is also commented out and only
     u_th = u_th_1loop is used, so this matches that.)
  - u_seed = u_tree_exact + u_thermal_finiteT, the boundary condition actually
    imposed by loss_bc0 (the CW contribution of uv_cw has default weight 0 and
    is therefore not included).
"""

import numpy as np

from perturbation.config_params import tau_uv, k_IR, t_range, finite_T
from perturbation.jbjf import J_Bnp as J_B, J_Fnp as J_F
from running_couplings import tree_params, get_running_couplings, get_running_quartics, get_running_masses

aH, aS, lamH, lamS, lamHS = tree_params

NGS = 3.0
_EPS = 1e-10


def u_tree_exact(t_phys, rho_phys, sigma_phys):
    muH2_t = aH * np.exp(-2.0 * t_phys)
    muS2_t = aS * np.exp(-2.0 * t_phys)

    return (
        muH2_t * rho_phys
        + lamH * rho_phys ** 2
        + muS2_t * sigma_phys
        + lamS * sigma_phys ** 2
        + lamHS * rho_phys * sigma_phys
    )


def _scalar_masses_bare(rho_phys, sigma_phys, t_phys):
    kt = k_IR * np.exp(-t_range + t_phys)

    g1, g2, yt, _, _ = get_running_couplings(t_phys)
    lamH_t, lamS_t, lamHS_t = get_running_quartics(t_phys)
    mhsq_t, mssq_t = get_running_masses(t_phys)

    muH2_t = mhsq_t / (kt * kt)
    muS2_t = mssq_t / (kt * kt)

    u_rho = muH2_t + 2.0 * lamH_t * rho_phys + lamHS_t * sigma_phys
    u_sigma = muS2_t + 2.0 * lamS_t * sigma_phys + lamHS_t * rho_phys

    u_rhorho = 2.0 * lamH_t
    u_sigmasigma = 2.0 * lamS_t
    u_rhosigma = lamHS_t

    mG2_b = u_rho
    M11_b = u_rho + 2.0 * rho_phys * u_rhorho
    M22_b = u_sigma + 2.0 * sigma_phys * u_sigmasigma
    M12_b = 2.0 * np.sqrt(np.clip(rho_phys * sigma_phys, _EPS, None)) * u_rhosigma

    disc_b = (M11_b - M22_b) ** 2 + 4.0 * M12_b ** 2
    sqrt_disc_b = np.sqrt(np.clip(disc_b, _EPS, None))

    m1_sq_b = 0.5 * (M11_b + M22_b - sqrt_disc_b)
    m2_sq_b = 0.5 * (M11_b + M22_b + sqrt_disc_b)

    return mG2_b, m1_sq_b, m2_sq_b, g1, g2, yt


def u_thermal_finiteT(t_phys, rho_phys, sigma_phys):
    """
    Finite-temperature one-loop thermal potential u_th(t, rho, sigma)
    (Arnold-Espinosa style, no ring term). Same content as
    thermal_functions.u_thermal_finiteT (bare masses only).
    """
    mG2_b, m1_sq_b, m2_sq_b, g1, g2, yt = _scalar_masses_bare(rho_phys, sigma_phys, t_phys)

    mW_T2 = 0.5 * g2 ** 2 * rho_phys
    mZ_T2 = 0.5 * (g2 ** 2 + g1 ** 2) * rho_phys
    mA_T2 = np.zeros_like(rho_phys, dtype=float) * 1.0
    mt2 = yt ** 2 * rho_phys

    tau_t = tau_uv * np.exp(-t_phys)
    tau2 = np.clip(tau_t ** 2, 1e-12, None)
    tau4 = tau2 * tau2

    JB_G = J_B(mG2_b / tau2)
    JB_1 = J_B(m1_sq_b / tau2)
    JB_2 = J_B(m2_sq_b / tau2)

    JB_W = J_B(mW_T2 / tau2)
    JB_Z = J_B(mZ_T2 / tau2)
    JB_A = J_B(mA_T2 / tau2)

    JF_t = J_F(mt2 / tau2)

    u_th_scalar = (tau4 / (2.0 * np.pi ** 2)) * (NGS * JB_G + JB_1 + JB_2)
    u_th_gauge = (tau4 / (2.0 * np.pi ** 2)) * (6.0 * JB_W + 3.0 * JB_Z + 1.0 * JB_A)
    u_th_top = -(12.0 * tau4 / (2.0 * np.pi ** 2)) * JF_t

    return u_th_scalar + u_th_gauge + u_th_top


def u_seed(t_phys, rho_phys, sigma_phys):
    """
    Initial condition of the FRG flow (t_phys=0, the UV edge corresponding to
    config_params.t_range). Same content as the boundary condition loss_bc0
    actually imposes on the PINN (default setting uv_cw=0).
    """
    u = u_tree_exact(t_phys, rho_phys, sigma_phys)
    if finite_T:
        u = u + u_thermal_finiteT(t_phys, rho_phys, sigma_phys)
    return u
