"""
FRGフローの初期条件 (UV端, t_phys=0) を与える種ポテンシャル。

frg_pinns_higgs_singlet の以下の関数の NumPy 移植:
  - my_networks.u_tree_exact
  - thermal_functions.u_thermal_finiteT (1-loop, bare質量のみ; リング項は無効化)
    (thermal_functions.py 内でも ring 項はコメントアウトされ u_th = u_th_1loop の
     みが使われているため、それに合わせている)
  - loss_bc0 で実際に境界条件として課される u_seed = u_tree_exact + u_thermal_finiteT
    (uv_cw のCW寄与はデフォルト重み0で無効なので含めていない)
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
    有限温度の1-loop熱ポテンシャル u_th(t, rho, sigma) (Arnold-Espinosa style, ring項なし)。
    thermal_functions.u_thermal_finiteT と同じ内容 (bare質量のみ使用)。
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
    FRGフローの初期条件 (t_phys=0, config_params.t_range に対応するUV端)。
    loss_bc0 が実際にPINNへ課す境界条件 (uv_cw=0 のデフォルト設定) と同じ内容。
    """
    u = u_tree_exact(t_phys, rho_phys, sigma_phys)
    if finite_T:
        u = u + u_thermal_finiteT(t_phys, rho_phys, sigma_phys)
    return u
