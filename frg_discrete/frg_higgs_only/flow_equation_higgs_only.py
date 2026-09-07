"""
frg_discrete/flow_equation.py の singlet除去版(Higgs単一チャンネル)。

Wetterich方程式 (LPA', 有限温度) の右辺を格子上で評価する。singletが存在
しないため、質量固有値の混合 (M11,M22,M12,disc,sqrt_disc) はそもそも
出てこず、Higgs質量 mH2 = u_rho + 2*rho*u_rhorho と Goldstone質量
mG2 = u_rho の2つだけになる。

frg_discrete/flow_equation.py の disc→0 分岐点非平滑性 (README.md参照) は、
この縮約では構造的に発生し得ない。
"""

import numpy as np

from perturbation.config_params import tau_uv, finite_T
from running_couplings_higgs_only import get_running_couplings
from seed_potential_higgs_only import aH, lamH

kloop = 1.0 / (12.0 * np.pi ** 2)
NGS = 3.0
LPAp = 1.0
EPS = 1e-10


def eta_rho(t_phys):
    g1, g2, yt = get_running_couplings(t_phys)
    gamma = (1.0 / (16.0 * np.pi ** 2)) * (
        3.0 * yt ** 2 - 3.0 * g1 ** 2 / 4.0 - 9.0 * g2 ** 2 / 4.0
    )
    return 2.0 * gamma


def _coth(x, eps=0.0):
    return np.cosh(x) / (np.sinh(x) + eps)


def _inv_sqrt(m2, floor=EPS):
    return 1.0 / np.sqrt(np.clip(1.0 + m2, floor, None))


def u_tree_derivs(t_phys, rho):
    """u_tree_exact の (u_rho, u_rhorho) を厳密に返す。"""
    muH2_t = aH * np.exp(-2.0 * t_phys)
    u_rho = muH2_t + 2.0 * lamH * rho
    u_rhorho = 2.0 * lamH
    return u_rho, u_rhorho


def _rloop_from_derivs(t_phys, rho, u_rho, u_rhorho, warn_on_tachyon=False, mass_floor=0.0):
    """
    質量固有値 (mG2, mH2、混合なし) と熱閾値関数から rloop = kloop*loop_sum を計算する。
    singletがないので disc_delta/disc_cut に相当するものはそもそも不要。
    """
    g1, g2, yt = get_running_couplings(t_phys)

    mG2 = u_rho
    mH2 = u_rho + 2.0 * rho * u_rhorho

    if warn_on_tachyon:
        if np.any(1.0 + mG2 < 0.0) or np.any(1.0 + mH2 < 0.0):
            import warnings
            warnings.warn(
                f"t={t_phys:.4f}: 1+mG2 or 1+mH2 went negative "
                f"(min 1+mG2={np.min(1.0+mG2):.3e}, min 1+mH2={np.min(1.0+mH2):.3e}); "
                "clamped at eps, regulator is being pushed into an unphysical regime."
            )

    mW_T2 = 0.5 * g2 ** 2 * rho
    mZ_T2 = 0.5 * (g2 ** 2 + g1 ** 2) * rho
    mt2 = yt ** 2 * rho
    mW_L2 = mW_T2
    mZ_L2 = mZ_T2

    mass_eps = mass_floor if mass_floor > 0.0 else EPS
    termG_0 = _inv_sqrt(mG2, floor=mass_eps)
    termH_0 = _inv_sqrt(mH2, floor=mass_eps)
    termWT0 = _inv_sqrt(mW_T2)
    termWL0 = _inv_sqrt(mW_L2)
    termZT0 = _inv_sqrt(mZ_T2)
    termZL0 = _inv_sqrt(mZ_L2)
    termT0 = _inv_sqrt(mt2)

    if finite_T:
        tau_t = tau_uv * np.exp(-t_phys)

        EG = np.sqrt(np.clip(1.0 + mG2, mass_eps, None))
        EH = np.sqrt(np.clip(1.0 + mH2, mass_eps, None))
        E_WT = np.sqrt(np.clip(1.0 + mW_T2, EPS, None))
        E_WL = np.sqrt(np.clip(1.0 + mW_L2, EPS, None))
        E_ZT = np.sqrt(np.clip(1.0 + mZ_T2, EPS, None))
        E_ZL = np.sqrt(np.clip(1.0 + mZ_L2, EPS, None))
        E_T = np.sqrt(np.clip(1.0 + mt2, EPS, None))

        factorG = _coth(EG / (2.0 * tau_t))
        factorH = _coth(EH / (2.0 * tau_t))
        factor_WT = _coth(E_WT / (2.0 * tau_t))
        factor_WL = _coth(E_WL / (2.0 * tau_t))
        factor_ZT = _coth(E_ZT / (2.0 * tau_t))
        factor_ZL = _coth(E_ZL / (2.0 * tau_t))
        factor_T = np.tanh(E_T / (2.0 * tau_t))

        termG = termG_0 * factorG
        termH = termH_0 * factorH
        termWT = termWT0 * factor_WT
        termWL = termWL0 * factor_WL
        termZT = termZT0 * factor_ZT
        termZL = termZL0 * factor_ZL
        termT = termT0 * factor_T
    else:
        termG, termH = termG_0, termH_0
        termWT, termWL = termWT0, termWL0
        termZT, termZL = termZT0, termZL0
        termT = termT0

    loop_sum = (
        3.0 * termG
        + termH
        + 4.0 * termWT
        + 2.0 * termWL
        + 2.0 * termZT
        + 1.0 * termZL
        - 12.0 * termT
    )

    rloop = kloop * loop_sum
    rloop = np.clip(rloop, -2.0, 2.0)
    return rloop


def flow_rhs(t_phys, U, grid, warn_on_tachyon=False, scheme="upwind", mass_floor=0.0):
    """du/dt(t_phys, rho) を格子全体で返す (フル u を直接積分)。"""
    u_rho, u_rhorho = grid.derivatives(U, scheme=scheme)
    rho = grid.RHO

    eta_r = eta_rho(t_phys)

    rloop = _rloop_from_derivs(
        t_phys, rho, u_rho, u_rhorho, warn_on_tachyon=warn_on_tachyon, mass_floor=mass_floor,
    )

    du_dt = -4.0 * U + (2.0 + eta_r * LPAp) * rho * u_rho + rloop
    return du_dt


def flow_rhs_w(t_phys, W, grid, warn_on_tachyon=False, scheme="upwind"):
    """u = u_tree_exact(t,rho) + w とおいたときの dw/dt を返す。"""
    w_rho, w_rhorho = grid.derivatives(W, scheme=scheme)
    rho = grid.RHO

    ut_rho, ut_rhorho = u_tree_derivs(t_phys, rho)

    u_rho = ut_rho + w_rho
    u_rhorho = ut_rhorho + w_rhorho

    eta_r = eta_rho(t_phys)

    rloop = _rloop_from_derivs(t_phys, rho, u_rho, u_rhorho, warn_on_tachyon=warn_on_tachyon)

    F_tree = eta_r * rho * ut_rho

    dw_dt = F_tree - 4.0 * W + (2.0 + eta_r * LPAp) * rho * w_rho + rloop
    return dw_dt
