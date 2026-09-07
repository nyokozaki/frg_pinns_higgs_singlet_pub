"""
frg_discrete4_main/cw_thermal.py の singlet除去版(Higgs単一チャンネル)。

用途: 「full flow eq. (rloop+eta)」のNewton-Krylov解と比較するための、
静的な参照カーブ。running couplings上で評価した1-loop Coleman-Weinberg
(T=0) + 有限温度1-loop熱ポテンシャルを "RGE-run tree + CW (+ thermal)" と
して提供する。
"""

import numpy as np

import _pathsetup  # noqa: F401
from perturbation.config_params import tau_uv, k_IR, t_range
from perturbation.jbjf import J_Bnp, J_Fnp
from running_couplings_higgs_only import get_running_couplings, get_running_quartics, get_running_masses

NGS = 3.0
EPS = 1e-10


def u_tree_rgerun(t_phys, rho_phys):
    """tree potential を、u_tree_exact の代わりに running couplings をそのまま代入して評価。"""
    kt = k_IR * np.exp(-t_range + t_phys)
    lamH_t = get_running_quartics(t_phys)
    mhsq_t = get_running_masses(t_phys)
    muH2_t = mhsq_t / (kt * kt)
    return muH2_t * rho_phys + lamH_t * rho_phys ** 2


def Pi_H_thermal_mass(t_phys):
    """
    Higgs/Goldstoneの熱Debye質量補正 (large-T展開のleading order, ~tau_t^2)。
    rho非依存 (質量項=aHへの一様なシフトのみ、quarticには効かない)。
    scalar_masses_bare_and_debyeが内部でも使う。
    """
    g1_, g2_, yt_ = get_running_couplings(t_phys)
    lamH_t = get_running_quartics(t_phys)
    tau_t = tau_uv * np.exp(-t_phys)
    return (
        (3.0 * g2_ ** 2 + g1_ ** 2) / 16.0
        + yt_ ** 2 / 4.0
        + lamH_t / 2.0
    ) * tau_t ** 2


def scalar_masses_bare_and_debye(t_phys, rho_phys):
    """u_tree_rgerun由来の裸の質量 (bare) と、熱Debye補正込みの質量 (Debye)。"""
    kt = k_IR * np.exp(-t_range + t_phys)

    lamH_t = get_running_quartics(t_phys)
    mhsq_t = get_running_masses(t_phys)
    muH2_t = mhsq_t / (kt * kt)

    u_rho = muH2_t + 2.0 * lamH_t * rho_phys
    u_rhorho = 2.0 * lamH_t

    Pi_H = Pi_H_thermal_mass(t_phys)

    mG2_b = u_rho
    mH2_b = u_rho + 2.0 * rho_phys * u_rhorho

    mG2_e = u_rho + Pi_H
    mH2_e = u_rho + 2.0 * rho_phys * u_rhorho + Pi_H

    return (mG2_b, mH2_b), (mG2_e, mH2_e)


def gauge_masses_bare_and_longitudinal_debye(t_phys, rho_phys):
    tau_t = tau_uv * np.exp(-t_phys)
    g1, g2, yt = get_running_couplings(t_phys)

    mW_T2 = 0.5 * g2 ** 2 * rho_phys
    mZ_T2 = 0.5 * (g2 ** 2 + g1 ** 2) * rho_phys
    mA_T2 = np.zeros_like(rho_phys)

    Pi_W = (11.0 / 6.0) * g2 ** 2 * tau_t ** 2
    Pi_B = (11.0 / 6.0) * g1 ** 2 * tau_t ** 2

    mW_L2 = mW_T2 + Pi_W
    M11 = mW_T2 + Pi_W
    M22 = 0.5 * g1 ** 2 * rho_phys + Pi_B
    M12 = -0.5 * g2 * g1 * rho_phys

    disc = (M11 - M22) ** 2 + 4.0 * M12 ** 2
    sqrt_disc = np.sqrt(np.clip(disc, EPS, None))
    mZ_L2 = 0.5 * (M11 + M22 + sqrt_disc)
    mA_L2 = 0.5 * (M11 + M22 - sqrt_disc)

    return (mW_T2, mZ_T2, mA_T2), (mW_L2, mZ_L2, mA_L2), g1, g2, yt, tau_t


def u_CW_zeroT(t_phys, rho_phys):
    """静的な T=0 1-loop Coleman-Weinberg補正 (mu=k)。renormalization scale = FRGスケールk。"""
    (mG2_b, mH2_b), _ = scalar_masses_bare_and_debye(t_phys, rho_phys)
    (mW_T2, mZ_T2, mA_T2), _, g1, g2, yt, _ = gauge_masses_bare_and_longitudinal_debye(t_phys, rho_phys)
    mt2 = yt ** 2 * rho_phys

    def cw_term(m2, dof, c):
        m2_abs = np.abs(m2) + 1e-5
        return (dof / (64.0 * np.pi ** 2)) * (m2 ** 2) * (np.log(m2_abs) - c)

    cw_G = cw_term(mG2_b, NGS, 1.5)
    cw_H = cw_term(mH2_b, 1.0, 1.5)
    cw_W = cw_term(mW_T2, 6.0, 5.0 / 6.0)
    cw_Z = cw_term(mZ_T2, 3.0, 5.0 / 6.0)
    cw_top = cw_term(mt2, -12.0, 1.5)

    return cw_G + cw_H + cw_W + cw_Z + cw_top


def u_thermal_finiteT(t_phys, rho_phys):
    """
    有限温度1-loop熱ポテンシャル (Arnold-Espinosaの1-loop部分のみ、ring項なし)。
    J_B/J_Fの引数には裸の(bare)質量を使う。
    """
    (mG2_b, mH2_b), _ = scalar_masses_bare_and_debye(t_phys, rho_phys)
    (mW_T2, mZ_T2, mA_T2), _, g1, g2, yt, tau_t = gauge_masses_bare_and_longitudinal_debye(t_phys, rho_phys)
    mt2 = yt ** 2 * rho_phys

    tau2 = np.clip(tau_t ** 2, 1e-12, None)
    tau4 = tau2 * tau2

    yG2 = mG2_b / tau2
    yH2 = mH2_b / tau2
    yWT2 = mW_T2 / tau2
    yZT2 = mZ_T2 / tau2
    yAT2 = mA_T2 / tau2
    yt2_T = mt2 / tau2

    JB_G = J_Bnp(yG2)
    JB_H = J_Bnp(yH2)
    JB_W = J_Bnp(yWT2)
    JB_Z = J_Bnp(yZT2)
    JB_A = J_Bnp(yAT2)
    JF_t = J_Fnp(yt2_T)

    u_th_scalar = (tau4 / (2.0 * np.pi ** 2)) * (NGS * JB_G + JB_H)
    u_th_gauge = (tau4 / (2.0 * np.pi ** 2)) * (6.0 * JB_W + 3.0 * JB_Z + 1.0 * JB_A)
    u_th_top = -(12.0 * tau4 / (2.0 * np.pi ** 2)) * JF_t

    return u_th_scalar + u_th_gauge + u_th_top


def _m3(m2):
    """(m^2)^{3/2}, clipped at 0 (tachyonic m^2<0 excluded from the ring sum,
    same convention as frg_pinns_higgs_only/thermal_np.py::get_m3)."""
    return np.power(np.clip(m2, 0.0, None), 1.5)


def u_ring_correction(t_phys, rho_phys):
    """
    Arnold-Espinosaのring(daisy)補正: ゼロモードの (m^2)^{3/2} を Debye補正済み
    (effective) 質量の (m^2_eff)^{3/2} に置き換えた差分。frg_pinns_higgs_only/
    thermal_np.py::u_thermal_finiteT_AE のring項をそのまま移植 (係数・符号とも同一)。
    """
    tau_t = tau_uv * np.exp(-t_phys)

    (mG2_b, mH2_b), (mG2_e, mH2_e) = scalar_masses_bare_and_debye(t_phys, rho_phys)
    (mW_T2, mZ_T2, mA_T2), (mW_L2, mZ_L2, mA_L2), *_ = gauge_masses_bare_and_longitudinal_debye(
        t_phys, rho_phys
    )

    ring_G = NGS * (_m3(mG2_e) - _m3(mG2_b))
    ring_H = 1.0 * (_m3(mH2_e) - _m3(mH2_b))
    ring_WL = 2.0 * (_m3(mW_L2) - _m3(mW_T2))
    ring_ZL = 1.0 * (_m3(mZ_L2) - _m3(mZ_T2))
    ring_AL = 1.0 * (_m3(mA_L2) - _m3(mA_T2))

    return -(tau_t / (12.0 * np.pi)) * (ring_G + ring_H + ring_WL + ring_ZL + ring_AL)


def u_thermal_finiteT_ring(t_phys, rho_phys):
    """1-loop (bare質量, u_thermal_finiteT) + ring/daisy補正 (Arnold-Espinosa完全版)。"""
    return u_thermal_finiteT(t_phys, rho_phys) + u_ring_correction(t_phys, rho_phys)


def set_T_raw(T_raw):
    """
    perturbation.config_params.tau_uv を T_RAW=T_raw [GeV] に差し替えた値で
    上書きする。このモジュール自身のグローバル (と、呼び出し元が同様に扱う
    flow_equation_higgs_only 側のグローバル) の両方を書き換える必要がある点に
    注意 (どちらも "from ... import tau_uv" で値をコピーしているため)。
    """
    global tau_uv
    from perturbation.config_params import k_IR as _k_IR, t as _t
    tau_uv = (T_raw / _k_IR) * np.exp(_t)
    return tau_uv
