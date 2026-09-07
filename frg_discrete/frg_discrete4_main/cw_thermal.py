"""
frg_pinns_higgs_singlet/thermal_functions.py の u_CW_zeroT / u_thermal_finiteT
(torch実装) を NumPy に移植したもの。jbjf.py (frg_discrete/perturbation/jbjf.py)
と同じ移植方針で、torch依存を frg_discrete の numpy版 perturbation/running_couplings
だけに置き換えている。

frg_discrete4_main の中で完結させるため、frg_pinns_higgs_singlet 側の
torchコードには一切依存しない (参照実装としてのみ使った)。既存の
frg_discrete4_main の慣習 (README参照) 通り、frg_discrete (兄弟フォルダ)
の共通numpyモジュール (perturbation/, running_couplings.py) は
_pathsetup.py 経由でそのまま再利用する。

用途: 「full flow eq. (rloop+eta)」のNewton-Krylov解 (収束していない) と
比較するための、静的な参照カーブ。running couplings上で評価した
1-loop Coleman-Weinberg (T=0) + 有限温度1-loop熱ポテンシャルを
"RGE-run tree + CW (+ thermal)" として提供する。

注意: これは Wetterich方程式の rloop (flow項、coth閾値関数、regulator込み)
とは別物 (静的な摂動論的1-loop有効ポテンシャル)。flow_equation.py 冒頭コメント
および README の "tree/残差分解" 節参照。
"""

import numpy as np

import _pathsetup  # noqa: F401
from perturbation.config_params import tau_uv, k_IR, t_range
from perturbation.jbjf import J_Bnp, J_Fnp
from running_couplings import get_running_couplings, get_running_quartics, get_running_masses

NGS = 3.0
EPS = 1e-10


def u_tree_rgerun(t_phys, rho_phys, sigma_phys):
    """
    tree potential を、u_tree_exact (canonical scaling, aH*exp(-2t)) の代わりに
    running couplings (RGE) をそのまま代入して評価したもの。
    "RGE-run tree" 系カーブの土台。t=0 (UV) では u_tree_exact と一致するが、
    t<0 では両者は一般に異なる (canonical scalingはtree-levelのRG不変量を
    厳密に保つ一方、RGE-runは2-loop走行couplingsの非自明な変化を反映する)。
    """
    kt = k_IR * np.exp(-t_range + t_phys)
    lamH_t, lamS_t, lamHS_t = get_running_quartics(t_phys)
    mhsq_t, mssq_t = get_running_masses(t_phys)
    muH2_t = mhsq_t / (kt * kt)
    muS2_t = mssq_t / (kt * kt)
    return (
        muH2_t * rho_phys + lamH_t * rho_phys ** 2
        + muS2_t * sigma_phys + lamS_t * sigma_phys ** 2
        + lamHS_t * rho_phys * sigma_phys
    )


def scalar_masses_bare_and_debye(t_phys, rho_phys, sigma_phys):
    """u_tree_rgerun由来の裸の質量固有値 (bare) と、熱Debye補正込みの質量 (Debye)。"""
    kt = k_IR * np.exp(-t_range + t_phys)

    g1_, g2_, yt_, _, _ = get_running_couplings(t_phys)
    lamH_t, lamS_t, lamHS_t = get_running_quartics(t_phys)
    mhsq_t, mssq_t = get_running_masses(t_phys)
    muH2_t = mhsq_t / (kt * kt)
    muS2_t = mssq_t / (kt * kt)

    u_rho = muH2_t + 2.0 * lamH_t * rho_phys + lamHS_t * sigma_phys
    u_sigma = muS2_t + 2.0 * lamS_t * sigma_phys + lamHS_t * rho_phys
    u_rhorho = 2.0 * lamH_t
    u_sigmasigma = 2.0 * lamS_t
    u_rhosigma = lamHS_t

    tau_t = tau_uv * np.exp(-t_phys)

    Pi_H = (
        (3.0 * g2_ ** 2 + g1_ ** 2) / 16.0
        + yt_ ** 2 / 4.0
        + lamH_t / 2.0
        + lamHS_t / 24.0
    ) * tau_t ** 2
    Pi_S = (lamS_t / 4.0 + lamHS_t / 6.0) * tau_t ** 2

    mG2_b = u_rho
    M11_b = u_rho + 2.0 * rho_phys * u_rhorho
    M22_b = u_sigma + 2.0 * sigma_phys * u_sigmasigma
    M12_b = 2.0 * np.sqrt(np.clip(rho_phys * sigma_phys, EPS, None)) * u_rhosigma

    disc_b = (M11_b - M22_b) ** 2 + 4.0 * M12_b ** 2
    sqrt_disc_b = np.sqrt(np.clip(disc_b, EPS, None))
    m1_sq_b = 0.5 * (M11_b + M22_b - sqrt_disc_b)
    m2_sq_b = 0.5 * (M11_b + M22_b + sqrt_disc_b)

    mG2_e = u_rho + Pi_H
    M11_e = u_rho + 2.0 * rho_phys * u_rhorho + Pi_H
    M22_e = u_sigma + 2.0 * sigma_phys * u_sigmasigma + Pi_S
    M12_e = M12_b
    disc_e = (M11_e - M22_e) ** 2 + 4.0 * M12_e ** 2
    sqrt_disc_e = np.sqrt(np.clip(disc_e, EPS, None))
    m1_sq_e = 0.5 * (M11_e + M22_e - sqrt_disc_e)
    m2_sq_e = 0.5 * (M11_e + M22_e + sqrt_disc_e)

    return (mG2_b, m1_sq_b, m2_sq_b), (mG2_e, m1_sq_e, m2_sq_e)


def gauge_masses_bare_and_longitudinal_debye(t_phys, rho_phys):
    tau_t = tau_uv * np.exp(-t_phys)
    g1, g2, yt, _, _ = get_running_couplings(t_phys)

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


def u_CW_zeroT(t_phys, rho_phys, sigma_phys):
    """静的な T=0 1-loop Coleman-Weinberg補正 (mu=k)。renormalization scale = FRGスケールk。"""
    (mG2_b, m1_sq_b, m2_sq_b), _ = scalar_masses_bare_and_debye(t_phys, rho_phys, sigma_phys)
    (mW_T2, mZ_T2, mA_T2), _, g1, g2, yt, _ = gauge_masses_bare_and_longitudinal_debye(t_phys, rho_phys)
    mt2 = yt ** 2 * rho_phys

    def cw_term(m2, dof, c):
        m2_abs = np.abs(m2) + 1e-5
        return (dof / (64.0 * np.pi ** 2)) * (m2 ** 2) * (np.log(m2_abs) - c)

    cw_G = cw_term(mG2_b, NGS, 1.5)
    cw_1 = cw_term(m1_sq_b, 1.0, 1.5)
    cw_2 = cw_term(m2_sq_b, 1.0, 1.5)
    cw_W = cw_term(mW_T2, 6.0, 5.0 / 6.0)
    cw_Z = cw_term(mZ_T2, 3.0, 5.0 / 6.0)
    cw_top = cw_term(mt2, -12.0, 1.5)

    return cw_G + cw_1 + cw_2 + cw_W + cw_Z + cw_top


def u_thermal_finiteT(t_phys, rho_phys, sigma_phys):
    """
    有限温度1-loop熱ポテンシャル (Arnold-Espinosaの1-loop部分のみ、ring項なし)。
    J_B/J_Fの引数には裸の(bare)質量を使う (thermal_functions.py と同じ規約)。
    """
    (mG2_b, m1_sq_b, m2_sq_b), _ = scalar_masses_bare_and_debye(t_phys, rho_phys, sigma_phys)
    (mW_T2, mZ_T2, mA_T2), _, g1, g2, yt, tau_t = gauge_masses_bare_and_longitudinal_debye(t_phys, rho_phys)
    mt2 = yt ** 2 * rho_phys

    tau2 = np.clip(tau_t ** 2, 1e-12, None)
    tau4 = tau2 * tau2

    yG2 = mG2_b / tau2
    y1_2 = m1_sq_b / tau2
    y2_2 = m2_sq_b / tau2
    yWT2 = mW_T2 / tau2
    yZT2 = mZ_T2 / tau2
    yAT2 = mA_T2 / tau2
    yt2_T = mt2 / tau2

    JB_G = J_Bnp(yG2)
    JB_1 = J_Bnp(y1_2)
    JB_2 = J_Bnp(y2_2)
    JB_W = J_Bnp(yWT2)
    JB_Z = J_Bnp(yZT2)
    JB_A = J_Bnp(yAT2)
    JF_t = J_Fnp(yt2_T)

    u_th_scalar = (tau4 / (2.0 * np.pi ** 2)) * (NGS * JB_G + JB_1 + JB_2)
    u_th_gauge = (tau4 / (2.0 * np.pi ** 2)) * (6.0 * JB_W + 3.0 * JB_Z + 1.0 * JB_A)
    u_th_top = -(12.0 * tau4 / (2.0 * np.pi ** 2)) * JF_t

    return u_th_scalar + u_th_gauge + u_th_top


def _m3(m2):
    """(m^2)^{3/2}, clipped at 0 (tachyonic m^2<0 excluded from the ring sum,
    same convention as frg_pinns_higgs_singlet/thermal_functions.py::_m32_pos
    and frg_higgs_only/cw_thermal.py::_m3)."""
    return np.power(np.clip(m2, 0.0, None), 1.5)


def u_ring_correction(t_phys, rho_phys, sigma_phys):
    """
    Arnold-Espinosa ring (daisy) correction: the difference obtained by replacing
    the zero-mode (m^2)^{3/2} of the bare masses with that of the Debye-corrected
    (effective) masses. Ported verbatim (coefficients and signs identical) from
    frg_pinns_higgs_singlet/thermal_functions.py's ring term (lines with ring_G,
    ring_1, ring_2, ring_WL, ring_ZL, ring_AL), the same port already done for the
    single-scalar case in frg_higgs_only/cw_thermal.py::u_ring_correction.
    """
    tau_t = tau_uv * np.exp(-t_phys)

    (mG2_b, m1_sq_b, m2_sq_b), (mG2_e, m1_sq_e, m2_sq_e) = scalar_masses_bare_and_debye(
        t_phys, rho_phys, sigma_phys
    )
    (mW_T2, mZ_T2, mA_T2), (mW_L2, mZ_L2, mA_L2), *_ = gauge_masses_bare_and_longitudinal_debye(
        t_phys, rho_phys
    )

    ring_G = NGS * (_m3(mG2_e) - _m3(mG2_b))
    ring_1 = 1.0 * (_m3(m1_sq_e) - _m3(m1_sq_b))
    ring_2 = 1.0 * (_m3(m2_sq_e) - _m3(m2_sq_b))
    ring_WL = 2.0 * (_m3(mW_L2) - _m3(mW_T2))
    ring_ZL = 1.0 * (_m3(mZ_L2) - _m3(mZ_T2))
    ring_AL = 1.0 * (_m3(mA_L2) - _m3(mA_T2))

    return -(tau_t / (12.0 * np.pi)) * (
        ring_G + ring_1 + ring_2 + ring_WL + ring_ZL + ring_AL
    )


def u_thermal_finiteT_ring(t_phys, rho_phys, sigma_phys):
    """1-loop (bare masses, u_thermal_finiteT) + ring/daisy correction
    (full Arnold-Espinosa). This is the physical finite-temperature reference
    curve, matching frg_higgs_only/cw_thermal.py::u_thermal_finiteT_ring and the
    PINN's thermal reference."""
    return (
        u_thermal_finiteT(t_phys, rho_phys, sigma_phys)
        + u_ring_correction(t_phys, rho_phys, sigma_phys)
    )


def set_T_raw(T_raw):
    """
    perturbation.config_params.tau_uv (モジュールimport時に一度だけ計算される
    T_RAW/k_IR*exp(t) 定数) を、T_RAW=T_raw [GeV] に差し替えた値で上書きする。
    tau_uv を直接importしているこのモジュール自身のグローバルと、
    (呼び出し元が同様に) flow_equation モジュールのグローバルの両方を
    書き換える必要がある点に注意 (どちらも "from ... import tau_uv" で
    値をコピーしているため、config_params.tau_uv を書き換えるだけでは
    反映されない)。
    """
    global tau_uv
    from perturbation.config_params import k_IR as _k_IR, t as _t
    tau_uv = (T_raw / _k_IR) * np.exp(_t)
    return tau_uv
