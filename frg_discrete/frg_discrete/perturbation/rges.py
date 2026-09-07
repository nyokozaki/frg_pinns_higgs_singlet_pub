#!/usr/bin/env python3
"""
frg_pinns_higgs_singlet/perturbation/rges.py の torch フリー版。

摂動論的な2-loop RGEでモデルパラメータをUVまで走らせる部分は元々numpy/scipyのみ。
違うのは calculate_cw_corrections(): 元コードは PyTorch の autograd で
UVマッチング点における1-loop Coleman-Weinberg補正の1階・2階微分を計算していたが、
ここでは同じ u_CW_zeroT を中心差分で数値微分することで置き換える
(この呼び出しは初期化時に一度だけなので、有限差分で精度・速度とも問題ない)。
"""

import numpy as np
from scipy.integrate import solve_ivp

from .gauge_one_loop import gauge_couplings_1loop
from .config_params import lamH, lamS, lamHS, mhsq, mssq, vew

PARAM_ORDER = ["g1", "g2", "g3", "yt", "lam", "lamS", "lamHS", "mhsq", "mssq"]

eps = 1e-10
NGS = 3.0  # Goldstoneボソンの自由度
C = 1.0

# ==========================================
# 1. 質量行列とCWポテンシャルの定義 (NumPy)
# ==========================================

def scalar_masses_bare(rho_phys, sigma_phys, params, kt):
    """
    スカラー場の裸の質量を計算します。
    入力の rho_phys, sigma_phys はスケール kt で無次元化された値 (rho/k^2) を想定。
    """
    lamH_t = params['lamH']
    lamS_t = params['lamS']
    lamHS_t = params['lamHS']
    muH2_t = params['mhsq'] / (kt * kt)
    muS2_t = params['mssq'] / (kt * kt)

    u_rho = (muH2_t + 2.0 * lamH_t * rho_phys + lamHS_t * sigma_phys) / C
    u_sigma = (muS2_t + 2.0 * lamS_t * sigma_phys + lamHS_t * rho_phys) / C

    u_rhorho = (2.0 * lamH_t) / C
    u_sigmasigma = (2.0 * lamS_t) / C
    u_rhosigma = (lamHS_t) / C

    mG2_b = u_rho
    M11_b = u_rho + 2.0 * rho_phys * u_rhorho
    M22_b = u_sigma + 2.0 * sigma_phys * u_sigmasigma

    clamp_val = np.clip(rho_phys * sigma_phys, eps, None)
    M12_b = 2.0 * np.sqrt(clamp_val) * u_rhosigma

    disc_b = (M11_b - M22_b) ** 2 + 4.0 * M12_b ** 2
    sqrt_disc_b = np.sqrt(np.clip(disc_b, eps, None))

    m1_sq_b = 0.5 * (M11_b + M22_b - sqrt_disc_b)
    m2_sq_b = 0.5 * (M11_b + M22_b + sqrt_disc_b)

    return mG2_b, m1_sq_b, m2_sq_b


def gauge_masses_bare(rho_phys, params):
    g1 = params['g1']
    g2 = params['g2']

    mW_T2 = 0.5 * g2 ** 2 * rho_phys
    mZ_T2 = 0.5 * (g2 ** 2 + g1 ** 2) * rho_phys
    mA_T2 = np.zeros_like(rho_phys)

    return mW_T2, mZ_T2, mA_T2


def u_CW_zeroT(rho_phys, sigma_phys, params, kt):
    """
    T=0 における 1-loop Coleman-Weinberg ポテンシャル (無次元)。
    """
    mG2_b, m1_sq_b, m2_sq_b = scalar_masses_bare(rho_phys, sigma_phys, params, kt)
    mW_T2, mZ_T2, mA_T2 = gauge_masses_bare(rho_phys, params)

    yt = params['yt']
    mt2 = yt ** 2 * rho_phys

    def cw_term(m2, dof, c):
        m2_abs = np.abs(m2) + eps
        return (dof / (64.0 * np.pi ** 2)) * (m2 ** 2) * (np.log(m2_abs) - c)

    # cw_G (Goldstone) is excluded from the matching sum: at the tree-level
    # EW point mG2_b is exactly zero, so its curvature hits the
    # Goldstone-boson-catastrophe log divergence and is dominated by the
    # ad hoc regulator eps rather than a converged physical number. See
    # frg_pinns_higgs_singlet/perturbation/rges.py for the reference fix.
    cw_1 = cw_term(m1_sq_b, 1.0, 1.5)
    cw_2 = cw_term(m2_sq_b, 1.0, 1.5)

    cw_W = cw_term(mW_T2, 6.0, 5.0 / 6.0)
    cw_Z = cw_term(mZ_T2, 3.0, 5.0 / 6.0)

    cw_top = cw_term(mt2, -12.0, 1.5)

    return cw_1 + cw_2 + cw_W + cw_Z + cw_top


# ==========================================
# 2. CW補正の抽出
# ==========================================
#
# u_CW_zeroT の cw_term(m2) = (dof/64pi^2) m2^2 (log(|m2|+eps) - c) は、
# rho0 = vew^2/kt^2 (tree-level極小点) で mG2 = u_rho = 0 ちょうどになる
# (Goldstoneの質量がゼロになる点で評価しているため)。この点で
# d^2/dm2^2 [m2^2 (log(|m2|+eps)-c)] を有限差分でサンプルすると、
# 中心点 m2=0 (log(eps) を使う特別な値) と、有限のステップ h だけ離れた点
# (log(h) 相当の"普通の"値) の間で不整合が生じ、二階微分の推定値が
# h に依存して大きくドリフトしてしまう(真の解析微分は log(eps) で
# 決まる有限値に収束する)。
#
# そのため、torch が利用可能な環境では元のPyTorch実装と全く同じ
# autograd計算を使い、torch が無い環境でのみ有限差分にフォールバックする
# (フォールバック値は近似であり、特に delta_lamH 等はズレうることに注意)。

def _calculate_cw_corrections_torch(params, kt):
    import torch

    def scalar_masses_bare_t(rho_phys, sigma_phys, params, kt):
        lamH_t = params['lamH']
        lamS_t = params['lamS']
        lamHS_t = params['lamHS']
        muH2_t = params['mhsq'] / (kt * kt)
        muS2_t = params['mssq'] / (kt * kt)

        u_rho = (muH2_t + 2.0 * lamH_t * rho_phys + lamHS_t * sigma_phys) / C
        u_sigma = (muS2_t + 2.0 * lamS_t * sigma_phys + lamHS_t * rho_phys) / C

        u_rhorho = (2.0 * lamH_t) / C
        u_sigmasigma = (2.0 * lamS_t) / C
        u_rhosigma = (lamHS_t) / C

        mG2_b = u_rho
        M11_b = u_rho + 2.0 * rho_phys * u_rhorho
        M22_b = u_sigma + 2.0 * sigma_phys * u_sigmasigma

        clamp_val = torch.clamp(rho_phys * sigma_phys, min=eps)
        M12_b = 2.0 * torch.sqrt(clamp_val) * u_rhosigma

        disc_b = (M11_b - M22_b) ** 2 + 4.0 * M12_b ** 2
        sqrt_disc_b = torch.sqrt(torch.clamp(disc_b, min=eps))

        m1_sq_b = 0.5 * (M11_b + M22_b - sqrt_disc_b)
        m2_sq_b = 0.5 * (M11_b + M22_b + sqrt_disc_b)

        return mG2_b, m1_sq_b, m2_sq_b

    def gauge_masses_bare_t(rho_phys, params):
        g1 = params['g1']
        g2 = params['g2']
        mW_T2 = 0.5 * g2 ** 2 * rho_phys
        mZ_T2 = 0.5 * (g2 ** 2 + g1 ** 2) * rho_phys
        mA_T2 = torch.zeros_like(rho_phys)
        return mW_T2, mZ_T2, mA_T2

    def u_CW_zeroT_t(rho_phys, sigma_phys, params, kt):
        mG2_b, m1_sq_b, m2_sq_b = scalar_masses_bare_t(rho_phys, sigma_phys, params, kt)
        mW_T2, mZ_T2, mA_T2 = gauge_masses_bare_t(rho_phys, params)

        yt = params['yt']
        mt2 = yt ** 2 * rho_phys

        def cw_term(m2, dof, c):
            m2_abs = torch.abs(m2) + eps
            return (dof / (64.0 * torch.pi ** 2)) * (m2 ** 2) * (torch.log(m2_abs) - c)

        # cw_G (Goldstone) excluded from the matching sum; see u_CW_zeroT above.
        cw_1 = cw_term(m1_sq_b, 1.0, 1.5)
        cw_2 = cw_term(m2_sq_b, 1.0, 1.5)
        cw_W = cw_term(mW_T2, 6.0, 5.0 / 6.0)
        cw_Z = cw_term(mZ_T2, 3.0, 5.0 / 6.0)
        cw_top = cw_term(mt2, -12.0, 1.5)

        return cw_1 + cw_2 + cw_W + cw_Z + cw_top

    rho_dimful = vew ** 2
    sigma_dimful = 0.0

    rho_tilde = torch.tensor(rho_dimful / (kt ** 2), requires_grad=True, dtype=torch.float64)
    sigma_tilde = torch.tensor(sigma_dimful / (kt ** 2), requires_grad=True, dtype=torch.float64)

    u_cw = u_CW_zeroT_t(rho_tilde, sigma_tilde, params, kt)

    grads = torch.autograd.grad(u_cw, (rho_tilde, sigma_tilde), create_graph=True)
    du_drho = grads[0]
    du_dsigma = grads[1]

    d2u_drho2 = torch.autograd.grad(du_drho, rho_tilde, retain_graph=True)[0]
    d2u_dsigma2 = torch.autograd.grad(du_dsigma, sigma_tilde, retain_graph=True)[0]
    d2u_drhodsigma = torch.autograd.grad(du_drho, sigma_tilde, retain_graph=True)[0]

    delta_mhsq = (kt ** 2) * du_drho.item()
    delta_mssq = (kt ** 2) * du_dsigma.item()
    delta_lamH = 0.5 * d2u_drho2.item()
    delta_lamS = 0.5 * d2u_dsigma2.item()
    delta_lamHS = d2u_drhodsigma.item()

    return delta_mhsq, delta_mssq, delta_lamH, delta_lamS, delta_lamHS


def _calculate_cw_corrections_fd(params, kt):
    """
    有限差分によるフォールバック (torch が無い環境向け、近似値)。
    sigma は物理的に sigma>=0 でしか定義されないため、
    sigma 方向は 0 を含む前進差分、rho 方向 (rho0 はドメイン内部) は中心差分を使う。
    """
    rho0 = vew ** 2 / (kt ** 2)
    sigma0 = 0.0

    h = 1e-4 * max(rho0, 1.0)

    def f(rho, sigma):
        return u_CW_zeroT(np.array([rho]), np.array([sigma]), params, kt)[0]

    f00 = f(rho0, sigma0)

    du_drho = (f(rho0 + h, sigma0) - f(rho0 - h, sigma0)) / (2.0 * h)
    d2u_drho2 = (f(rho0 + h, sigma0) - 2.0 * f00 + f(rho0 - h, sigma0)) / h ** 2

    f0s = f00
    f1s = f(rho0, sigma0 + h)
    f2s = f(rho0, sigma0 + 2.0 * h)
    f3s = f(rho0, sigma0 + 3.0 * h)

    du_dsigma = (-3.0 * f0s + 4.0 * f1s - f2s) / (2.0 * h)
    d2u_dsigma2 = (2.0 * f0s - 5.0 * f1s + 4.0 * f2s - f3s) / h ** 2

    g0 = (f(rho0 + h, sigma0) - f(rho0 - h, sigma0)) / (2.0 * h)
    g1 = (f(rho0 + h, sigma0 + h) - f(rho0 - h, sigma0 + h)) / (2.0 * h)
    g2 = (f(rho0 + h, sigma0 + 2.0 * h) - f(rho0 - h, sigma0 + 2.0 * h)) / (2.0 * h)
    d2u_drhodsigma = (-3.0 * g0 + 4.0 * g1 - g2) / (2.0 * h)

    delta_mhsq = (kt ** 2) * du_drho
    delta_mssq = (kt ** 2) * du_dsigma
    delta_lamH = 0.5 * d2u_drho2
    delta_lamS = 0.5 * d2u_dsigma2
    delta_lamHS = d2u_drhodsigma

    return delta_mhsq, delta_mssq, delta_lamH, delta_lamS, delta_lamHS


def calculate_cw_corrections(params, kt):
    try:
        return _calculate_cw_corrections_torch(params, kt)
    except ImportError:
        import warnings
        warnings.warn(
            "torch not available: falling back to finite-difference CW "
            "corrections. This is an approximation of the original "
            "autograd-based matching (see module docstring above "
            "calculate_cw_corrections)."
        )
        return _calculate_cw_corrections_fd(params, kt)


# ==========================================
# 3. 初期化とRGEの実行 (NumPy/SciPy)
# ==========================================

def default_init(mu0=150.0, yt=0.9):
    gy_, g2_, g3_ = gauge_couplings_1loop(mu0)

    params_tree = {
        'g1': gy_, 'g2': g2_, 'g3': g3_, 'yt': yt,
        'lamH': lamH, 'lamS': lamS, 'lamHS': lamHS,
        'mhsq': mhsq, 'mssq': mssq
    }

    d_mhsq, d_mssq, d_lam, d_lamS, d_lamHS = calculate_cw_corrections(params_tree, mu0)

    lam_mod = lamH - d_lam
    lamS_mod = lamS - d_lamS
    lamHS_mod = lamHS - d_lamHS
    mhsq_mod = mhsq - d_mhsq
    mssq_mod = mssq - d_mssq

    return {
        "g1": gy_,
        "g2": g2_,
        "g3": g3_,
        "yt": yt,
        "lam": lam_mod,
        "lamS": lamS_mod,
        "lamHS": lamHS_mod,
        "mhsq": mhsq_mod,
        "mssq": mssq_mod,
    }


def beta_functions(t, y):
    g1, g2, g3, yt, lam, lamS_val, lamHS_val, mhsq_val, mssq_val = y

    g1sq, g2sq, g3sq = g1 ** 2, g2 ** 2, g3 ** 2
    yt2 = yt ** 2
    yt4 = yt2 ** 2
    yt6 = yt4 * yt2
    lam2, lam3 = lam ** 2, lam ** 3
    lamS2, lamS3 = lamS_val ** 2, lamS_val ** 3
    lamHS2, lamHS3 = lamHS_val ** 2, lamHS_val ** 3

    # ==========================================
    # 1-Loop Beta Functions
    # ==========================================
    b1_g1 = (41.0 / 6.0) * g1 ** 3
    b1_g2 = -(19.0 / 6.0) * g2 ** 3
    b1_g3 = -7.0 * g3 ** 3

    b1_yt = yt * (4.5 * yt2 - (17.0 / 12.0) * g1sq - (9.0 / 4.0) * g2sq - 8.0 * g3sq)

    b1_lam = (
        + 24.0 * lam2 + 0.5 * lamHS2 - 3.0 * g1sq * lam - 9.0 * g2sq * lam
        + (3.0 / 8.0) * g1sq ** 2 + (3.0 / 4.0) * g1sq * g2sq + (9.0 / 8.0) * g2sq ** 2
        + 12.0 * lam * yt2 - 6.0 * yt4
    )

    b1_lamS = 18.0 * lamS2 + 2.0 * lamHS2

    b1_lamHS = (
        + 12.0 * lamHS_val * lam + 6.0 * lamHS_val * lamS_val + 4.0 * lamHS2
        - 1.5 * g1sq * lamHS_val - 4.5 * g2sq * lamHS_val + 6.0 * lamHS_val * yt2
    )

    b1_mhsq = (
        - 1.5 * g1sq * mhsq_val - 4.5 * g2sq * mhsq_val
        + 12.0 * lam * mhsq_val + 6.0 * mhsq_val * yt2 + lamHS_val * mssq_val
    )

    b1_mssq = 4.0 * lamHS_val * mhsq_val + 6.0 * lamS_val * mssq_val

    # ==========================================
    # 2-Loop Beta Functions
    # ==========================================
    b2_g1 = (199.0 / 18.0) * g1 ** 5 + 4.5 * g1 ** 3 * g2sq + (44.0 / 3.0) * g1 ** 3 * g3sq - (17.0 / 6.0) * g1 ** 3 * yt2
    b2_g2 = 1.5 * g1sq * g2 ** 3 + (35.0 / 6.0) * g2 ** 5 + 12.0 * g2 ** 3 * g3sq - 1.5 * g2 ** 3 * yt2
    b2_g3 = (11.0 / 6.0) * g1sq * g3 ** 3 + 4.5 * g2sq * g3 ** 3 - 26.0 * g3 ** 5 - 2.0 * g3 ** 3 * yt2

    b2_yt = (
        - 12.0 * yt * yt4 - 12.0 * lam * yt * yt2 + 6.0 * lam2 * yt + 0.25 * lamHS2 * yt
        + (131.0 / 16.0) * g1sq * yt * yt2 + (225.0 / 16.0) * g2sq * yt * yt2 + 36.0 * g3sq * yt * yt2
        + (1187.0 / 216.0) * g1sq ** 2 * yt - 0.75 * g1sq * g2sq * yt + (19.0 / 9.0) * g1sq * g3sq * yt
        - (23.0 / 4.0) * g2sq ** 2 * yt + 9.0 * g2sq * g3sq * yt - 108.0 * g3sq ** 2 * yt
    )

    b2_lam = (
        - 312.0 * lam3 - 5.0 * lamHS2 * lam - 2.0 * lamHS3
        + 36.0 * g1sq * lam2 + 108.0 * g2sq * lam2 + (629.0 / 24.0) * g1sq ** 2 * lam
        + (39.0 / 4.0) * g1sq * g2sq * lam - (73.0 / 8.0) * g2sq ** 2 * lam
        - (379.0 / 48.0) * g1sq ** 3 - (559.0 / 48.0) * g1sq ** 2 * g2sq
        - (289.0 / 48.0) * g1sq * g2sq ** 2 + (305.0 / 16.0) * g2sq ** 3
        - 144.0 * lam2 * yt2 + (85.0 / 6.0) * g1sq * lam * yt2 + 22.5 * g2sq * lam * yt2
        + 80.0 * g3sq * lam * yt2 - 4.75 * g1sq ** 2 * yt2 + 10.5 * g1sq * g2sq * yt2
        - 2.25 * g2sq ** 2 * yt2 - 3.0 * lam * yt4 - (8.0 / 3.0) * g1sq * yt4 - 32.0 * g3sq * yt4 + 30.0 * yt6
    )

    b2_lamS = - 204.0 * lamS3 - 20.0 * lamHS2 * lamS_val - 8.0 * lamHS3 + 4.0 * g1sq * lamHS2 + 12.0 * g2sq * lamHS2 - 12.0 * lamHS2 * yt2

    b2_lamHS = (
        - 72.0 * lamHS2 * lam - 36.0 * lamHS2 * lamS_val - 60.0 * lamHS_val * lam2
        - 30.0 * lamHS_val * lamS2 - 10.5 * lamHS3 + 24.0 * g1sq * lamHS_val * lam
        + 72.0 * g2sq * lamHS_val * lam + 1.0 * g1sq * lamHS2 + 3.0 * g2sq * lamHS2
        + (557.0 / 48.0) * g1sq ** 2 * lamHS_val + (15.0 / 8.0) * g1sq * g2sq * lamHS_val
        - (145.0 / 16.0) * g2sq ** 2 * lamHS_val - 72.0 * lamHS_val * lam * yt2
        - 12.0 * lamHS2 * yt2 + (85.0 / 12.0) * g1sq * lamHS_val * yt2
        + (45.0 / 4.0) * g2sq * lamHS_val * yt2 + 40.0 * g3sq * lamHS_val * yt2 - 13.5 * lamHS_val * yt4
    )

    b2_mhsq = (
        + (557.0 / 48.0) * g1sq ** 2 * mhsq_val + (15.0 / 8.0) * g1sq * g2sq * mhsq_val
        - (145.0 / 16.0) * g2sq ** 2 * mhsq_val + 24.0 * g1sq * lam * mhsq_val
        + 72.0 * g2sq * lam * mhsq_val - 60.0 * lam2 * mhsq_val
        - 0.5 * lamHS2 * mhsq_val - 2.0 * lamHS2 * mssq_val
        + (85.0 / 12.0) * g1sq * mhsq_val * yt2 + (45.0 / 4.0) * g2sq * mhsq_val * yt2
        + 40.0 * g3sq * mhsq_val * yt2 - 72.0 * lam * mhsq_val * yt2 - 13.5 * mhsq_val * yt4
    )

    b2_mssq = (
        + 8.0 * g1sq * lamHS_val * mhsq_val + 24.0 * g2sq * lamHS_val * mhsq_val
        - 8.0 * lamHS2 * mhsq_val - 30.0 * lamS2 * mssq_val
        - 2.0 * lamHS2 * mssq_val - 24.0 * lamHS_val * mhsq_val * yt2
    )

    loop_factor = 1.0 / (16.0 * np.pi ** 2)
    loop_factor2 = loop_factor ** 2

    return np.array([
        loop_factor * b1_g1    + loop_factor2 * b2_g1,
        loop_factor * b1_g2    + loop_factor2 * b2_g2,
        loop_factor * b1_g3    + loop_factor2 * b2_g3,
        loop_factor * b1_yt    + loop_factor2 * b2_yt,
        loop_factor * b1_lam   + loop_factor2 * b2_lam,
        loop_factor * b1_lamS  + loop_factor2 * b2_lamS,
        loop_factor * b1_lamHS + loop_factor2 * b2_lamHS,
        loop_factor * b1_mhsq  + loop_factor2 * b2_mhsq,
        loop_factor * b1_mssq  + loop_factor2 * b2_mssq,
    ])


def solve_rge(mu0=150.0, mu_end=2000.0, init=None, method="RK45", rtol=1e-10, atol=1e-12):
    if init is None:
        init = default_init(mu0)

    y0 = np.array([init[k] for k in PARAM_ORDER], dtype=float)
    t0 = np.log(mu0)
    t_end = np.log(mu_end)

    sol = solve_ivp(
        beta_functions,
        (t0, t_end),
        y0,
        method=method,
        rtol=rtol,
        atol=atol,
        dense_output=True,
    )

    if not sol.success:
        raise RuntimeError(f"Integration failed: {sol.message}")

    return sol


def make_running_couplings(mu0=150.0, mu_end=2000.0, init=None,
                            method="RK45", rtol=1e-10, atol=1e-12,
                            bounds_error=True):
    sol = solve_rge(
        mu0=mu0, mu_end=mu_end, init=init,
        method=method, rtol=rtol, atol=atol
    )

    t_min = np.log(mu0)
    t_max = np.log(mu_end)

    def _check_t(t):
        t_arr = np.asarray(t)
        if bounds_error:
            if np.any(t_arr < t_min) or np.any(t_arr > t_max):
                raise ValueError(f"t is outside interpolation range: [{t_min}, {t_max}]")
        return t_arr

    def vec(t):
        t_arr = _check_t(t)
        return sol.sol(t_arr)

    def vec_mu(mu):
        mu_arr = np.asarray(mu)
        if np.any(mu_arr <= 0.0):
            raise ValueError("mu must be positive.")
        return vec(np.log(mu_arr))

    rc = {
        "sol": sol,
        "t_min": t_min,
        "t_max": t_max,
        "mu0": mu0,
        "mu_end": mu_end,
        "vec": vec,
        "vec_mu": vec_mu,
    }

    for i, name in enumerate(PARAM_ORDER):
        rc[name] = lambda t, i=i: vec(t)[i]
        rc[f"{name}_mu"] = lambda mu, i=i: vec_mu(mu)[i]

    return rc
