"""
frg_discrete/perturbation/rges.py + frg_discrete/running_couplings.py の
singlet除去版(6変数 [g1,g2,g3,yt,lam,mhsq] のみ)。

frg_pinns_higgs_only/perturbation/rges.py で検証済みの縮約と同じ
beta functions を移植したもの(lamHS=0 を恒等的に課すのと数値的に同値 —
b1_lamHS, b2_lamHS は全項が lamHS に比例するため)。

CW補正 (calculate_cw_corrections) は空間1方向 (rho) だけなので、共有版が
torch自動微分と併用しているnumpy中心差分フォールバックのみで十分
(precision上の問題はない)。
"""

import numpy as np
from scipy.integrate import solve_ivp

from perturbation.config_params import k_IR, t_range, lamH, mhsq, vew
from perturbation.gauge_one_loop import gauge_couplings_1loop

PARAM_ORDER = ["g1", "g2", "g3", "yt", "lam", "mhsq"]

eps = 1e-10


# ==========================================
# 1. 質量とCWポテンシャル (Higgs単一チャンネル、NumPy)
# ==========================================

def scalar_masses_bare(rho_phys, params, kt):
    lamH_t = params['lamH']
    muH2_t = params['mhsq'] / (kt * kt)

    u_rho = muH2_t + 2.0 * lamH_t * rho_phys
    u_rhorho = 2.0 * lamH_t

    mG2_b = u_rho
    mH2_b = u_rho + 2.0 * rho_phys * u_rhorho
    return mG2_b, mH2_b


def gauge_masses_bare(rho_phys, params):
    g1 = params['g1']
    g2 = params['g2']
    mW_T2 = 0.5 * g2 ** 2 * rho_phys
    mZ_T2 = 0.5 * (g2 ** 2 + g1 ** 2) * rho_phys
    mA_T2 = np.zeros_like(rho_phys)
    return mW_T2, mZ_T2, mA_T2


def u_CW_zeroT(rho_phys, params, kt):
    mG2_b, mH2_b = scalar_masses_bare(rho_phys, params, kt)
    mW_T2, mZ_T2, mA_T2 = gauge_masses_bare(rho_phys, params)
    yt = params['yt']
    mt2 = yt ** 2 * rho_phys

    def cw_term(m2, dof, c):
        m2_abs = np.abs(m2) + eps
        return (dof / (64.0 * np.pi ** 2)) * (m2 ** 2) * (np.log(m2_abs) - c)

    # Goldstone (cw_G) は matching sum から除外 (tree-level EW点でmG2=0
    # ちょうどになり curvature が Goldstone-boson-catastrophe log発散に
    # 支配されるため; frg_pinns_higgs_only/perturbation/rges.py と同じ扱い)
    cw_H = cw_term(mH2_b, 1.0, 1.5)
    cw_W = cw_term(mW_T2, 6.0, 5.0 / 6.0)
    cw_Z = cw_term(mZ_T2, 3.0, 5.0 / 6.0)
    cw_top = cw_term(mt2, -12.0, 1.5)
    return cw_H + cw_W + cw_Z + cw_top


def calculate_cw_corrections(params, kt):
    """rho方向1本の数値中心差分でCW補正 (delta_mhsq, delta_lamH) を求める。"""
    rho0 = vew ** 2 / (kt ** 2)
    h = 1e-4 * max(rho0, 1.0)

    def f(rho):
        return u_CW_zeroT(np.array([rho]), params, kt)[0]

    f0 = f(rho0)
    du_drho = (f(rho0 + h) - f(rho0 - h)) / (2.0 * h)
    d2u_drho2 = (f(rho0 + h) - 2.0 * f0 + f(rho0 - h)) / h ** 2

    delta_mhsq = (kt ** 2) * du_drho
    delta_lamH = 0.5 * d2u_drho2
    return delta_mhsq, delta_lamH


# ==========================================
# 2. 初期化とRGEの実行
# ==========================================

def default_init(mu0=150.0, yt=0.9):
    gy_, g2_, g3_ = gauge_couplings_1loop(mu0)
    params_tree = {'g1': gy_, 'g2': g2_, 'g3': g3_, 'yt': yt, 'lamH': lamH, 'mhsq': mhsq}

    d_mhsq, d_lam = calculate_cw_corrections(params_tree, mu0)

    lam_mod = lamH - d_lam
    mhsq_mod = mhsq - d_mhsq

    return {"g1": gy_, "g2": g2_, "g3": g3_, "yt": yt, "lam": lam_mod, "mhsq": mhsq_mod}


def beta_functions(t, y):
    g1, g2, g3, yt, lam, mhsq_val = y

    g1sq, g2sq, g3sq = g1 ** 2, g2 ** 2, g3 ** 2
    yt2 = yt ** 2
    yt4 = yt2 ** 2
    yt6 = yt4 * yt2
    lam2, lam3 = lam ** 2, lam ** 3

    b1_g1 = (41.0 / 6.0) * g1 ** 3
    b1_g2 = -(19.0 / 6.0) * g2 ** 3
    b1_g3 = -7.0 * g3 ** 3

    b1_yt = yt * (4.5 * yt2 - (17.0 / 12.0) * g1sq - (9.0 / 4.0) * g2sq - 8.0 * g3sq)

    b1_lam = (
        + 24.0 * lam2 - 3.0 * g1sq * lam - 9.0 * g2sq * lam
        + (3.0 / 8.0) * g1sq ** 2 + (3.0 / 4.0) * g1sq * g2sq + (9.0 / 8.0) * g2sq ** 2
        + 12.0 * lam * yt2 - 6.0 * yt4
    )

    b1_mhsq = (
        - 1.5 * g1sq * mhsq_val - 4.5 * g2sq * mhsq_val
        + 12.0 * lam * mhsq_val + 6.0 * mhsq_val * yt2
    )

    b2_g1 = (199.0 / 18.0) * g1 ** 5 + 4.5 * g1 ** 3 * g2sq + (44.0 / 3.0) * g1 ** 3 * g3sq - (17.0 / 6.0) * g1 ** 3 * yt2
    b2_g2 = 1.5 * g1sq * g2 ** 3 + (35.0 / 6.0) * g2 ** 5 + 12.0 * g2 ** 3 * g3sq - 1.5 * g2 ** 3 * yt2
    b2_g3 = (11.0 / 6.0) * g1sq * g3 ** 3 + 4.5 * g2sq * g3 ** 3 - 26.0 * g3 ** 5 - 2.0 * g3 ** 3 * yt2

    b2_yt = (
        - 12.0 * yt * yt4 - 12.0 * lam * yt * yt2 + 6.0 * lam2 * yt
        + (131.0 / 16.0) * g1sq * yt * yt2 + (225.0 / 16.0) * g2sq * yt * yt2 + 36.0 * g3sq * yt * yt2
        + (1187.0 / 216.0) * g1sq ** 2 * yt - 0.75 * g1sq * g2sq * yt + (19.0 / 9.0) * g1sq * g3sq * yt
        - (23.0 / 4.0) * g2sq ** 2 * yt + 9.0 * g2sq * g3sq * yt - 108.0 * g3sq ** 2 * yt
    )

    b2_lam = (
        - 312.0 * lam3
        + 36.0 * g1sq * lam2 + 108.0 * g2sq * lam2 + (629.0 / 24.0) * g1sq ** 2 * lam
        + (39.0 / 4.0) * g1sq * g2sq * lam - (73.0 / 8.0) * g2sq ** 2 * lam
        - (379.0 / 48.0) * g1sq ** 3 - (559.0 / 48.0) * g1sq ** 2 * g2sq
        - (289.0 / 48.0) * g1sq * g2sq ** 2 + (305.0 / 16.0) * g2sq ** 3
        - 144.0 * lam2 * yt2 + (85.0 / 6.0) * g1sq * lam * yt2 + 22.5 * g2sq * lam * yt2
        + 80.0 * g3sq * lam * yt2 - 4.75 * g1sq ** 2 * yt2 + 10.5 * g1sq * g2sq * yt2
        - 2.25 * g2sq ** 2 * yt2 - 3.0 * lam * yt4 - (8.0 / 3.0) * g1sq * yt4 - 32.0 * g3sq * yt4 + 30.0 * yt6
    )

    b2_mhsq = (
        + (557.0 / 48.0) * g1sq ** 2 * mhsq_val + (15.0 / 8.0) * g1sq * g2sq * mhsq_val
        - (145.0 / 16.0) * g2sq ** 2 * mhsq_val + 24.0 * g1sq * lam * mhsq_val
        + 72.0 * g2sq * lam * mhsq_val - 60.0 * lam2 * mhsq_val
        + (85.0 / 12.0) * g1sq * mhsq_val * yt2 + (45.0 / 4.0) * g2sq * mhsq_val * yt2
        + 40.0 * g3sq * mhsq_val * yt2 - 72.0 * lam * mhsq_val * yt2 - 13.5 * mhsq_val * yt4
    )

    loop_factor = 1.0 / (16.0 * np.pi ** 2)
    loop_factor2 = loop_factor ** 2

    return np.array([
        loop_factor * b1_g1 + loop_factor2 * b2_g1,
        loop_factor * b1_g2 + loop_factor2 * b2_g2,
        loop_factor * b1_g3 + loop_factor2 * b2_g3,
        loop_factor * b1_yt + loop_factor2 * b2_yt,
        loop_factor * b1_lam + loop_factor2 * b2_lam,
        loop_factor * b1_mhsq + loop_factor2 * b2_mhsq,
    ])


def solve_rge(mu0=150.0, mu_end=2000.0, init=None, method="RK45", rtol=1e-10, atol=1e-12):
    if init is None:
        init = default_init(mu0)
    y0 = np.array([init[k] for k in PARAM_ORDER], dtype=float)
    t0 = np.log(mu0)
    t_end = np.log(mu_end)
    sol = solve_ivp(beta_functions, (t0, t_end), y0, method=method, rtol=rtol, atol=atol, dense_output=True)
    if not sol.success:
        raise RuntimeError(f"Integration failed: {sol.message}")
    return sol


def make_running_couplings(mu0=150.0, mu_end=2000.0, init=None, method="RK45", rtol=1e-10, atol=1e-12,
                            bounds_error=True):
    sol = solve_rge(mu0=mu0, mu_end=mu_end, init=init, method=method, rtol=rtol, atol=atol)
    t_min = np.log(mu0)
    t_max = np.log(mu_end)

    def _check_t(t):
        t_arr = np.asarray(t)
        if bounds_error:
            if np.any(t_arr < t_min) or np.any(t_arr > t_max):
                raise ValueError(f"t is outside interpolation range: [{t_min}, {t_max}]")
        return t_arr

    def vec(t):
        return sol.sol(_check_t(t))

    def vec_mu(mu):
        mu_arr = np.asarray(mu)
        if np.any(mu_arr <= 0.0):
            raise ValueError("mu must be positive.")
        return vec(np.log(mu_arr))

    rc = {"sol": sol, "t_min": t_min, "t_max": t_max, "mu0": mu0, "mu_end": mu_end, "vec": vec, "vec_mu": vec_mu}
    for i, name in enumerate(PARAM_ORDER):
        rc[name] = lambda t, i=i: vec(t)[i]
        rc[f"{name}_mu"] = lambda mu, i=i: vec_mu(mu)[i]
    return rc


# ==========================================
# 3. FRG UVマッチング (frg_discrete/running_couplings.py と同じ役割)
# ==========================================

rc = make_running_couplings(mu0=k_IR, mu_end=2000.0)

_tuv = 0.0
_trge = np.log(k_IR) + (-t_range) + _tuv

lamH_uv = float(rc["lam"](_trge))
_mhsq_uv = float(rc["mhsq"](_trge))

aH = (_mhsq_uv / k_IR ** 2) * np.exp(2.0 * t_range)

print('lamH, mhsq at UV scale (Higgs-only)', lamH_uv, _mhsq_uv)

tree_params = [aH, lamH_uv]


def _rge_t_from_phys_t(t_phys):
    t_phys_arr = np.asarray(t_phys, dtype=float)
    t_rge = np.log(k_IR) + (-t_range) + t_phys_arr
    t_min = rc["t_min"]
    t_rge = np.maximum(t_rge, t_min + 1e-8)
    return t_rge


def get_running_couplings(t_phys):
    """Return (g1, g2, yt) at the given FRG time(s) t_phys."""
    t_rge = _rge_t_from_phys_t(t_phys)
    return rc["g1"](t_rge), rc["g2"](t_rge), rc["yt"](t_rge)


def get_running_quartics(t_phys):
    """Return lamH at the given FRG time(s) t_phys."""
    t_rge = _rge_t_from_phys_t(t_phys)
    return rc["lam"](t_rge)


def get_running_masses(t_phys):
    """Return mhsq at the given FRG time(s) t_phys."""
    t_rge = _rge_t_from_phys_t(t_phys)
    return rc["mhsq"](t_rge)
