#!/usr/bin/env python3
#!/usr/bin/env python3

import numpy as np
import torch
from scipy.integrate import solve_ivp

# config_params から必要なパラメータをすべてインポート
from .gauge_one_loop import gauge_couplings_1loop
from .config_params import lamH, lamS, lamHS, mhsq, mssq, vew

PARAM_ORDER = ["g1", "g2", "g3", "yt", "lam", "lamS", "lamHS", "mhsq", "mssq"]

# 計算用の微小値と定数
eps = 1e-10
NGS = 3.0  # Goldstoneボソンの自由度
C = 1.0

# ==========================================
# 1. 質量行列とCWポテンシャルの定義 (PyTorch)
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

    # Bare masses
    mG2_b = u_rho
    M11_b = u_rho + 2.0 * rho_phys * u_rhorho
    M22_b = u_sigma + 2.0 * sigma_phys * u_sigmasigma
    
    clamp_val = torch.clamp(rho_phys * sigma_phys, min=eps)
    M12_b = 2.0 * torch.sqrt(clamp_val) * u_rhosigma

    disc_b = (M11_b - M22_b)**2 + 4.0 * M12_b**2
    sqrt_disc_b = torch.sqrt(torch.clamp(disc_b, min=eps))

    m1_sq_b = 0.5 * (M11_b + M22_b - sqrt_disc_b)
    m2_sq_b = 0.5 * (M11_b + M22_b + sqrt_disc_b)

    return mG2_b, m1_sq_b, m2_sq_b

def gauge_masses_bare(rho_phys, params):
    """
    ゲージボソンの裸の質量を計算します。
    """
    g1 = params['g1']
    g2 = params['g2']

    mW_T2 = 0.5 * g2**2 * rho_phys
    mZ_T2 = 0.5 * (g2**2 + g1**2) * rho_phys
    mA_T2 = torch.zeros_like(rho_phys)

    return mW_T2, mZ_T2, mA_T2

def u_CW_zeroT(rho_phys, sigma_phys, params, kt):
    """
    T=0 における 1-loop Coleman-Weinberg ポテンシャル (無次元)。
    """
    mG2_b, m1_sq_b, m2_sq_b = scalar_masses_bare(rho_phys, sigma_phys, params, kt)
    mW_T2, mZ_T2, mA_T2 = gauge_masses_bare(rho_phys, params)

    yt = params['yt']
    mt2 = yt**2 * rho_phys

    def cw_term(m2, dof, c):
        m2_abs = torch.abs(m2) + eps
        return (dof / (64.0 * torch.pi**2)) * (m2**2) * (torch.log(m2_abs) - c)

    # Calculate CW contributions.
    # The Goldstone contribution is excluded here: at the tree-level EW
    # point mG2_b is exactly zero by construction, so its curvature
    # d^2(cw_G)/dsigma^2 hits the Goldstone-boson-catastrophe log
    # divergence (Elias-Miro, Espinosa, Konstandin, JHEP 1408:034 (2014);
    # Martin, PRD 90:016013 (2014)) and the regularized value is highly
    # sensitive to the ad hoc cutoff `eps` above, not a converged physical
    # number. The other loops (m1, m2, W, Z, top) are all evaluated at
    # genuinely nonzero tree masses and are unaffected. The Goldstone is
    # still included as usual in the finite-temperature potential
    # (thermal_functions.py), where its thermal Debye mass regulates it.
    cw_1 = cw_term(m1_sq_b, 1.0, 1.5)
    cw_2 = cw_term(m2_sq_b, 1.0, 1.5)

    cw_W = cw_term(mW_T2, 6.0, 5.0 / 6.0)
    cw_Z = cw_term(mZ_T2, 3.0, 5.0 / 6.0)

    # Top Quark (Fermion -> negative sign)
    cw_top = cw_term(mt2, -12.0, 1.5)

    return cw_1 + cw_2 + cw_W + cw_Z + cw_top

# ==========================================
# 2. CW補正の抽出 (Autogradによる自動微分)
# ==========================================

def calculate_cw_corrections(params, kt):
    """
    CWポテンシャルの微分を計算し、各パラメータへの補正値を返します。
    """
    rho_dimful = vew**2
    sigma_dimful = 0.0  # Singlet側のVEVは0と仮定

    # 無次元化してPyTorchテンソル化 (勾配計算を有効にする)
    rho_tilde = torch.tensor(rho_dimful / (kt**2), requires_grad=True, dtype=torch.float64)
    sigma_tilde = torch.tensor(sigma_dimful / (kt**2), requires_grad=True, dtype=torch.float64)

    # 無次元CWポテンシャル
    u_cw = u_CW_zeroT(rho_tilde, sigma_tilde, params, kt)

    # 1次微分 (mu_H^2, mu_S^2 への寄与)
    grads = torch.autograd.grad(u_cw, (rho_tilde, sigma_tilde), create_graph=True)
    du_drho = grads[0]
    du_dsigma = grads[1]

    # 2次微分 (lamH, lamS, lamHS への寄与)
    d2u_drho2 = torch.autograd.grad(du_drho, rho_tilde, retain_graph=True)[0]
    d2u_dsigma2 = torch.autograd.grad(du_dsigma, sigma_tilde, retain_graph=True)[0]
    d2u_drhodsigma = torch.autograd.grad(du_drho, sigma_tilde, retain_graph=True)[0]

    # 次元を復元して物理補正値へ変換 (V_cw = k^4 * u_cw, rho = k^2 * rho_tilde)
    delta_mhsq = (kt**2) * du_drho.item()
    delta_mssq = (kt**2) * du_dsigma.item()
    delta_lamH = 0.5 * d2u_drho2.item()
    delta_lamS = 0.5 * d2u_dsigma2.item()
    delta_lamHS = d2u_drhodsigma.item()

    return delta_mhsq, delta_mssq, delta_lamH, delta_lamS, delta_lamHS

# ==========================================
# 3. 初期化とRGEの実行 (NumPy/SciPy)
# ==========================================

def default_init(mu0=150.0, yt=0.9):
    """
    【修正】 引数から lamS_val などを排除し、config_params のインポート値を直接使用
    """
    gy_, g2_, g3_ = gauge_couplings_1loop(mu0)
    
    # config_params から読み込んだ物理パラメータをセット
    params_tree = {
        'g1': gy_, 'g2': g2_, 'g3': g3_, 'yt': yt,
        'lamH': lamH, 'lamS': lamS, 'lamHS': lamHS,
        'mhsq': mhsq, 'mssq': mssq
    }

    # CW補正量の算出
    d_mhsq, d_mssq, d_lam, d_lamS, d_lamHS = calculate_cw_corrections(params_tree, mu0)

    # 有効ポテンシャルにおいて物理的観測量に合わせるため、
    # 裸のパラメータ(初期値)からCW寄与分を引き去ります。
    lam_mod   = lamH - d_lam
    lamS_mod  = lamS - d_lamS
    lamHS_mod = lamHS - d_lamHS
    mhsq_mod  = mhsq - d_mhsq
    mssq_mod  = mssq - d_mssq
    
    #print(mhsq,mhsq_mod)
    #print(lam_mod,lamH)
    #print(lamHS_mod,lamHS)
    #print(mssq,mssq_mod)

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

    # --- よく使うべき乗の事前計算 ---
    g1sq, g2sq, g3sq = g1**2, g2**2, g3**2
    yt2 = yt**2
    yt4 = yt2**2
    yt6 = yt4 * yt2
    lam2, lam3 = lam**2, lam**3
    lamS2, lamS3 = lamS_val**2, lamS_val**3
    lamHS2, lamHS3 = lamHS_val**2, lamHS_val**3

    # ==========================================
    # 1-Loop Beta Functions
    # ==========================================
    b1_g1 = (41.0 / 6.0) * g1**3
    b1_g2 = -(19.0 / 6.0) * g2**3
    b1_g3 = -7.0 * g3**3

    b1_yt = yt * (4.5 * yt2 - (17.0 / 12.0) * g1sq - (9.0 / 4.0) * g2sq - 8.0 * g3sq)

    b1_lam = (
        + 24.0 * lam2 + 0.5 * lamHS2 - 3.0 * g1sq * lam - 9.0 * g2sq * lam
        + (3.0 / 8.0) * g1sq**2 + (3.0 / 4.0) * g1sq * g2sq + (9.0 / 8.0) * g2sq**2
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
    b2_g1 = (199.0 / 18.0) * g1**5 + 4.5 * g1**3 * g2sq + (44.0 / 3.0) * g1**3 * g3sq - (17.0 / 6.0) * g1**3 * yt2
    b2_g2 = 1.5 * g1sq * g2**3 + (35.0 / 6.0) * g2**5 + 12.0 * g2**3 * g3sq - 1.5 * g2**3 * yt2
    b2_g3 = (11.0 / 6.0) * g1sq * g3**3 + 4.5 * g2sq * g3**3 - 26.0 * g3**5 - 2.0 * g3**3 * yt2

    b2_yt = (
        - 12.0 * yt * yt4 - 12.0 * lam * yt * yt2 + 6.0 * lam2 * yt + 0.25 * lamHS2 * yt
        + (131.0 / 16.0) * g1sq * yt * yt2 + (225.0 / 16.0) * g2sq * yt * yt2 + 36.0 * g3sq * yt * yt2
        + (1187.0 / 216.0) * g1sq**2 * yt - 0.75 * g1sq * g2sq * yt + (19.0 / 9.0) * g1sq * g3sq * yt
        - (23.0 / 4.0) * g2sq**2 * yt + 9.0 * g2sq * g3sq * yt - 108.0 * g3sq**2 * yt
    )

    b2_lam = (
        - 312.0 * lam3 - 5.0 * lamHS2 * lam - 2.0 * lamHS3
        + 36.0 * g1sq * lam2 + 108.0 * g2sq * lam2 + (629.0 / 24.0) * g1sq**2 * lam
        + (39.0 / 4.0) * g1sq * g2sq * lam - (73.0 / 8.0) * g2sq**2 * lam
        - (379.0 / 48.0) * g1sq**3 - (559.0 / 48.0) * g1sq**2 * g2sq
        - (289.0 / 48.0) * g1sq * g2sq**2 + (305.0 / 16.0) * g2sq**3
        - 144.0 * lam2 * yt2 + (85.0 / 6.0) * g1sq * lam * yt2 + 22.5 * g2sq * lam * yt2
        + 80.0 * g3sq * lam * yt2 - 4.75 * g1sq**2 * yt2 + 10.5 * g1sq * g2sq * yt2
        - 2.25 * g2sq**2 * yt2 - 3.0 * lam * yt4 - (8.0 / 3.0) * g1sq * yt4 - 32.0 * g3sq * yt4 + 30.0 * yt6
    )

    b2_lamS = - 204.0 * lamS3 - 20.0 * lamHS2 * lamS_val - 8.0 * lamHS3 + 4.0 * g1sq * lamHS2 + 12.0 * g2sq * lamHS2 - 12.0 * lamHS2 * yt2

    b2_lamHS = (
        - 72.0 * lamHS2 * lam - 36.0 * lamHS2 * lamS_val - 60.0 * lamHS_val * lam2
        - 30.0 * lamHS_val * lamS2 - 10.5 * lamHS3 + 24.0 * g1sq * lamHS_val * lam
        + 72.0 * g2sq * lamHS_val * lam + 1.0 * g1sq * lamHS2 + 3.0 * g2sq * lamHS2
        + (557.0 / 48.0) * g1sq**2 * lamHS_val + (15.0 / 8.0) * g1sq * g2sq * lamHS_val
        - (145.0 / 16.0) * g2sq**2 * lamHS_val - 72.0 * lamHS_val * lam * yt2
        - 12.0 * lamHS2 * yt2 + (85.0 / 12.0) * g1sq * lamHS_val * yt2
        + (45.0 / 4.0) * g2sq * lamHS_val * yt2 + 40.0 * g3sq * lamHS_val * yt2 - 13.5 * lamHS_val * yt4
    )

    b2_mhsq = (
        + (557.0 / 48.0) * g1sq**2 * mhsq_val + (15.0 / 8.0) * g1sq * g2sq * mhsq_val
        - (145.0 / 16.0) * g2sq**2 * mhsq_val + 24.0 * g1sq * lam * mhsq_val
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

    # ==========================================
    # Combine 1-Loop and 2-Loop
    # ==========================================
    loop_factor = 1.0 / (16.0 * np.pi**2)
    loop_factor2 = loop_factor**2

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

'''

import numpy as np
from scipy.integrate import solve_ivp
from .gauge_one_loop import gauge_couplings_1loop
from .config_params import lamH, lamS, lamHS, mhsq, mssq

PARAM_ORDER = ["g1", "g2", "g3", "yt", "lam", "lamS", "lamHS", "mhsq", "mssq"]

def default_init(mu0=150.0, yt=0.9, lam=0.13, lamS_val=1.0, lamHS_val=0.5):
    gy_, g2_, g3_ = gauge_couplings_1loop(mu0)
    return {
        "g1": gy_,
        "g2": g2_,
        "g3": g3_,
        "yt": yt,
        "lam": lamH,
        "lamS": lamS,
        "lamHS": lamHS,
        "mhsq": mhsq,
        "mssq": mssq,
    }

def beta_functions(t, y):
    g1, g2, g3, yt, lam, lamS_val, lamHS_val, mhsq_val, mssq_val = y

    # --- よく使うべき乗の事前計算 ---
    g1sq = g1**2
    g2sq = g2**2
    g3sq = g3**2
    
    yt2 = yt**2
    yt4 = yt2**2
    yt6 = yt4 * yt2
    
    lam2 = lam**2
    lam3 = lam**3
    lamS2 = lamS_val**2
    lamS3 = lamS_val**3
    lamHS2 = lamHS_val**2
    lamHS3 = lamHS_val**3

    # ==========================================
    # 1-Loop Beta Functions
    # ==========================================
    b1_g1 = (41.0 / 6.0) * g1**3
    b1_g2 = -(19.0 / 6.0) * g2**3
    b1_g3 = -7.0 * g3**3

    b1_yt = yt * (
        + 4.5 * yt2
        - (17.0 / 12.0) * g1sq
        - (9.0 / 4.0) * g2sq
        - 8.0 * g3sq
    )

    b1_lam = (
        + 24.0 * lam2
        + 0.5 * lamHS2
        - 3.0 * g1sq * lam
        - 9.0 * g2sq * lam
        + (3.0 / 8.0) * g1sq**2
        + (3.0 / 4.0) * g1sq * g2sq
        + (9.0 / 8.0) * g2sq**2
        + 12.0 * lam * yt2
        - 6.0 * yt4
    )

    b1_lamS = (
        + 18.0 * lamS2
        + 2.0 * lamHS2
    )

    b1_lamHS = (
        + 12.0 * lamHS_val * lam
        + 6.0 * lamHS_val * lamS_val
        + 4.0 * lamHS2
        - 1.5 * g1sq * lamHS_val
        - 4.5 * g2sq * lamHS_val
        + 6.0 * lamHS_val * yt2
    )

    b1_mhsq = (
        - 1.5 * g1sq * mhsq_val
        - 4.5 * g2sq * mhsq_val
        + 12.0 * lam * mhsq_val
        + 6.0 * mhsq_val * yt2
        + lamHS_val * mssq_val
    )

    b1_mssq = (
        + 4.0 * lamHS_val * mhsq_val
        + 6.0 * lamS_val * mssq_val
    )

    # ==========================================
    # 2-Loop Beta Functions
    # ==========================================
    b2_g1 = (
        + (199.0 / 18.0) * g1**5
        + 4.5 * g1**3 * g2sq
        + (44.0 / 3.0) * g1**3 * g3sq
        - (17.0 / 6.0) * g1**3 * yt2
    )
    
    b2_g2 = (
        + 1.5 * g1sq * g2**3
        + (35.0 / 6.0) * g2**5
        + 12.0 * g2**3 * g3sq
        - 1.5 * g2**3 * yt2
    )
    
    b2_g3 = (
        + (11.0 / 6.0) * g1sq * g3**3
        + 4.5 * g2sq * g3**3
        - 26.0 * g3**5
        - 2.0 * g3**3 * yt2
    )

    b2_yt = (
        - 12.0 * yt * yt4
        - 12.0 * lam * yt * yt2
        + 6.0 * lam2 * yt
        + 0.25 * lamHS2 * yt
        + (131.0 / 16.0) * g1sq * yt * yt2
        + (225.0 / 16.0) * g2sq * yt * yt2
        + 36.0 * g3sq * yt * yt2
        + (1187.0 / 216.0) * g1sq**2 * yt
        - 0.75 * g1sq * g2sq * yt
        + (19.0 / 9.0) * g1sq * g3sq * yt
        - (23.0 / 4.0) * g2sq**2 * yt
        + 9.0 * g2sq * g3sq * yt
        - 108.0 * g3sq**2 * yt
    )

    b2_lam = (
        - 312.0 * lam3
        - 5.0 * lamHS2 * lam
        - 2.0 * lamHS3
        + 36.0 * g1sq * lam2
        + 108.0 * g2sq * lam2
        + (629.0 / 24.0) * g1sq**2 * lam
        + (39.0 / 4.0) * g1sq * g2sq * lam
        - (73.0 / 8.0) * g2sq**2 * lam
        - (379.0 / 48.0) * g1sq**3
        - (559.0 / 48.0) * g1sq**2 * g2sq
        - (289.0 / 48.0) * g1sq * g2sq**2
        + (305.0 / 16.0) * g2sq**3
        - 144.0 * lam2 * yt2
        + (85.0 / 6.0) * g1sq * lam * yt2
        + 22.5 * g2sq * lam * yt2
        + 80.0 * g3sq * lam * yt2
        - 4.75 * g1sq**2 * yt2
        + 10.5 * g1sq * g2sq * yt2
        - 2.25 * g2sq**2 * yt2
        - 3.0 * lam * yt4
        - (8.0 / 3.0) * g1sq * yt4
        - 32.0 * g3sq * yt4
        + 30.0 * yt6
    )

    b2_lamS = (
        - 204.0 * lamS3
        - 20.0 * lamHS2 * lamS_val
        - 8.0 * lamHS3
        + 4.0 * g1sq * lamHS2
        + 12.0 * g2sq * lamHS2
        - 12.0 * lamHS2 * yt2
    )

    b2_lamHS = (
        - 72.0 * lamHS2 * lam
        - 36.0 * lamHS2 * lamS_val
        - 60.0 * lamHS_val * lam2
        - 30.0 * lamHS_val * lamS2
        - 10.5 * lamHS3
        + 24.0 * g1sq * lamHS_val * lam
        + 72.0 * g2sq * lamHS_val * lam
        + 1.0 * g1sq * lamHS2
        + 3.0 * g2sq * lamHS2
        + (557.0 / 48.0) * g1sq**2 * lamHS_val
        + (15.0 / 8.0) * g1sq * g2sq * lamHS_val
        - (145.0 / 16.0) * g2sq**2 * lamHS_val
        - 72.0 * lamHS_val * lam * yt2
        - 12.0 * lamHS2 * yt2
        + (85.0 / 12.0) * g1sq * lamHS_val * yt2
        + (45.0 / 4.0) * g2sq * lamHS_val * yt2
        + 40.0 * g3sq * lamHS_val * yt2
        - 13.5 * lamHS_val * yt4
    )

    b2_mhsq = (
        + (557.0 / 48.0) * g1sq**2 * mhsq_val
        + (15.0 / 8.0) * g1sq * g2sq * mhsq_val
        - (145.0 / 16.0) * g2sq**2 * mhsq_val
        + 24.0 * g1sq * lam * mhsq_val
        + 72.0 * g2sq * lam * mhsq_val
        - 60.0 * lam2 * mhsq_val
        - 0.5 * lamHS2 * mhsq_val
        - 2.0 * lamHS2 * mssq_val
        + (85.0 / 12.0) * g1sq * mhsq_val * yt2
        + (45.0 / 4.0) * g2sq * mhsq_val * yt2
        + 40.0 * g3sq * mhsq_val * yt2
        - 72.0 * lam * mhsq_val * yt2
        - 13.5 * mhsq_val * yt4
    )

    b2_mssq = (
        + 8.0 * g1sq * lamHS_val * mhsq_val
        + 24.0 * g2sq * lamHS_val * mhsq_val
        - 8.0 * lamHS2 * mhsq_val
        - 30.0 * lamS2 * mssq_val
        - 2.0 * lamHS2 * mssq_val
        - 24.0 * lamHS_val * mhsq_val * yt2
    )

    # ==========================================
    # Combine 1-Loop and 2-Loop
    # ==========================================
    loop_factor = 1.0 / (16.0 * np.pi**2)
    loop_factor2 = loop_factor**2

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


def solve_rge(mu0=150.0, mu_end=2000.0, init=None,
              method="RK45", rtol=1e-10, atol=1e-12):
    """
    Solve the RGE once and return the raw solve_ivp solution object.
    """
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
    """
    Return callable interpolating functions for running couplings.

    Returns
    -------
    rc : dict
        rc["g1"](t), rc["yt"](t), ..., rc["mhsq"](t), rc["mssq"](t)
        rc["g1_mu"](mu), rc["yt_mu"](mu), ..., rc["mhsq_mu"](mu), rc["mssq_mu"](mu)
        rc["vec"](t)        -> array([g1, g2, g3, yt, lam, lamS, lamHS, mhsq, mssq])
        rc["vec_mu"](mu)    -> same but using mu
        rc["t_min"], rc["t_max"], rc["mu0"], rc["mu_end"], rc["sol"]
    """
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
                raise ValueError(
                    f"t is outside interpolation range: [{t_min}, {t_max}]"
                )
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

    # PARAM_ORDER に追加した mhsq, mssq も自動的に callable として生成されます
    for i, name in enumerate(PARAM_ORDER):
        rc[name] = lambda t, i=i: vec(t)[i]
        rc[f"{name}_mu"] = lambda mu, i=i: vec_mu(mu)[i]

    return rc
'''