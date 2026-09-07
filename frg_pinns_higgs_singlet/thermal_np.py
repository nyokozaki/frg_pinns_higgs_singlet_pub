# %%
import numpy as np
import matplotlib.pyplot as plt

# 必要なモジュールとパラメータのインポート
from perturbation.config_params import tau_uv, finite_T, k_IR, t_range, fixed_tau 
from perturbation.rges import make_running_couplings
from perturbation.JBJF_helper import J_Bnp as J_B, J_Fnp as J_F

# ============================================================
# RGEのセットアップ
# ============================================================

rc = make_running_couplings(mu0=k_IR, mu_end=2000.0)

# ============================================================
# ヘルパー関数
# ============================================================

def _rge_t_from_phys_t(t_phys):
    t_phys_np = np.asarray(t_phys)
    t_flat = t_phys_np.reshape(-1)
    t_rge_flat = np.log(k_IR) + (-t_range) + t_flat
    t_min = rc["t_min"]
    t_rge_flat = np.maximum(t_rge_flat, t_min + 1e-8)
    return t_rge_flat, t_phys_np.shape

def get_running_couplings(t):
    t_rge_flat, orig_shape = _rge_t_from_phys_t(t)
    g1_   = rc["g1"](t_rge_flat).reshape(orig_shape)
    g2_   = rc["g2"](t_rge_flat).reshape(orig_shape)
    yt_   = rc["yt"](t_rge_flat).reshape(orig_shape)
    lam_  = rc["lam"](t_rge_flat).reshape(orig_shape)
    lamS_ = rc["lamS"](t_rge_flat).reshape(orig_shape)
    lamHS_= rc["lamHS"](t_rge_flat).reshape(orig_shape)
    mhsq_ = rc["mhsq"](t_rge_flat).reshape(orig_shape)
    mssq_ = rc["mssq"](t_rge_flat).reshape(orig_shape)
    return g1_, g2_, yt_, lam_, lamS_, lamHS_, mhsq_, mssq_

# ============================================================
# 質量とポテンシャルの定義 (Arnold-Espinosa方式)
# ============================================================

def scalar_masses_AE(rho_phys, sigma_phys, t_phys):
    """
    Arnold-Espinosa方式用：Bare質量と、デバイ熱質量補正を含んだEffective質量を両方返す
    """
    g1, g2, yt, lam_t, lamS_t, lamHS_t, mhsq_t, mssq_t = get_running_couplings(t_phys)
    if fixed_tau == False:
        tau_t = tau_uv * np.exp(-t_phys)
    else:
        base_value = tau_uv * np.exp(-t_range)
        tau_t = np.full_like(t_phys, base_value)
    
    kt = k_IR * np.exp(-t_range + t_phys)
    muH2_t = mhsq_t / kt**2
    muS2_t = mssq_t / kt**2

    # デバイ質量補正 (Pi)
    Pi_H = ( (3.0 * g2**2 + g1**2) / 16.0 
             + yt**2 / 4.0 
             + lam_t / 2.0 
             + lamHS_t / 24.0 ) * tau_t**2
             
    Pi_S = ( lamS_t / 4.0 
             + lamHS_t / 6.0 ) * tau_t**2

    muH2_eff = muH2_t + Pi_H
    muS2_eff = muS2_t + Pi_S

    C = 1.0  
    u_rhorho = (2.0 * lam_t) / C
    u_sigmasigma = (2.0 * lamS_t) / C
    u_rhosigma = (lamHS_t) / C
    eps = 1e-10

    # 1. Bare (Tree-level) Masses
    u_rho_b = (muH2_t + 2.0 * lam_t * rho_phys + lamHS_t * sigma_phys) / C
    u_sigma_b = (muS2_t + 2.0 * lamS_t * sigma_phys + lamHS_t * rho_phys) / C

    mG2_b = u_rho_b
    M11_b = u_rho_b + 2.0 * rho_phys * u_rhorho
    M22_b = u_sigma_b + 2.0 * sigma_phys * u_sigmasigma
    M12_b = 2.0 * np.sqrt(np.clip(rho_phys * sigma_phys, a_min=eps, a_max=None)) * u_rhosigma
    disc_b = (M11_b - M22_b)**2 + 4.0 * M12_b**2
    sqrt_disc_b = np.sqrt(np.clip(disc_b, a_min=eps, a_max=None))
    m1_sq_b = 0.5 * (M11_b + M22_b - sqrt_disc_b)
    m2_sq_b = 0.5 * (M11_b + M22_b + sqrt_disc_b)

    # 2. Effective Masses (with Pi)
    u_rho_e = (muH2_eff + 2.0 * lam_t * rho_phys + lamHS_t * sigma_phys) / C
    u_sigma_e = (muS2_eff + 2.0 * lamS_t * sigma_phys + lamHS_t * rho_phys) / C

    mG2_e = u_rho_e
    M11_e = u_rho_e + 2.0 * rho_phys * u_rhorho
    M22_e = u_sigma_e + 2.0 * sigma_phys * u_sigmasigma
    M12_e = 2.0 * np.sqrt(np.clip(rho_phys * sigma_phys, a_min=eps, a_max=None)) * u_rhosigma
    disc_e = (M11_e - M22_e)**2 + 4.0 * M12_e**2
    sqrt_disc_e = np.sqrt(np.clip(disc_e, a_min=eps, a_max=None))
    m1_sq_e = 0.5 * (M11_e + M22_e - sqrt_disc_e)
    m2_sq_e = 0.5 * (M11_e + M22_e + sqrt_disc_e)

    return (mG2_b, m1_sq_b, m2_sq_b), (mG2_e, m1_sq_e, m2_sq_e)

def gauge_masses_AE(rho_phys, t_phys):
    eps = 1e-10
    if fixed_tau == False:
        tau_t = tau_uv * np.exp(-t_phys)
    else:
        base_value = tau_uv * np.exp(-t_range)
        tau_t = np.full_like(t_phys, base_value)

    g1, g2, yt, _, _, _, _, _ = get_running_couplings(t_phys)

    # 1. Bare (Tree-level) Masses
    mW_2_b = 0.5 * g2**2 * rho_phys
    mZ_2_b = 0.5 * (g2**2 + g1**2) * rho_phys
    mA_2_b = np.zeros_like(rho_phys)

    # 2. Effective Longitudinal Masses (with Pi)
    Pi_W = (11.0 / 6.0) * g2**2 * tau_t**2
    Pi_B = (11.0 / 6.0) * g1**2 * tau_t**2

    mW_L2_e = mW_2_b + Pi_W
    M11_e = mW_2_b + Pi_W
    M22_e = 0.5 * g1**2 * rho_phys + Pi_B
    M12_e = -0.5 * g2 * g1 * rho_phys

    disc_e = (M11_e - M22_e)**2 + 4.0 * M12_e**2
    sqrt_disc_e = np.sqrt(np.clip(disc_e, a_min=eps, a_max=None))
    mZ_L2_e = 0.5 * (M11_e + M22_e + sqrt_disc_e)
    mA_L2_e = 0.5 * (M11_e + M22_e - sqrt_disc_e)

    return (mW_2_b, mZ_2_b, mA_2_b), (mW_L2_e, mZ_L2_e, mA_L2_e), g1, g2, yt, tau_t

def u_thermal_finiteT_AE(t_phys, rho_phys, sigma_phys):
    orig_shape = np.asarray(rho_phys).shape

    t_phys = np.asarray(t_phys).reshape(-1, 1)
    rho_phys_inner = np.asarray(rho_phys).reshape(-1, 1)
    sigma_phys = np.asarray(sigma_phys).reshape(-1, 1)

    # BareとEffectiveの質量を両方取得
    scalars_b, scalars_e = scalar_masses_AE(rho_phys_inner, sigma_phys, t_phys)
    mG2_b, m1_sq_b, m2_sq_b = scalars_b
    mG2_e, m1_sq_e, m2_sq_e = scalars_e

    gauges_b, gauges_e, g1, g2, yt, tau_t = gauge_masses_AE(rho_phys_inner, t_phys)
    mW_2_b, mZ_2_b, mA_2_b = gauges_b
    mW_L2_e, mZ_L2_e, mA_L2_e = gauges_e

    mt2 = yt**2 * rho_phys_inner

    eps = 1e-12
    tau2 = np.clip(tau_t**2, a_min=eps, a_max=None)
    tau4 = tau2 * tau2
    NGS = 3.0

    # ============================================================
    # 1-loop項: J_B, J_F にはデバイ質量を含まない Bare質量 を渡す
    # ============================================================
    JB_G = J_B(mG2_b / tau2)
    JB_1 = J_B(m1_sq_b / tau2)
    JB_2 = J_B(m2_sq_b / tau2)

    # W, ZはTransverseとLongitudinalで同じBare質量を持つため統合 (W: 6 dof, Z: 3 dof, A: 1 dof(longitudinal zero mode))
    JB_W = J_B(mW_2_b / tau2)
    JB_Z = J_B(mZ_2_b / tau2)
    JB_A = J_B(mA_2_b / tau2)
    
    JF_t = J_F(mt2 / tau2)

    u_th_scalar = (tau4 / (2.0 * np.pi**2)) * (NGS * JB_G + JB_1 + JB_2)
    u_th_gauge = (tau4 / (2.0 * np.pi**2)) * (6.0 * JB_W + 3.0 * JB_Z + 1.0 * JB_A)
    u_th_top = -(12.0 * tau4 / (2.0 * np.pi**2)) * JF_t

    u_th_1loop = u_th_scalar + u_th_gauge + u_th_top

    # ============================================================
    # リング補正項 (Arnold-Espinosa方式): 
    # ゼロモード (m^2)^{3/2} を (m^2 + Pi)^{3/2} に置き換える
    # ============================================================
    def get_m3(m2):
        # 虚数質量(m^2 < 0)はリング補正から除外するため 0 にクリップ
        return np.power(np.clip(m2, a_min=0.0, a_max=None), 1.5)

    ring_G = NGS * (get_m3(mG2_e) - get_m3(mG2_b))
    ring_1 = 1.0 * (get_m3(m1_sq_e) - get_m3(m1_sq_b))
    ring_2 = 1.0 * (get_m3(m2_sq_e) - get_m3(m2_sq_b))

    ring_WL = 2.0 * (get_m3(mW_L2_e) - get_m3(mW_2_b)) # W+, W- longitudinal
    ring_ZL = 1.0 * (get_m3(mZ_L2_e) - get_m3(mZ_2_b)) # Z longitudinal
    ring_AL = 1.0 * (get_m3(mA_L2_e) - get_m3(mA_2_b)) # Photon longitudinal

    # 係数は次元解析より -tau / (12 pi) になる
    u_ring = - (tau_t / (12.0 * np.pi)) * (ring_G + ring_1 + ring_2 + ring_WL + ring_ZL + ring_AL)

    u_th_total = u_th_1loop + u_ring

    return u_th_total.reshape(orig_shape)

def u_pert(t_phys, rho_phys, sigma_phys):
    g1_t, g2_t, yt_t, lam_t, lamS_t, lamHS_t, mhsq_t, mssq_t = get_running_couplings(t_phys)
    kt = k_IR * np.exp(-t_range + t_phys)
    
    muH2_t = mhsq_t / kt**2
    muS2_t = mssq_t / kt**2

    return (
        + muH2_t * rho_phys
        + lam_t * rho_phys**2
        + muS2_t * sigma_phys
        + lamS_t * sigma_phys**2
        + lamHS_t * rho_phys * sigma_phys
    )

def u_seed_finiteT(t_phys, rho_phys, sigma_phys):
    u_0 = u_pert(t_phys, rho_phys, sigma_phys)
    u_th = u_thermal_finiteT_AE(t_phys, rho_phys, sigma_phys) # AE方式を呼び出し
    u_total = u_0 + u_th
    
    # オフセットの計算
    rho_zero = np.zeros_like(rho_phys)
    sigma_zero = np.zeros_like(sigma_phys)
    
    u_0_offset = u_pert(t_phys, rho_zero, sigma_zero)
    u_th_offset = u_thermal_finiteT_AE(t_phys, rho_zero, sigma_zero)
    u_total_offset = u_0_offset + u_th_offset
    
    return u_total - u_total_offset

# ============================================================
# プロットの実行
# ============================================================
def plot():
    sigma_fixed = 0.0
    t_vals = [-2.0]

    rho_np = np.linspace(0.0, 2.0, 200) 
    sigma_np = np.full_like(rho_np, sigma_fixed)

    plt.figure(figsize=(10, 6))
    colors = ['blue', 'orange', 'red']

    for i, t_val in enumerate(t_vals):
        t_np = np.full_like(rho_np, t_val)
        
        u_pert_vals = u_pert(t_np, rho_np, sigma_np)
        u_pert_offset = u_pert(t_np, np.zeros_like(rho_np), np.zeros_like(sigma_np))
        u_pert_vals = u_pert_vals - u_pert_offset

        u_finiteT_vals = u_seed_finiteT(t_np, rho_np, sigma_np)
        
        plt.plot(rho_np, u_pert_vals, linestyle='--', color=colors[i], alpha=0.5, label=f'Zero-T $u_{{pert}}$ (t={t_val})')
        plt.plot(rho_np, u_finiteT_vals, linestyle='-', color=colors[i], label=f'Finite-T $u_{{seed}}$ (t={t_val})')

    plt.title(f"Thermal Potential vs Zero-T Potential ($\\sigma$ = {sigma_fixed}) \n*RG-improved mass & Arnold-Espinosa resummation*")
    plt.xlabel("$\\rho$")
    plt.ylabel("Potential")
    plt.legend(bbox_to_anchor=(1.05, 1), loc='upper left')
    plt.grid(True)
    plt.tight_layout()
    #plt.ylim(-0.002, 0.002)
    #plt.xlim(-0.01, 0.5)
    plt.show()

# %%



