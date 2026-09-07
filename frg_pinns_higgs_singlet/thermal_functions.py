import torch
import numpy as np
import torch.autograd as autograd

from perturbation.config_params import tau_uv, k_IR, t_range
from running_couplings import tree_params, get_running_couplings, get_running_masses, get_running_quartics
[aH,aS,lamH,lamS,lamHS] = tree_params

NGS=3.0
# ============================================================
# Finite-T helper: thermal integrals J_B, J_F
# argument: y2 = (m/T)^2 (both m and T are dimensionless here)
# ============================================================

from perturbation.JBJF_helper import get_leggauss_for, make_leggauss_constants, J_B, J_F

_DEG = 100
_GL_t_cpu, _GL_w_cpu = make_leggauss_constants(
    deg=_DEG,
    device="cpu",
    dtype=torch.float64,
    copy=True,   # 安全側。共有メモリを避ける
)

# ============================================================
# Finite-T seed potential u_seed_finiteT
#  - tree-level Higgs-singlet potential
#  - + finite-T thermal pieces:
#      * scalar: 2 mixed eigenstates (Higgs-singlet) + 3 Goldstones
#      * gauge bosons: W_T, W_L, Z_T, Z_L, gamma_L (Daisy for longitudinal)
#      * fermion: top quark
# All written in the same dimensionless variables as FRG:
#   rho = |H|^2 / k^2, sigma = S^2 / (2 k^2), tau = T/k
# ============================================================

def scalar_masses_bare_and_debye(rho_phys, sigma_phys, t_phys):
    """
    Return both bare scalar masses and Debye-improved scalar masses.

    Outputs:
      (mG2_b, m1_sq_b, m2_sq_b), (mG2_e, m1_sq_e, m2_sq_e)

    All masses are dimensionless m^2 in the FRG convention.
    """
    kt = k_IR * torch.exp(-t_range + t_phys)

    with torch.no_grad():
        g1_, g2_, yt_, _, _ = get_running_couplings(t_phys)
        lamH_t, lamS_t, lamHS_t = get_running_quartics(t_phys)
        _mhsq, _mssq = get_running_masses(t_phys)
        muH2_t = _mhsq / (kt * kt)
        muS2_t = _mssq / (kt * kt)

    #print(t_phys[0:1])
    #print(lamH_t,lamS_t)

    u_rho = (muH2_t + 2.0 * lamH_t * rho_phys + lamHS_t * sigma_phys) 
    u_sigma = (muS2_t + 2.0 * lamS_t * sigma_phys + lamHS_t * rho_phys)

    u_rhorho = (2.0 * lamH_t) 
    u_sigmasigma = (2.0 * lamS_t)
    u_rhosigma = (lamHS_t) 

    eps = 1e-10
    tau_t = tau_uv * torch.exp(-t_phys)

    Pi_H = (
        (3.0 * g2_**2 + g1_**2) / 16.0
        + yt_**2 / 4.0
        + lamH_t / 2.0
        + lamHS_t / 24.0
    ) * tau_t**2

    Pi_S = (
        lamS_t / 4.0
        + lamHS_t / 6.0
    ) * tau_t**2

    # -----------------------------
    # Bare masses
    # -----------------------------
    mG2_b = u_rho

    M11_b = u_rho + 2.0 * rho_phys * u_rhorho
    M22_b = u_sigma + 2.0 * sigma_phys * u_sigmasigma
    M12_b = 2.0 * torch.sqrt(torch.clamp(rho_phys * sigma_phys, min=eps)) * u_rhosigma

    disc_b = (M11_b - M22_b)**2 + 4.0 * M12_b**2
    sqrt_disc_b = torch.sqrt(torch.clamp(disc_b, min=eps))

    m1_sq_b = 0.5 * (M11_b + M22_b - sqrt_disc_b)
    m2_sq_b = 0.5 * (M11_b + M22_b + sqrt_disc_b)

    # -----------------------------
    # Debye-improved masses
    # -----------------------------
    mG2_e = u_rho + Pi_H

    M11_e = u_rho + 2.0 * rho_phys * u_rhorho + Pi_H
    M22_e = u_sigma + 2.0 * sigma_phys * u_sigmasigma + Pi_S
    M12_e = M12_b

    disc_e = (M11_e - M22_e)**2 + 4.0 * M12_e**2
    sqrt_disc_e = torch.sqrt(torch.clamp(disc_e, min=eps))

    m1_sq_e = 0.5 * (M11_e + M22_e - sqrt_disc_e)
    m2_sq_e = 0.5 * (M11_e + M22_e + sqrt_disc_e)

    return (mG2_b, m1_sq_b, m2_sq_b), (mG2_e, m1_sq_e, m2_sq_e)


def gauge_masses_bare_and_longitudinal_debye(rho_phys, t_phys):
    """
    Return bare gauge masses and Debye-improved longitudinal masses.

    Outputs:
      (mW_T2, mZ_T2, mA_T2), (mW_L2, mZ_L2, mA_L2), g1, g2, yt, tau_t

    Notes:
      - J_B should use bare masses.
      - Ring term should use longitudinal Debye-improved masses only.
    """
    eps = 1e-10
    tau_t = tau_uv * torch.exp(-t_phys)

    with torch.no_grad():
        g1, g2, yt, _, _ = get_running_couplings(t_phys)

    # bare / transverse masses
    mW_T2 = 0.5 * g2**2 * rho_phys
    mZ_T2 = 0.5 * (g2**2 + g1**2) * rho_phys
    mA_T2 = torch.zeros_like(rho_phys)

    # Debye masses for longitudinal modes
    Pi_W = (11.0 / 6.0) * g2**2 * tau_t**2
    Pi_B = (11.0 / 6.0) * g1**2 * tau_t**2

    mW_L2 = mW_T2 + Pi_W

    M11 = mW_T2 + Pi_W
    M22 = 0.5 * g1**2 * rho_phys + Pi_B
    M12 = -0.5 * g2 * g1 * rho_phys

    disc = (M11 - M22)**2 + 4.0 * M12**2
    sqrt_disc = torch.sqrt(torch.clamp(disc, min=eps))

    mZ_L2 = 0.5 * (M11 + M22 + sqrt_disc)
    mA_L2 = 0.5 * (M11 + M22 - sqrt_disc)

    return (mW_T2, mZ_T2, mA_T2), (mW_L2, mZ_L2, mA_L2), g1, g2, yt, tau_t


def _m32_pos(m2):
    """
    (m^2)^(3/2) with negative m^2 clipped to zero.
    This is the standard AE ring prescription for the zero modes.
    """
    return torch.clamp(m2, min=0.0) ** 1.5


def u_thermal_finiteT(t_phys, rho_phys, sigma_phys):
    """
    Finite-T thermal part of the dimensionless potential u_th(t, rho, sigma),
    1-loop (non-ring-resummed) piece only:

      u_th = u_1loop(bare masses in J_B/J_F)

    The Arnold-Espinosa ring/daisy correction (zero-mode only) is intentionally
    NOT included here -- it lives separately in get_u_ring() below. Callers that
    need the physical (ring-resummed) potential must add get_u_ring() themselves;
    this bare 1-loop version is what loss_extensions.py's del_aH (-> F_H_target)
    is built from, deliberately, since a third rho-derivative of the ring piece
    is non-analytic at rho=0 (see CLAUDE.md).

    Included modes:
      - scalar: Goldstones + 2 scalar eigenmodes
      - gauge: W_T, Z_T in 1-loop (longitudinal W/Z/gamma ring piece excluded,
        see get_u_ring())
      - fermion: top in 1-loop only

    All masses are dimensionless in the FRG convention, tau_t = T/k.
    """
    orig_shape = rho_phys.shape

    t_phys = t_phys.view(-1, 1)
    rho_phys = rho_phys.view(-1, 1)
    sigma_phys = sigma_phys.view(-1, 1)

    # -----------------------------
    # Scalars: bare and Debye masses
    # -----------------------------
    scalars_b, scalars_e = scalar_masses_bare_and_debye(rho_phys, sigma_phys, t_phys)
    mG2_b, m1_sq_b, m2_sq_b = scalars_b
    mG2_e, m1_sq_e, m2_sq_e = scalars_e

    # -----------------------------
    # Gauge: bare and longitudinal Debye masses
    # -----------------------------
    gauges_b, gauges_e, g1, g2, yt, tau_t = gauge_masses_bare_and_longitudinal_debye(
        rho_phys, t_phys
    )
    mW_T2, mZ_T2, mA_T2 = gauges_b
    mW_L2, mZ_L2, mA_L2 = gauges_e

    # fermion
    mt2 = yt**2 * rho_phys

    eps = 1e-12
    tau2 = torch.clamp(tau_t**2, min=eps)
    tau4 = tau2 * tau2

    # -----------------------------
    # 1-loop thermal part:
    # use BARE masses in J_B
    # -----------------------------
    yG2   = mG2_b   / tau2
    y1_2  = m1_sq_b / tau2
    y2_2  = m2_sq_b / tau2

    yWT2  = mW_T2 / tau2
    yZT2  = mZ_T2 / tau2
    yAT2  = mA_T2 / tau2

    yt2_T = mt2 / tau2

    _t, _w = get_leggauss_for(yt2_T, _GL_t_cpu, _GL_w_cpu)

    JB_G  = J_B(yG2,  _t, _w)
    JB_1  = J_B(y1_2, _t, _w)
    JB_2  = J_B(y2_2, _t, _w)

    JB_W  = J_B(yWT2, _t, _w)
    JB_Z  = J_B(yZT2, _t, _w)
    JB_A  = J_B(yAT2, _t, _w)

    JF_t  = J_F(yt2_T, _t, _w)

    # Scalars:
    # 3 Goldstones + 2 scalar eigenmodes
    u_th_scalar = (tau4 / (2.0 * torch.pi**2)) * (
        NGS * JB_G + JB_1 + JB_2
    )

    # Gauge:
    # W: 2 charges x 3 pol = 6 dof
    # Z: 3 dof
    # A: 1 longitudinal-like zero-mass contribution in J_B(0)
    # For AE bookkeeping in the 1-loop thermal piece:
    u_th_gauge_1loop = (tau4 / (2.0 * torch.pi**2)) * (
        6.0 * JB_W + 3.0 * JB_Z + 1.0 * JB_A
    )

    # Top: 12 dof
    u_th_top = -(12.0 * tau4 / (2.0 * torch.pi**2)) * JF_t

    u_th_1loop = u_th_scalar + u_th_gauge_1loop + u_th_top

    # -----------------------------
    # Ring term:
    # zero modes only, AE prescription
    # -----------------------------
    '''
    # In any cases, the ressumations are already incldued in 
    # wetterich-equations for scalars
    # The ring terms for the gauge bosons will be included after the learning phase
    ring_G  = NGS * (_m32_pos(mG2_e)   - _m32_pos(mG2_b))
    ring_1  =       (_m32_pos(m1_sq_e) - _m32_pos(m1_sq_b))
    ring_2  =       (_m32_pos(m2_sq_e) - _m32_pos(m2_sq_b))

    # longitudinal gauge zero modes only
    ring_WL = 2.0 * (_m32_pos(mW_L2) - _m32_pos(mW_T2))
    ring_ZL = 1.0 * (_m32_pos(mZ_L2) - _m32_pos(mZ_T2))
    ring_AL = 1.0 * (_m32_pos(mA_L2) - _m32_pos(mA_T2))

    u_ring = - (tau_t / (12.0 * torch.pi)) * (
        ring_G + ring_1 + ring_2 + ring_WL + ring_ZL + ring_AL
    )
    '''
    #u_th = u_th_1loop + u_ring
    u_th = u_th_1loop

    return u_th.view(orig_shape)

'''
def u_seed_finiteT(t_phys, rho_phys, sigma_phys):
    """
    Seed potential at finite T:
      u_seed(t, rho, sigma) = u_tree_exact + u_thermal_finiteT

    Zero-T CW parts are NOT included (assumed in running couplings).
    """
    u_tree = u_tree_exact(t_phys, rho_phys, sigma_phys)
    u_th   = u_thermal_finiteT(t_phys, rho_phys, sigma_phys)
    return u_tree + u_th
'''

def get_u_ring(t_phys, rho_phys):
    """
    All masses are dimensionless in the FRG convention, tau_t = T/k.
    """
    orig_shape = rho_phys.shape

    t_phys = t_phys.view(-1, 1)
    rho_phys = rho_phys.view(-1, 1)
 
    # -----------------------------
    # Gauge: bare and longitudinal Debye masses
    # -----------------------------
    gauges_b, gauges_e, g1, g2, yt, tau_t = gauge_masses_bare_and_longitudinal_debye(
        rho_phys, t_phys
    )
    mW_T2, mZ_T2, mA_T2 = gauges_b
    mW_L2, mZ_L2, mA_L2 = gauges_e

    # -----------------------------
    # Ring term:
    # zero modes only, AE prescription
    # -----------------------------
    # longitudinal gauge zero modes only
    ring_WL = 2.0 * (_m32_pos(mW_L2) - _m32_pos(mW_T2))
    ring_ZL = 1.0 * (_m32_pos(mZ_L2) - _m32_pos(mZ_T2))
    ring_AL = 1.0 * (_m32_pos(mA_L2) - _m32_pos(mA_T2))

    u_ring = - (tau_t / (12.0 * torch.pi)) * (
        ring_WL + ring_ZL + ring_AL
    )

    return u_ring.view(orig_shape)

def u_CW_zeroT(t_phys, rho_phys, sigma_phys):
    """
    Zero-T 1-loop Coleman-Weinberg potential (dimensionless).
    Renormalization scale is identified with the FRG sliding scale k (mu = k).
    
    All masses are bare and dimensionless in the FRG convention.
    """
    orig_shape = rho_phys.shape

    t_phys = t_phys.view(-1, 1)
    rho_phys = rho_phys.view(-1, 1)
    sigma_phys = sigma_phys.view(-1, 1)

    # -----------------------------
    # Scalars: Bare masses
    # -----------------------------
    scalars_b, _ = scalar_masses_bare_and_debye(rho_phys, sigma_phys, t_phys)
    mG2_b, m1_sq_b, m2_sq_b = scalars_b

    # -----------------------------
    # Gauge: Bare masses & Couplings
    # -----------------------------
    gauges_b, _, g1, g2, yt, _ = gauge_masses_bare_and_longitudinal_debye(rho_phys, t_phys)
    mW_T2, mZ_T2, mA_T2 = gauges_b

    # Top quark mass
    mt2 = yt**2 * rho_phys

    # take real part for tachyonic masses
    eps = 1e-5
    def cw_term(m2, dof, c):
        m2_abs = torch.abs(m2) + eps
        return (dof / (64.0 * torch.pi**2)) * (m2**2) * (torch.log(m2_abs) - c)
    # -----------------------------
    # Calculate CW contributions
    # -----------------------------
    # Scalars (c = 3/2), NGS=3.0 for Goldstones
    cw_G = cw_term(mG2_b, NGS, 1.5)
    cw_1 = cw_term(m1_sq_b, 1.0, 1.5)
    cw_2 = cw_term(m2_sq_b, 1.0, 1.5)

    # Gauge Bosons (c = 5/6 in Landau gauge)
    # W: 2 charges x 3 polarizations = 6 dof
    # Z: 3 dof
    # A (Photon): massless, omitted (or results in 0 due to clamp handling if added)
    cw_W = cw_term(mW_T2, 6.0, 5.0 / 6.0)
    cw_Z = cw_term(mZ_T2, 3.0, 5.0 / 6.0)

    # Top Quark (c = 3/2)
    # 3 colors x 4 (Dirac) = 12 dof (Fermion -> negative sign)
    cw_top = cw_term(mt2, -12.0, 1.5)

    # Total CW potential
    u_cw = cw_G + cw_1 + cw_2 + cw_W + cw_Z + cw_top

    return u_cw.view(orig_shape)