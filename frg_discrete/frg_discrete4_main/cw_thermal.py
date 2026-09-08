"""
NumPy port of u_CW_zeroT / u_thermal_finiteT (torch implementation) from
frg_pinns_higgs_singlet/thermal_functions.py. Same porting approach as jbjf.py
(frg_discrete/perturbation/jbjf.py): the torch dependency is replaced only by
the numpy perturbation/running_couplings of frg_discrete.

To keep everything self-contained within frg_discrete4_main, this does not
depend on any torch code from frg_pinns_higgs_singlet (used only as a reference
implementation). Following the existing frg_discrete4_main convention (see the
README), the shared numpy modules of the sibling folder frg_discrete
(perturbation/, running_couplings.py) are reused as-is via _pathsetup.py.

Purpose: static reference curves for comparison with the (non-converged)
Newton-Krylov solution of "the full flow eq. (rloop+eta)". The one-loop
Coleman-Weinberg (T=0) plus the finite-temperature one-loop thermal potential,
evaluated on the running couplings, are provided as "RGE-run tree + CW (+ thermal)".

Note: this is distinct from rloop of the Wetterich equation (the flow term, with
the coth threshold functions and the regulator); it is a static perturbative
one-loop effective potential. See the header comment of flow_equation.py and the
"tree/residual split" section of the README.
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
    The tree potential evaluated by substituting the running couplings (RGE)
    directly, instead of u_tree_exact (canonical scaling, aH*exp(-2t)).
    The basis of the "RGE-run tree" family of curves. At t=0 (UV) it agrees with
    u_tree_exact, but for t<0 the two differ in general (canonical scaling keeps
    the tree-level RG invariants exactly, whereas RGE-run reflects the nontrivial
    change of the two-loop running couplings).
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
    """Bare mass eigenvalues (from u_tree_rgerun) and the masses including the thermal Debye correction (Debye)."""
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
    """Static T=0 one-loop Coleman-Weinberg correction (mu=k). Renormalization scale = the FRG scale k."""
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
    Finite-temperature one-loop thermal potential (only the one-loop part of
    Arnold-Espinosa, no ring term). The arguments of J_B/J_F use the bare masses
    (same convention as thermal_functions.py).
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
    Overwrite perturbation.config_params.tau_uv (the T_RAW/k_IR*exp(t) constant
    computed once at module import) with the value for T_RAW=T_raw [GeV].
    Note that both this module's own global (which imports tau_uv directly) and
    (analogously, from the caller) the global of the flow_equation module need to
    be rewritten -- both copy the value via "from ... import tau_uv", so
    rewriting config_params.tau_uv alone has no effect.
    """
    global tau_uv
    from perturbation.config_params import k_IR as _k_IR, t as _t
    tau_uv = (T_raw / _k_IR) * np.exp(_t)
    return tau_uv
