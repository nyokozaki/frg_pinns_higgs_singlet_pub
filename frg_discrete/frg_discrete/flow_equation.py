"""
Evaluate the right-hand side of the Wetterich equation (LPA', finite T) on the grid.

NumPy port of R_FRG_in from frg_pinns_higgs_singlet/FRG_residual.py. On the PINN
side the residual R = rtree - rloop is trained to R=0 as a loss; here the same
R=0, solved for u_t, i.e.

    du/dt = -4u + (2+eta_rho)*rho*u_rho + (2+eta_sigma)*sigma*u_sigma + rloop

is used directly as the method-of-lines right-hand side (RHS).

The FRG-time convention is the same as in running_couplings.py:
  t_phys=0 is the UV edge, t_phys=t_range (negative) the IR edge.

--------------------------------------------------------------------
On the tree/residual split (flow_rhs_w)
--------------------------------------------------------------------
u_tree_exact(t,rho,sigma) (only the mass terms scale as a -> a*exp(-2t); the
quartics lamH,lamS,lamHS are fixed) is the exact solution of the "free" flow
with eta=0, rloop=0,

    du/dt = -4u + 2*rho*u_rho + 2*sigma*u_sigma

(because u_tree is exactly the RG canonical scaling). Writing u = u_tree + w,
the equation satisfied by w = u - u_tree is

    dw/dt = F_tree(t,rho,sigma) - 4w + (2+eta_rho)*rho*w_rho
            + (2+eta_sigma)*sigma*w_sigma + rloop(masses from u_tree+w, t)

    F_tree = eta_rho(t)*rho*u_tree_rho + eta_sigma(t)*sigma*u_tree_sigma

(the u_tree piece "-4u_tree+2rho*u_tree_rho+2sigma*u_tree_sigma-du_tree/dt"
vanishes identically, so only the eta-generated terms remain). Since eta_rho,
eta_sigma are loop-generated quantities coming from the running couplings,
F_tree is essentially part of the loop effect too and belongs on the w
(residual) side, not "tree".

The u_tree piece is evaluated analytically and exactly (zero discretization
error), so only the residual w is exposed to the exponential amplification of
the -4u canonical-scaling term (over an integration range Delta t=2,
e^{4*2}=e^8 ~ 3000x). Because w is much smaller than the tree part (of thermal-
correction size), the impact of amplified numerical error is greatly reduced.
"""

import numpy as np

from perturbation.config_params import tau_uv, finite_T
from running_couplings import get_running_couplings
from seed_potential import aH, aS, lamH, lamS, lamHS

kloop = 1.0 / (12.0 * np.pi ** 2)
NGS = 3.0
LPAp = 1.0
EPS = 1e-10


def eta_rho(t_phys):
    g1, g2, yt, _, _ = get_running_couplings(t_phys)
    gamma = (1.0 / (16.0 * np.pi ** 2)) * (
        3.0 * yt ** 2 - 3.0 * g1 ** 2 / 4.0 - 9.0 * g2 ** 2 / 4.0
    )
    return 2.0 * gamma


def eta_sigma(t_phys):
    _, _, _, lamS_t, lamHS_t = get_running_couplings(t_phys)
    # gamma_S from PyR@TE3 for this model (matches frg_pinn.tex Sec. 2.3 and the
    # PINN get_eta_sigma in frg_pinns_higgs_singlet/FRG_residual.py):
    #   gamma_S = (16 pi^2)^-2 (lambda_HS^2 + 3 lambda_S^2),  eta_S = 2 gamma_S.
    # Runs before 2026-09 used (18 lamS^2 + 2 lamHS^2) -- the 1-loop beta_lamS
    # coefficient reused as a proxy; eta_S ~ 1e-4 either way and its effect on the
    # flow is ~1e-5.
    gamma = (1.0 / (16.0 * np.pi ** 2)) ** 2 * (3.0 * lamS_t ** 2 + 1.0 * lamHS_t ** 2)
    return 2.0 * gamma


def _coth(x, eps=0.0):
    return np.cosh(x) / (np.sinh(x) + eps)


def _inv_sqrt(m2, floor=EPS):
    return 1.0 / np.sqrt(np.clip(1.0 + m2, floor, None))


def u_tree_derivs(t_phys, rho, sigma):
    """Return the exact (u_rho, u_sigma, u_rhorho, u_sigmasigma, u_rhosigma) of u_tree_exact."""
    muH2_t = aH * np.exp(-2.0 * t_phys)
    muS2_t = aS * np.exp(-2.0 * t_phys)
    u_rho = muH2_t + 2.0 * lamH * rho + lamHS * sigma
    u_sigma = muS2_t + 2.0 * lamS * sigma + lamHS * rho
    u_rhorho = 2.0 * lamH
    u_sigmasigma = 2.0 * lamS
    u_rhosigma = lamHS
    return u_rho, u_sigma, u_rhorho, u_sigmasigma, u_rhosigma


def _rloop_from_derivs(t_phys, rho, sigma, u_rho, u_sigma, u_rhorho, u_sigmasigma, u_rhosigma,
                        warn_on_tachyon=False, disc_delta=0.0, disc_cut=0.0, mass_floor=0.0):
    """
    Compute rloop = kloop*loop_sum (clipped) from the mass eigenvalues and
    thermal threshold functions. Shared by flow_rhs (full u) and flow_rhs_w
    (residual w, which passes the composite derivatives from u_tree+w).

    disc_delta : regularization parameter for the discriminant
                 disc=(M11-M22)^2+4*M12^2 of the Higgs/singlet mass eigenvalues.
                 When disc_delta>0, sqrt_disc = sqrt(disc + disc_delta**2) is
                 used, smoothly rounding the conical singularity at disc=0 with
                 radius ~disc_delta (though the derivative near disc=0 still
                 remains of order ~1/disc_delta).
    disc_cut : an alternative regularization, mutually exclusive with disc_delta.
               When disc_cut>0, a hard clip sqrt_disc = sqrt(max(disc, disc_cut**2))
               is used. In the region disc < disc_cut**2, sqrt_disc is constant,
               so its derivative with respect to U is exactly zero there (the aim
               being that the finite-difference Jacobian of JFNK no longer picks
               up noise from this term; but a kink in the derivative remains at
               the boundary disc=disc_cut**2).
    The default (disc_delta=disc_cut=0.0) uses the previous sqrt(clip(disc, EPS,
    None)) and does not change behaviour (backward compatibility with existing
    callers).
    mass_floor : hard-clip floor for the normalized mass-squared 1+m^2 of the
                 Higgs/singlet mass eigenvalues (mG2, m1_sq, m2_sq). When
                 mass_floor>0, both _inv_sqrt and the E of the finite-temperature
                 terms use clip(1+m^2, mass_floor, None) instead of
                 clip(1+m^2, EPS, None). Gauge/top are unaffected since
                 1+m^2>=1 never touches the floor. The default 0.0 keeps the
                 previous EPS floor.
    """
    g1, g2, yt, _, _ = get_running_couplings(t_phys)

    mG2 = u_rho
    M11 = u_rho + 2.0 * rho * u_rhorho
    M22 = u_sigma + 2.0 * sigma * u_sigmasigma
    M12 = 2.0 * np.sqrt(np.clip(rho * sigma, EPS, None)) * u_rhosigma

    disc = (M11 - M22) ** 2 + 4.0 * M12 ** 2
    if disc_cut > 0.0:
        sqrt_disc = np.sqrt(np.clip(disc, disc_cut ** 2, None))
    elif disc_delta > 0.0:
        sqrt_disc = np.sqrt(disc + disc_delta ** 2)
    else:
        sqrt_disc = np.sqrt(np.clip(disc, EPS, None))

    m1_sq = 0.5 * (M11 + M22 - sqrt_disc)
    m2_sq = 0.5 * (M11 + M22 + sqrt_disc)

    if warn_on_tachyon:
        if np.any(1.0 + mG2 < 0.0) or np.any(1.0 + m1_sq < 0.0):
            import warnings
            warnings.warn(
                f"t={t_phys:.4f}: 1+mG2 or 1+m1_sq went negative "
                f"(min 1+mG2={np.min(1.0+mG2):.3e}, min 1+m1_sq={np.min(1.0+m1_sq):.3e}); "
                "clamped at eps, regulator is being pushed into an unphysical regime."
            )

    mW_T2 = 0.5 * g2 ** 2 * rho
    mZ_T2 = 0.5 * (g2 ** 2 + g1 ** 2) * rho
    mt2 = yt ** 2 * rho
    mW_L2 = mW_T2
    mZ_L2 = mZ_T2

    mass_eps = mass_floor if mass_floor > 0.0 else EPS
    termG_0 = _inv_sqrt(mG2, floor=mass_eps)
    term1_0 = _inv_sqrt(m1_sq, floor=mass_eps)
    term2_0 = _inv_sqrt(m2_sq, floor=mass_eps)
    termWT0 = _inv_sqrt(mW_T2)
    termWL0 = _inv_sqrt(mW_L2)
    termZT0 = _inv_sqrt(mZ_T2)
    termZL0 = _inv_sqrt(mZ_L2)
    termT0 = _inv_sqrt(mt2)

    if finite_T:
        tau_t = tau_uv * np.exp(-t_phys)

        EG = np.sqrt(np.clip(1.0 + mG2, mass_eps, None))
        E1 = np.sqrt(np.clip(1.0 + m1_sq, mass_eps, None))
        E2 = np.sqrt(np.clip(1.0 + m2_sq, mass_eps, None))
        E_WT = np.sqrt(np.clip(1.0 + mW_T2, EPS, None))
        E_WL = np.sqrt(np.clip(1.0 + mW_L2, EPS, None))
        E_ZT = np.sqrt(np.clip(1.0 + mZ_T2, EPS, None))
        E_ZL = np.sqrt(np.clip(1.0 + mZ_L2, EPS, None))
        E_T = np.sqrt(np.clip(1.0 + mt2, EPS, None))

        factorG = _coth(EG / (2.0 * tau_t))
        factor1 = _coth(E1 / (2.0 * tau_t))
        factor2 = _coth(E2 / (2.0 * tau_t))
        factor_WT = _coth(E_WT / (2.0 * tau_t))
        factor_WL = _coth(E_WL / (2.0 * tau_t))
        factor_ZT = _coth(E_ZT / (2.0 * tau_t))
        factor_ZL = _coth(E_ZL / (2.0 * tau_t))
        factor_T = np.tanh(E_T / (2.0 * tau_t))

        termG = termG_0 * factorG
        term1 = term1_0 * factor1
        term2 = term2_0 * factor2
        termWT = termWT0 * factor_WT
        termWL = termWL0 * factor_WL
        termZT = termZT0 * factor_ZT
        termZL = termZL0 * factor_ZL
        termT = termT0 * factor_T
    else:
        termG, term1, term2 = termG_0, term1_0, term2_0
        termWT, termWL = termWT0, termWL0
        termZT, termZL = termZT0, termZL0
        termT = termT0

    loop_sum = (
        3.0 * termG
        + term1
        + term2
        + 4.0 * termWT
        + 2.0 * termWL
        + 2.0 * termZT
        + 1.0 * termZL
        - 12.0 * termT
    )

    rloop = kloop * loop_sum
    rloop = np.clip(rloop, -2.0, 2.0)
    return rloop


def flow_rhs(t_phys, U, grid, warn_on_tachyon=False, scheme="upwind", disc_delta=0.0, disc_cut=0.0,
             mass_floor=0.0):
    """
    Return du/dt(t_phys, rho, sigma) over the whole grid (a 2D array of
    grid.shape); integrates the full u directly.

    U : ndarray, shape grid.shape
        the dimensionless potential u(rho, sigma) at the current RG time t_phys.
    scheme : "upwind" (default, stable for explicit time integration) or
             "central" (higher accuracy but explicit integration can become
             unstable at high resolution). See grid_fd.Grid2D.derivatives.
    disc_delta, disc_cut, mass_floor : see _rloop_from_derivs
        (regularization of the mass-eigenvalue discriminant; disc_delta and
        disc_cut are mutually exclusive, mass_floor can be combined with either).
    """
    u_rho, u_sigma, u_rhorho, u_sigmasigma, u_rhosigma = grid.derivatives(U, scheme=scheme)

    rho = grid.RHO
    sigma = grid.SIGMA

    eta_r = eta_rho(t_phys)
    eta_s = eta_sigma(t_phys)

    rloop = _rloop_from_derivs(
        t_phys, rho, sigma, u_rho, u_sigma, u_rhorho, u_sigmasigma, u_rhosigma,
        warn_on_tachyon=warn_on_tachyon, disc_delta=disc_delta, disc_cut=disc_cut,
        mass_floor=mass_floor,
    )

    du_dt = (
        -4.0 * U
        + (2.0 + eta_r * LPAp) * rho * u_rho
        + (2.0 + eta_s * LPAp) * sigma * u_sigma
        + rloop
    )

    return du_dt


def flow_rhs_w(t_phys, W, grid, warn_on_tachyon=False, scheme="upwind"):
    """
    Return dw/dt for u = u_tree_exact(t,rho,sigma) + w.

    The tree piece "-4u_tree+2rho*u_tree_rho+2sigma*u_tree_sigma-du_tree/dt"
    vanishes identically (u_tree itself is the exact solution of the free flow
    with eta=0, rloop=0), so the only remaining forcing term is the
    eta-generated F_tree. The u_tree piece is evaluated analytically and exactly
    (no discretization error); the grid finite differences are applied only to w.
    """
    w_rho, w_sigma, w_rhorho, w_sigmasigma, w_rhosigma = grid.derivatives(W, scheme=scheme)

    rho = grid.RHO
    sigma = grid.SIGMA

    ut_rho, ut_sigma, ut_rhorho, ut_sigmasigma, ut_rhosigma = u_tree_derivs(t_phys, rho, sigma)

    u_rho = ut_rho + w_rho
    u_sigma = ut_sigma + w_sigma
    u_rhorho = ut_rhorho + w_rhorho
    u_sigmasigma = ut_sigmasigma + w_sigmasigma
    u_rhosigma = ut_rhosigma + w_rhosigma

    eta_r = eta_rho(t_phys)
    eta_s = eta_sigma(t_phys)

    rloop = _rloop_from_derivs(
        t_phys, rho, sigma, u_rho, u_sigma, u_rhorho, u_sigmasigma, u_rhosigma,
        warn_on_tachyon=warn_on_tachyon,
    )

    F_tree = eta_r * rho * ut_rho + eta_s * sigma * ut_sigma

    dw_dt = (
        F_tree
        - 4.0 * W
        + (2.0 + eta_r * LPAp) * rho * w_rho
        + (2.0 + eta_s * LPAp) * sigma * w_sigma
        + rloop
    )

    return dw_dt
