"""
Translation of `loss_sign_and_mag_couplings` (the soft sign/magnitude
constraint) from frg_pinns_higgs_singlet/loss_extensions.py into a form usable as
the "residual" of a Newton-Krylov relaxation method rather than a PINN "loss".

On the PINN side, the effective quartic couplings at each point (u_rhorho/2,
u_sigmasigma/2, u_rhosigma) and the effective mass parameters
(u_rho - u_rhorho*rho - u_rhosigma*sigma, etc.) are compared against the
perturbative RG-improved running couplings/masses, and a loose constraint that
"only fixes the rough direction and magnitude" is added to the loss:
  - a penalty if the sign disagrees (sign hinge)
  - a penalty if the ratio to the target is outside the band [c_lower, c_upper]
    (magnitude band)
(it does not force pointwise agreement).

Here this is implemented as a forcing term added to flow_rhs (du/dt) at each grid
point. It enters as a second kind of reaction term in the same "place" as rloop
(the reaction term coming from the mass eigenvalues and thermal threshold
functions) within du/dt.

Note (a fairness caveat):
    While weight_sign, weight_mag > 0, the equation Newton actually solves is not
    "the pure Wetterich flow R=0" but "the Wetterich flow + this soft forcing".
    The converged solution is a biased solution pulled toward this perturbative
    prior and does not in general coincide with the true FRG solution (the PINN
    side does not drive this loss all the way to zero either; it stops at a
    weighted compromise with the other loss terms, so this shares the same
    property as the PINN side). When used as a continuation method
    (--coupling-prior-anneal-iters), decaying the weight stepwise to 0 recovers
    the pure-flow-equation solution as the final converged solution.
"""

import numpy as np

import _pathsetup  # noqa: F401
from perturbation.config_params import k_IR, t_range
from running_couplings import get_running_quartics, get_running_masses
from seed_potential import aH, aS, lamH, lamS, lamHS

_EPS = 1e-12


def _relu(x):
    return np.maximum(x, 0.0)


# relu(x)=max(x,0) has a kink at x=0; Jacobian-free Newton-Krylov's finite-
# difference directional derivatives see that kink as a source of noise (same
# mechanism as the sqrt(disc) branch point diagnosed elsewhere), and empirically
# fails to converge once this prior is kept active with a fixed nonzero weight.
# _softplus is a smooth (C^infinity) stand-in: beta->inf recovers relu exactly;
# beta=10 keeps the transition within ~0.1-0.3 (natural scale of the normalized
# arguments below) while removing the corner. Matches frg_higgs_only/coupling_prior.py.
_SOFTPLUS_BETA = 10.0


def _softplus(x, beta=_SOFTPLUS_BETA):
    bx = beta * x
    return (np.maximum(bx, 0.0) + np.log1p(np.exp(-np.abs(bx)))) / beta


def _sign_hinge(target, val, norm):
    """0 if the sign agrees with target; if the signs are opposite, a penalty
    proportional to |val|/norm."""
    s = np.sign(target)
    return _softplus(-(s * val) / norm)


def _mag_band(target, val, norm, c_lower, c_upper):
    """0 if |val| lies in the band [c_lower, c_upper] times |target|."""
    mt = np.abs(target)
    mv = np.abs(val)
    return _softplus((c_lower * mt - mv) / norm) + _softplus((mv - c_upper * mt) / norm)


def coupling_prior_residual(
    t_phys, U, grid,
    weight_sign=0.0,
    weight_mag=0.0,
    c_lower=0.1,
    c_upper=3.0,
    quartic_ratio_cap=0.99,
    rho_cut=3.0,
    sigma_cut=3.0,
    mask_temp=10.0,
    f_threshold=1e-2,
    d_threshold=1e-2,
    disable_sign_F_H=False,
):
    """
    Evaluate the soft sign/magnitude constraint against the perturbative
    running quartics/masses as a per-grid-point forcing term (added to flow_rhs).

    Returns zero whenever weight_sign == weight_mag == 0.0 (does not change the
    default behaviour).

    rho_cut, sigma_cut : soft-mask boundary playing the same role as the PINN's
        hp.rho_cw_cut/sigma_cw_cut (default 3.0). The default 3.0 covers this
        solver's grid range (rho_max, sigma_max ~ 1.75), so the mask is
        effectively inactive.
    f_threshold, d_threshold : thresholds for ignoring regions where the target
        has barely moved from the tree value (too small relative to it to be
        perturbatively trustworthy).
    disable_sign_F_H : when True, remove only the sign hinge penalty for the
        rho-direction mass term (F_H, aH_NN vs aH_run/aH_tree) from sign_pen
        (the F_H constraint on the magnitude-band penalty side is kept).
    """
    if weight_sign == 0.0 and weight_mag == 0.0:
        return 0.0

    rho, sigma = grid.RHO, grid.SIGMA
    u_rho, u_sigma, u_rhorho, u_sigmasigma, u_rhosigma = grid.derivatives(U, scheme="central")

    nn_H = 0.5 * u_rhorho
    nn_S = 0.5 * u_sigmasigma
    nn_HS = u_rhosigma
    aH_NN = u_rho - u_rhorho * rho - u_rhosigma * sigma
    aS_NN = u_sigma - u_sigmasigma * sigma - u_rhosigma * rho

    lamH_t, lamS_t, lamHS_t = get_running_quartics(t_phys)
    mhsq_t, mssq_t = get_running_masses(t_phys)
    kt = k_IR * np.exp(-t_range + t_phys)
    aH_run = mhsq_t / kt ** 2
    aS_run = mssq_t / kt ** 2

    aH_tree = aH * np.exp(-2.0 * t_phys)
    aS_tree = aS * np.exp(-2.0 * t_phys)

    D_H_target, D_S_target, D_HS_target = lamH_t - lamH, lamS_t - lamS, lamHS_t - lamHS
    D_H_NN, D_S_NN, D_HS_NN = nn_H - lamH, nn_S - lamS, nn_HS - lamHS

    F_H_target, F_S_target = aH_run - aH_tree, aS_run - aS_tree
    F_H_NN, F_S_NN = aH_NN - aH_tree, aS_NN - aS_tree

    norm_lamH = abs(lamH) + _EPS
    norm_lamS = abs(lamS) + _EPS
    norm_lamHS = abs(lamHS) + _EPS
    norm_aH = np.abs(aH_tree) + _EPS
    norm_aS = np.abs(aS_tree) + _EPS

    mask_D_H = float(abs(D_H_target / norm_lamH) > d_threshold)
    mask_D_S = float(abs(D_S_target / norm_lamS) > d_threshold)
    mask_D_HS = float(abs(D_HS_target / norm_lamHS) > d_threshold)
    mask_F_H = float(abs(F_H_target / norm_aH) > f_threshold)
    mask_F_S = float(abs(F_S_target / norm_aS) > f_threshold)

    mask_rho = 1.0 / (1.0 + np.exp(-mask_temp * (rho_cut - rho)))
    mask_sigma = 1.0 / (1.0 + np.exp(-mask_temp * (sigma_cut - sigma)))
    soft_mask = mask_rho * mask_sigma

    sign_pen = (
        mask_D_H * _sign_hinge(D_H_target, D_H_NN, norm_lamH)
        + mask_D_S * _sign_hinge(D_S_target, D_S_NN, norm_lamS)
        + mask_D_HS * _sign_hinge(D_HS_target, D_HS_NN, norm_lamHS)
        + (0.0 if disable_sign_F_H else mask_F_H * _sign_hinge(F_H_target, F_H_NN, norm_aH))
        + mask_F_S * _sign_hinge(F_S_target, F_S_NN, norm_aS)
    )

    mag_pen = (
        mask_F_H * _mag_band(F_H_target, F_H_NN, norm_aH, c_lower, c_upper)
        + mask_F_S * _mag_band(F_S_target, F_S_NN, norm_aS, c_lower, c_upper)
        # quartics: upper bound only (keeping the NN from running away in a
        # direction that would break vacuum stability; same idea as
        # _loss_sign_and_mag_couplings_tval_finiteT in loss_extensions.py)
        + _softplus(np.abs(D_H_NN / lamH) - quartic_ratio_cap)
        + _softplus(np.abs(D_S_NN / lamS) - quartic_ratio_cap)
        + _softplus(np.abs(D_HS_NN / lamHS) - quartic_ratio_cap)
    )

    return soft_mask * (weight_sign * sign_pen + weight_mag * mag_pen)
