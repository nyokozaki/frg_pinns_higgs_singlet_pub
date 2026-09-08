"""
Higgs-only version of `loss_sign_and_mag_couplings` (the soft sign/magnitude
constraint) from frg_pinns_higgs_only/loss_extensions.py, translated into a form
usable as the "residual" of a Newton-Krylov relaxation method.

While weight_sign, weight_mag > 0, the equation Newton actually solves is not
"the pure Wetterich flow R=0" but "the Wetterich flow + this soft forcing"
(when used as a continuation method, decay the weight stepwise to 0).
"""

import numpy as np

import _pathsetup  # noqa: F401
from perturbation.config_params import k_IR, t_range
from running_couplings_higgs_only import get_running_quartics, get_running_masses
from seed_potential_higgs_only import aH, lamH
import cw_thermal

_EPS = 1e-12


def _del_aH_thermal(t_phys, rho=None):
    """
    "Effective aH" contribution that folds the thermal correction into
    F_H_target (the forcing target of the mass term). The honest way -- taking
    a rho-derivative of u_thermal_finiteT (numerical J_B/J_F integrals) and
    extracting the aH component of a local quadratic fit -- adds three J_B
    integrals per residual evaluation of the Jacobian-free Newton-Krylov, on top
    of a large call count, and was measured to slow convergence extremely
    (>20x), so it is not used.

    Instead only the leading order of the large-T expansion is used (the Debye
    thermal mass Pi_H, cw_thermal.Pi_H_thermal_mass). Pi_H is rho-independent
    (a uniform shift of the mass term only, no effect on the quartic), so adding
    this term keeps F_H_target itself a rho-independent scalar as before -- there
    is no need to make mask_F_H an array.
    """
    return cw_thermal.Pi_H_thermal_mass(t_phys)

# relu(x)=max(x,0) has a kink at x=0; Jacobian-free Newton-Krylov's finite-
# difference directional derivatives see that kink as a source of noise
# (same mechanism as the sqrt(disc) branch point diagnosed elsewhere), and
# empirically fails to converge once this prior is kept active with a fixed
# nonzero weight. _softplus is a smooth (C^infinity) stand-in: beta->inf
# recovers relu exactly; beta=10 keeps the transition within ~0.1-0.3 (natural
# scale of the normalized arguments below) while removing the corner.
_SOFTPLUS_BETA = 10.0


def _softplus(x, beta=_SOFTPLUS_BETA):
    bx = beta * x
    return (np.maximum(bx, 0.0) + np.log1p(np.exp(-np.abs(bx)))) / beta


def _sign_hinge(target, val, norm):
    s = np.sign(target)
    return _softplus(-(s * val) / norm)


def _mag_band(target, val, norm, c_lower, c_upper):
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
    rho_cw_cut=0.6,
    mask_temp=10.0,
    f_threshold=1e-2,
    d_threshold=1e-2,
    disable_sign_F_H=False,
):
    """
    Evaluate the soft sign/magnitude constraint against the perturbative
    running quartic/mass (Higgs side only) as a per-grid-point forcing term
    (added to flow_rhs).

    rho_cw_cut: ports the same low-rho/high-rho division of labour as the
    finite-T branch of frg_pinns_higgs_only/loss_extensions.py
    (`_loss_sign_and_mag_couplings_tval_finiteT`, hp.rho_cw_cut=0.6) -- the
    sign+magnitude constraint on the mass term (F_H) applies only for
    rho < rho_cw_cut, and the sign+magnitude band (c_lower/c_upper) on the
    RGE-run target of the quartic (D_H) applies only for rho > rho_cw_cut
    (following the PINN-side design of not constraining the quartic at low rho
    nor the mass at high rho). The constant cap on D_H_NN/lamH
    (quartic_ratio_cap) was originally a separate term active over all rho, so
    it is kept as-is independently of this split.
    """
    if weight_sign == 0.0 and weight_mag == 0.0:
        return 0.0

    rho = grid.RHO
    u_rho, u_rhorho = grid.derivatives(U, scheme="central")

    nn_H = 0.5 * u_rhorho
    aH_NN = u_rho - u_rhorho * rho

    lamH_t = get_running_quartics(t_phys)
    mhsq_t = get_running_masses(t_phys)
    kt = k_IR * np.exp(-t_range + t_phys)
    aH_run = mhsq_t / kt ** 2

    aH_tree = aH * np.exp(-2.0 * t_phys)

    D_H_target = lamH_t - lamH
    D_H_NN = nn_H - lamH

    del_aH = _del_aH_thermal(t_phys)
    aH_run = aH_run + del_aH

    F_H_target = aH_run - aH_tree
    F_H_NN = aH_NN - aH_tree

    norm_lamH = abs(lamH) + _EPS
    norm_aH = np.abs(aH_tree) + _EPS

    mask_D_H = float(abs(D_H_target / norm_lamH) > d_threshold)
    mask_F_H = float(abs(F_H_target / norm_aH) > f_threshold)

    # original domain-wide gate (rho_cut=3.0 > domain max 1.75, so effectively 1)
    mask_domain = 1.0 / (1.0 + np.exp(-mask_temp * (rho_cut - rho)))

    # low-rho/high-rho split at rho_cw_cut
    mask_rho_lo = 1.0 / (1.0 + np.exp(-mask_temp * (rho_cw_cut - rho)))
    mask_rho_hi = 1.0 / (1.0 + np.exp(-mask_temp * (rho - rho_cw_cut)))

    soft_mask_lo = mask_domain * mask_rho_lo
    soft_mask_hi = mask_domain * mask_rho_hi

    sign_pen = (
        soft_mask_hi * mask_D_H * _sign_hinge(D_H_target, D_H_NN, norm_lamH)
        + (0.0 if disable_sign_F_H else soft_mask_lo * mask_F_H * _sign_hinge(F_H_target, F_H_NN, norm_aH))
    )

    mag_pen = (
        soft_mask_lo * mask_F_H * _mag_band(F_H_target, F_H_NN, norm_aH, c_lower, c_upper)
        + soft_mask_hi * mask_D_H * _mag_band(D_H_target, D_H_NN, norm_lamH, c_lower, c_upper)
        + mask_domain * _softplus(np.abs(D_H_NN / lamH) - quartic_ratio_cap)
    )

    return weight_sign * sign_pen + weight_mag * mag_pen
