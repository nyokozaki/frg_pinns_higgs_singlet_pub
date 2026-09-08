import torch
import numpy as np
import torch.autograd as autograd
import math

from perturbation.config_params import finite_T, k_IR, t_range, tau_uv
from thermal_functions import u_thermal_finiteT, u_CW_zeroT
from running_couplings import tree_params, get_running_masses, get_running_quartics, get_running_couplings
[aH,lamH] = tree_params

import hyperparams as hp
device = hp.device

def _hinge(x, beta=None):
    """Softplus hinge: smooth stand-in for torch.relu(x). beta controls sharpness
    (higher = closer to ReLU); default reads hp.hinge_beta so it's tunable without
    touching this file."""
    if beta is None:
        beta = hp.hinge_beta
    return torch.nn.functional.softplus(x, beta=beta)

def loss_sign_and_mag_couplings(
    model,
    n_samp,
    rho_cut=0.1,
    mag_factor=0.1,
    margin=0
):

    if finite_T == False:
        tval = -1.0
        L_sign_tot, L_mag_tot = _loss_sign_and_mag_couplings_tval(
            model,
            n_samp,
            rho_cut=rho_cut,
            mag_factor=mag_factor,
            margin=margin,
            tval=tval,
        )

    else:
        tval = -1.0
        L_sign_tot, L_mag_tot = _loss_sign_and_mag_couplings_tval_finiteT(
            model,
            n_samp,
            rho_cut=rho_cut,
            mag_factor=mag_factor,
            margin=margin,
            tval=tval,
        )

    return L_sign_tot, L_mag_tot

# Compared NN potential and perturvative potential with the tree-level potential at UV scale (without CW)
def _loss_sign_and_mag_couplings_tval(
    model,
    n_samp,
    rho_cut=0.1,
    mag_factor=0.1,
    margin=0,
    tval=-1.0,
    mask_temp=10.0  # temperature parameter setting the sharpness of the soft mask
):

    # relative weights
    w_mass_rho = hp.w_mass_rho
    w_sign_r2 = hp.w_sign_r2
    w_mag_mass_rho = hp.w_mass_mag_rho

    rho_cw_cut = hp.rho_cw_cut

    t_in = torch.full((n_samp, 1), tval, device=device, requires_grad=False)

    if rho_cut is None:
        rho_cut_in = 1.0
    else:
        rho_cut_in = model.scale_rho(rho_cut)

    rho_in = torch.empty(n_samp, 1, device=device).uniform_(-1.0, rho_cut_in).requires_grad_(True)

    t_phys = model.unscale_t(t_in)
    rho_phys = model.unscale_rho(rho_in)

    u, u_rho, u_rhorho = model(t_in, rho_in)

    nn_H = 0.5 * u_rhorho

    t_phys0 = t_phys[0:1]

    lamH_run = get_running_quartics(t_phys0)

    lamH_run_b = lamH_run.expand_as(nn_H)

    # mass terms
    mhsq_run = get_running_masses(t_phys0)

    kt = k_IR * torch.exp(-t_range + t_phys)

    aH_run = mhsq_run / (kt**2)

    # quadratic-like terms (if the form of the potential is close to the renormalizable one)
    aH_NN = u_rho - u_rhorho * rho_phys

    lamH_tree = lamH

    D_H_target = (lamH_run_b) - lamH_tree

    D_H_NN = nn_H - lamH_tree

    aH_tree = aH * torch.exp(-2.0*t_phys0)

    F_H_target = (aH_run) - aH_tree

    F_H_NN = aH_NN - aH_tree

    z = torch.tensor(0.0, device=device)
    sign_terms = []

    eps = 1e-12


    # ==========================================
    # compute the soft mask (restricted to the region rho_phys < x0)
    # ==========================================
    mask_rho = torch.sigmoid(mask_temp * (rho_cw_cut - rho_phys))
    soft_mask = mask_rho

    def masked_mean(loss_tensor, extra_mask=1.0):
        final_mask = soft_mask * extra_mask
        return (loss_tensor * final_mask).sum() / (final_mask.sum() + eps)


    # ==========================================
    # compute the penalty term (using masked_mean)
    # ==========================================
    norm_lamH = float(abs(lamH_tree)) + eps
    norm_aH = torch.abs(aH_tree) + eps

    f_threshold = 1e-2
    d_threshold = 1e-2

    mask_F_H = (torch.abs(F_H_target / aH_tree) > f_threshold).float()

    mask_D_H = (torch.abs(D_H_target / norm_lamH) > d_threshold).float()

    sH = torch.sign(D_H_target)
    sign_terms.append(
        w_sign_r2 * masked_mean(
            _hinge(-(sH * D_H_NN / norm_lamH) + margin),
            extra_mask=mask_D_H
        )
    )

    fH = torch.sign(F_H_target)

    loss_H = _hinge(-(fH * F_H_NN / norm_aH) + margin)

    sign_terms.append(w_mass_rho * masked_mean(loss_H, extra_mask=mask_F_H))

    L_sign = torch.stack(sign_terms).mean() if len(sign_terms) > 0 else z

    mag_H_target = torch.abs(F_H_target)
    mag_H_NN = torch.abs(F_H_NN)

    c_lower = hp.c_mag_lower
    c_upper = hp.c_mag_upper

    L_mag_H_lower = w_mag_mass_rho * masked_mean(
        _hinge((c_lower * mag_H_target - mag_H_NN) / norm_aH),
        extra_mask=mask_F_H
    )
    L_mag_H_upper = w_mag_mass_rho * masked_mean(
        _hinge((mag_H_NN - c_upper * mag_H_target) / norm_aH),
        extra_mask=mask_F_H
    )

    mag_H_target2 = torch.abs(D_H_target)
    mag_H_NN2 = torch.abs(D_H_NN)

    L_mag_H_lower2 = hp.w_mag_r2 * masked_mean(
        _hinge((c_lower * mag_H_target2 - mag_H_NN2) / norm_lamH),
        extra_mask=mask_D_H
    )
    L_mag_H_upper2 = hp.w_mag_r2 * masked_mean(
        _hinge((mag_H_NN2 - c_upper * mag_H_target2) / norm_lamH),
        extra_mask=mask_D_H
    )

    L_mag = (
        L_mag_H_lower + L_mag_H_upper
        + L_mag_H_lower2 + L_mag_H_upper2
    )


    return L_sign, L_mag

def _loss_sign_and_mag_couplings_tval_finiteT(
    model,
    n_samp,
    rho_cut=0.1,
    mag_factor=0.1,
    margin=0,
    tval=-1.0,
    mask_temp=10.0  # temperature parameter setting the sharpness of the soft mask
):

    # relative weights
    w_mass_rho = hp.w_mass_rho
    w_mag_mass_rho = hp.w_mass_mag_rho

    rho_cw_cut = hp.rho_cw_cut

    t_in = torch.full((n_samp, 1), tval, device=device, requires_grad=False)

    if rho_cut is None:
        rho_cut_in = 1.0
    else:
        rho_cut_in = model.scale_rho(rho_cut)

    rho_in = torch.empty(n_samp, 1, device=device).uniform_(-1.0, rho_cut_in).requires_grad_(True)

    t_phys = model.unscale_t(t_in)
    rho_phys = model.unscale_rho(rho_in)

    u, u_rho, u_rhorho = model(t_in, rho_in)

    nn_H = 0.5 * u_rhorho

    t_phys0 = t_phys[0:1]

    lamH_run = get_running_quartics(t_phys0)

    lamH_run_b = lamH_run.expand_as(nn_H)

    # mass terms
    mhsq_run = get_running_masses(t_phys0)

    kt = k_IR * torch.exp(-t_range + t_phys)

    aH_run = mhsq_run / (kt**2)


    # thermal correction
    rho1 = rho_phys.clone()
    rho1.requires_grad_(True)

    t_phys1 = t_phys.clone().detach()
    u_th1 = u_thermal_finiteT(t_phys1, rho1)

    (uth_r1,) = torch.autograd.grad(
        u_th1,
        (rho1,),
        grad_outputs=torch.ones_like(u_th1),
        create_graph=True
    )

    (uth_rr,) = torch.autograd.grad(
        uth_r1,
        rho1,
        grad_outputs=torch.ones_like(uth_r1),
        create_graph=True
    )

    del_aH = uth_r1 - uth_rr * rho_phys

    # for thermal correction safe
    rho_cut = 0.05
    temp = 2000.0

    mask_rhoT = torch.sigmoid(temp * (rho1 - rho_cut))
    soft_maskT = mask_rhoT

    aH_run = aH_run + del_aH.detach()


    # quadratic-like terms (if the form of the potential is close to the renormalizable one)
    aH_NN = u_rho - u_rhorho * rho_phys

    lamH_tree = lamH

    D_H_target = (lamH_run_b) - lamH_tree

    D_H_NN = nn_H - lamH_tree

    aH_tree = aH * torch.exp(-2.0*t_phys0)

    F_H_target = (aH_run) - aH_tree

    # --- a mask to ignore the region near 0 ---
    f_threshold = 1e-2  # adjust to the scale of the values as appropriate
    mask_F_H = (torch.abs(F_H_target/aH_tree) > f_threshold).float()

    F_H_NN = aH_NN - aH_tree

    z = torch.tensor(0.0, device=device)
    sign_terms = []

    eps = 1e-12


    # ==========================================
    # compute the soft mask (restricted to the region rho_phys < x0)
    # ==========================================
    mask_rho = torch.sigmoid(mask_temp * (rho_cw_cut - rho_phys))
    soft_mask = mask_rho

    soft_mask = soft_mask * soft_maskT

    def masked_mean(loss_tensor, extra_mask=1.0):
        """Weighted average by the mask; evaluate the loss only over the region of valid samples."""
        final_mask = soft_mask * extra_mask
        return (loss_tensor * final_mask).sum() / (final_mask.sum() + eps)

    # ==========================================
    # soft mask on the rho > rho_cw_cut side (the complement of mask_rho).
    # In this region only the quartic coupling (D_H_NN) is constrained in
    # sign+magnitude; the mass term (F_H_NN) is not constrained
    # ==========================================
    mask_rho_hi = torch.sigmoid(mask_temp * (rho_phys - rho_cw_cut))
    soft_mask_hi = mask_rho_hi * soft_maskT

    def masked_mean_hi(loss_tensor, extra_mask=1.0):
        final_mask = soft_mask_hi * extra_mask
        return (loss_tensor * final_mask).sum() / (final_mask.sum() + eps)

    norm_lamH = float(abs(lamH_tree)) + eps

    d_threshold = 1e-2
    mask_D_H = (torch.abs(D_H_target / norm_lamH) > d_threshold).float()

    #
    # mass terms
    #
    norm_aH = torch.abs(aH_tree) + eps

    fH = torch.sign(F_H_target)

    loss_H = _hinge(-(fH * F_H_NN / norm_aH) + margin)

    sign_terms.append(w_mass_rho * masked_mean(loss_H, extra_mask=mask_F_H))

    # rho > rho_cw_cut: quartic self-coupling sign constraint (mass-term is
    # intentionally left unconstrained here, per soft_mask_hi's region)
    sH2 = torch.sign(D_H_target)

    loss_D_sign = _hinge(-(sH2 * D_H_NN / norm_lamH) + margin)

    sign_terms.append(hp.w_sign_r2 * masked_mean_hi(loss_D_sign, extra_mask=mask_D_H))

    L_sign = torch.stack(sign_terms).mean() if len(sign_terms) > 0 else z

    #############################
    #
    #   magnitude loss
    #   For the mass-like combination we use the same logic as that of the zero T
    #
    #############################

    mag_H_target = torch.abs(F_H_target)

    mag_H_NN = torch.abs(F_H_NN)

    c_lower = hp.c_mag_lower
    c_upper = hp.c_mag_upper

    L_mag_H_lower = w_mag_mass_rho*masked_mean(_hinge((c_lower * mag_H_target - mag_H_NN)/norm_aH), extra_mask=mask_F_H)
    L_mag_H_upper = w_mag_mass_rho*masked_mean(_hinge((mag_H_NN - c_upper * mag_H_target)/norm_aH), extra_mask=mask_F_H)


    # self coupling magnitude
    # uppper lower bounds are taken loosely
    # self couplings we use the same weights as those for the mass terms

    c_upper = 0.99

    mag_H_NN2 = torch.abs(D_H_NN/lamH_tree)

    L_mag_H_lower2 = z
    L_mag_H_upper2 = hp.w_mag_r2*masked_mean(_hinge(mag_H_NN2 - c_upper))

    # rho > rho_cw_cut: quartic self-coupling magnitude band vs the RGE-running
    # target D_H_target (mass-term is intentionally left unconstrained here)
    c_lower_hi = hp.c_mag_lower
    c_upper_hi = hp.c_mag_upper

    mag_D_target_hi = torch.abs(D_H_target)
    mag_D_NN_hi = torch.abs(D_H_NN)

    L_mag_D_lower_hi = hp.w_mag_r2 * masked_mean_hi(
        _hinge((c_lower_hi * mag_D_target_hi - mag_D_NN_hi) / norm_lamH),
        extra_mask=mask_D_H
    )
    L_mag_D_upper_hi = hp.w_mag_r2 * masked_mean_hi(
        _hinge((mag_D_NN_hi - c_upper_hi * mag_D_target_hi) / norm_lamH),
        extra_mask=mask_D_H
    )

    L_mag = (
            L_mag_H_lower + L_mag_H_upper
            + L_mag_H_lower2 + L_mag_H_upper2
            + L_mag_D_lower_hi + L_mag_D_upper_hi
            )

    return L_sign, L_mag
