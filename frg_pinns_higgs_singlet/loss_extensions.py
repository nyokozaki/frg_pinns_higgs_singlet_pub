import torch
import numpy as np
import torch.autograd as autograd
import math

from perturbation.config_params import finite_T, k_IR, t_range, tau_uv
from thermal_functions import u_thermal_finiteT, u_CW_zeroT
from running_couplings import tree_params, get_running_masses, get_running_quartics, get_running_couplings
[aH,aS,lamH,lamS,lamHS] = tree_params

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
    sigma_cut=0.1,
    mag_factor=0.1,
    margin=0
):

    if finite_T == False:
        t_in_list = (-1.0, 0.0)
        t_weights = (0.5, 0.5)

        L_sign_tot = 0.0
        L_mag_tot = 0.0

        '''
        for tval, w in zip(t_in_list, t_weights):
            Ls, Lm = _loss_sign_and_mag_couplings_tval(
                model,
                n_samp,
                rho_cut=rho_cut,
                sigma_cut=sigma_cut,
                mag_factor=mag_factor,
                margin=margin,
                tval=tval,
            )
            L_sign_tot = L_sign_tot + w * Ls
            L_mag_tot = L_mag_tot + w * Lm

        '''
        tval = -1.0
        L_sign_tot, L_mag_tot = _loss_sign_and_mag_couplings_tval(
            model,
            n_samp,
            rho_cut=rho_cut,
            sigma_cut=sigma_cut,
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
            sigma_cut=sigma_cut,
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
    sigma_cut=0.1,
    mag_factor=0.1,
    margin=0,
    tval=-1.0,
    mask_temp=10.0  # ソフトマスクの急峻さを決める温度パラメータ
):
    
    # relative weights
    w_sign_mix = hp.w_sign_mix
    w_mass_rho = hp.w_mass_rho
    w_mass_sigma = hp.w_mass_sigma
    w_sign_s2 = hp.w_sign_s2

    w_sign_r2 = hp.w_sign_r2

    w_mag_mass_rho = hp.w_mass_mag_rho
    w_mag_mass_sigma = hp.w_mass_mag_sigma


    rho_cw_cut = hp.rho_cw_cut
    sigma_cw_cut = hp.sigma_cw_cut


    t_in = torch.full((n_samp, 1), tval, device=device, requires_grad=False)

    if rho_cut is None:
        rho_cut_in = 1.0
    else:
        rho_cut_in = model.scale_rho(rho_cut)
    if sigma_cut is None:
        sigma_cut_in = 1.0
    else:
        sigma_cut_in = model.scale_sigma(sigma_cut)

    rho_in = torch.empty(n_samp, 1, device=device).uniform_(-1.0, rho_cut_in).requires_grad_(True)
    sigma_in = torch.empty(n_samp, 1, device=device).uniform_(-1.0, sigma_cut_in).requires_grad_(True)

    t_phys = model.unscale_t(t_in)
    rho_phys = model.unscale_rho(rho_in)
    sigma_phys = model.unscale_sigma(sigma_in)

    u, u_rho, u_sigma, u_rhorho, u_sigmasigma, u_rhosigma = model(t_in, rho_in, sigma_in)

    nn_H = 0.5 * u_rhorho
    nn_S = 0.5 * u_sigmasigma
    nn_HS = u_rhosigma

    t_phys0 = t_phys[0:1]
    #print(t_phys0)

    lamH_run, lamS_run, lamHS_run = get_running_quartics(t_phys0)

    lamH_run_b = lamH_run.expand_as(nn_H)
    lamS_run_b = lamS_run.expand_as(nn_S)
    lamHS_run_b = lamHS_run.expand_as(nn_HS)

    # mass terms
    mhsq_run, mssq_run = get_running_masses(t_phys0)

    kt = k_IR * torch.exp(-t_range + t_phys)

    aH_run = mhsq_run / (kt**2)
    aS_run = mssq_run / (kt**2)

    # quadratic-like terms (if the form of the potential is close to the renormalizable one)
    aH_NN = u_rho - u_rhorho * rho_phys - u_rhosigma * sigma_phys
    aS_NN = u_sigma - u_sigmasigma * sigma_phys - u_rhosigma * rho_phys

    lamH_tree = lamH
    lamS_tree = lamS
    lamHS_tree = lamHS

    D_H_target = (lamH_run_b) - lamH_tree
    D_S_target = (lamS_run_b) - lamS_tree
    D_HS_target = (lamHS_run_b) - lamHS_tree

    D_H_NN = nn_H - lamH_tree
    D_S_NN = nn_S - lamS_tree
    D_HS_NN = nn_HS - lamHS_tree

    aH_tree = aH * torch.exp(-2.0*t_phys0)
    aS_tree = aS * torch.exp(-2.0*t_phys0)
    
    F_H_target = (aH_run) - aH_tree
    F_S_target = (aS_run) - aS_tree

    F_H_NN = aH_NN - aH_tree
    F_S_NN = aS_NN - aS_tree

    z = torch.tensor(0.0, device=device)
    sign_terms = []

    eps = 1e-12
    

    # ==========================================
    # ソフトマスクの計算 (rho_phys, sigma_phys < x0 の領域に限定)
    # ==========================================
    mask_rho = torch.sigmoid(mask_temp * (rho_cw_cut - rho_phys))
    mask_sigma = torch.sigmoid(mask_temp * (sigma_cw_cut - sigma_phys))
    soft_mask = mask_rho * mask_sigma

    def masked_mean(loss_tensor, extra_mask=1.0):
        final_mask = soft_mask * extra_mask
        return (loss_tensor * final_mask).sum() / (final_mask.sum() + eps)
        
    
    # ==========================================
    # ペナルティ項の計算 (masked_mean を使用)
    # ==========================================
    norm_lamH = float(abs(lamH_tree)) + eps
    norm_lamS = float(abs(lamS_tree)) + eps
    norm_lamHS = float(abs(lamHS_tree)) + eps
    norm_aH = torch.abs(aH_tree) + eps
    norm_aS = torch.abs(aS_tree) + eps

    f_threshold = 1e-2
    d_threshold = 1e-2
    
    mask_F_H = (torch.abs(F_H_target / aH_tree) > f_threshold).float()
    mask_F_S = (torch.abs(F_S_target / aS_tree) > f_threshold).float()

    mask_D_H = (torch.abs(D_H_target / norm_lamH) > d_threshold).float()
    mask_D_S = (torch.abs(D_S_target / norm_lamS) > d_threshold).float()
    mask_D_HS = (torch.abs(D_HS_target / norm_lamHS) > d_threshold).float()

    sH = torch.sign(D_H_target)
    sign_terms.append(
        w_sign_r2 * masked_mean(
            _hinge(-(sH * D_H_NN / norm_lamH) + margin),
            extra_mask=mask_D_H
        )
    )

    sS = torch.sign(D_S_target)
    sign_terms.append(
        w_sign_s2 * masked_mean(
            _hinge(-(sS * D_S_NN / norm_lamS) + margin),
            extra_mask=mask_D_S
        )
    )

    sHS = torch.sign(D_HS_target)
    sign_terms.append(
        w_sign_mix * masked_mean(
            _hinge(-(sHS * D_HS_NN / norm_lamHS) + margin),
            extra_mask=mask_D_HS
        )
    )

    fH = torch.sign(F_H_target)
    fS = torch.sign(F_S_target)

    loss_H = _hinge(-(fH * F_H_NN / norm_aH) + margin)
    loss_S = _hinge(-(fS * F_S_NN / norm_aS) + margin)

    sign_terms.append(w_mass_rho * masked_mean(loss_H, extra_mask=mask_F_H))
    sign_terms.append(w_mass_sigma * masked_mean(loss_S, extra_mask=mask_F_S))

    L_sign = torch.stack(sign_terms).mean() if len(sign_terms) > 0 else z

    mag_H_target = torch.abs(F_H_target)
    mag_S_target = torch.abs(F_S_target)
    mag_H_NN = torch.abs(F_H_NN)
    mag_S_NN = torch.abs(F_S_NN)

    c_lower_H = hp.c_mag_lower_rho
    c_upper_H = hp.c_mag_upper_rho
    c_lower_S = hp.c_mag_lower_sigma
    c_upper_S = hp.c_mag_upper_sigma

    L_mag_H_lower = w_mag_mass_rho * masked_mean(
        _hinge((c_lower_H * mag_H_target - mag_H_NN) / norm_aH),
        extra_mask=mask_F_H
    )
    L_mag_H_upper = w_mag_mass_rho * masked_mean(
        _hinge((mag_H_NN - c_upper_H * mag_H_target) / norm_aH),
        extra_mask=mask_F_H
    )

    L_mag_S_lower = w_mag_mass_sigma * masked_mean(
        _hinge((c_lower_S * mag_S_target - mag_S_NN) / norm_aS),
        extra_mask=mask_F_S
    )
    L_mag_S_upper = w_mag_mass_sigma * masked_mean(
        _hinge((mag_S_NN - c_upper_S * mag_S_target) / norm_aS),
        extra_mask=mask_F_S
    )

    mag_H_target2 = torch.abs(D_H_target)
    mag_S_target2 = torch.abs(D_S_target)
    mag_HS_target2 = torch.abs(D_HS_target)

    mag_H_NN2 = torch.abs(D_H_NN)
    mag_S_NN2 = torch.abs(D_S_NN)
    mag_HS_NN2 = torch.abs(D_HS_NN)

    L_mag_H_lower2 = hp.w_mag_r2 * masked_mean(
        _hinge((c_lower_H * mag_H_target2 - mag_H_NN2) / norm_lamH),
        extra_mask=mask_D_H
    )
    L_mag_H_upper2 = hp.w_mag_r2 * masked_mean(
        _hinge((mag_H_NN2 - c_upper_H * mag_H_target2) / norm_lamH),
        extra_mask=mask_D_H
    )

    L_mag_S_lower2 = hp.w_mag_s2 * masked_mean(
        _hinge((c_lower_S * mag_S_target2 - mag_S_NN2) / norm_lamS),
        extra_mask=mask_D_S
    )
    L_mag_S_upper2 = hp.w_mag_s2 * masked_mean(
        _hinge((mag_S_NN2 - c_upper_S * mag_S_target2) / norm_lamS),
        extra_mask=mask_D_S
    )

    # HS mixing has no dedicated rho/sigma side of its own; tied to the sigma-side
    # band since it's the "extra" (singlet-involving) direction, same as w_mag_mix
    # being grouped with the sigma/S terms elsewhere in this file.
    L_mag_HS_lower2 = hp.w_mag_mix * masked_mean(
        _hinge((c_lower_S * mag_HS_target2 - mag_HS_NN2) / norm_lamHS),
        extra_mask=mask_D_HS
    )
    L_mag_HS_upper2 = hp.w_mag_mix * masked_mean(
        _hinge((mag_HS_NN2 - c_upper_S * mag_HS_target2) / norm_lamHS),
        extra_mask=mask_D_HS
    )

    L_mag = (
        L_mag_H_lower + L_mag_H_upper + L_mag_S_lower + L_mag_S_upper
        + L_mag_H_lower2 + L_mag_H_upper2
        + L_mag_S_lower2 + L_mag_S_upper2
        + L_mag_HS_lower2 + L_mag_HS_upper2
    )


    return L_sign, L_mag

def _loss_sign_and_mag_couplings_tval_finiteT(
    model,
    n_samp,
    rho_cut=0.1,
    sigma_cut=0.1,
    mag_factor=0.1,
    margin=0,
    tval=-1.0,
    mask_temp=10.0  # ソフトマスクの急峻さを決める温度パラメータ
):
    
    # relative weights
    w_mass_rho = hp.w_mass_rho
    w_mass_sigma = hp.w_mass_sigma
    w_sign_s2 = hp.w_sign_s2
    w_sign_r2 = hp.w_sign_r2
    #w_sign_mix = hp.w_sign_mix

    w_mag_mass_rho = hp.w_mass_mag_rho
    w_mag_mass_sigma = hp.w_mass_mag_sigma


    rho_cw_cut = hp.rho_cw_cut
    sigma_cw_cut = hp.sigma_cw_cut


    t_in = torch.full((n_samp, 1), tval, device=device, requires_grad=False)

    if rho_cut is None:
        rho_cut_in = 1.0
    else:
        rho_cut_in = model.scale_rho(rho_cut)
    if sigma_cut is None:
        sigma_cut_in = 1.0
    else:
        sigma_cut_in = model.scale_sigma(sigma_cut)

    rho_in = torch.empty(n_samp, 1, device=device).uniform_(-1.0, rho_cut_in).requires_grad_(True)
    sigma_in = torch.empty(n_samp, 1, device=device).uniform_(-1.0, sigma_cut_in).requires_grad_(True)

    t_phys = model.unscale_t(t_in)
    rho_phys = model.unscale_rho(rho_in)
    sigma_phys = model.unscale_sigma(sigma_in)

    u, u_rho, u_sigma, u_rhorho, u_sigmasigma, u_rhosigma = model(t_in, rho_in, sigma_in)

    nn_H = 0.5 * u_rhorho
    nn_S = 0.5 * u_sigmasigma
    nn_HS = u_rhosigma

    t_phys0 = t_phys[0:1]
    #print(t_phys0)

    lamH_run, lamS_run, lamHS_run = get_running_quartics(t_phys0)

    lamH_run_b = lamH_run.expand_as(nn_H)
    lamS_run_b = lamS_run.expand_as(nn_S)
    lamHS_run_b = lamHS_run.expand_as(nn_HS)

    # mass terms
    mhsq_run, mssq_run = get_running_masses(t_phys0)

    kt = k_IR * torch.exp(-t_range + t_phys)

    aH_run = mhsq_run / (kt**2)
    aS_run = mssq_run / (kt**2)    


    # thermal correction
    rho1 = rho_phys.clone()
    sigma1 = sigma_phys.clone()
    rho1.requires_grad_(True)
    sigma1.requires_grad_(True)

    t_phys1 = t_phys.clone().detach()
    u_th1 = u_thermal_finiteT(t_phys1, rho1, sigma1)

    uth_r1, uth_s1 = torch.autograd.grad(
        u_th1,
        (rho1, sigma1),
        grad_outputs=torch.ones_like(u_th1),
        create_graph=True
    )

    (uth_rr,) = torch.autograd.grad(
        uth_r1,
        rho1,
        grad_outputs=torch.ones_like(uth_r1),
        create_graph=True
    )
    (uth_ss,) = torch.autograd.grad(
        uth_s1,
        sigma1,
        grad_outputs=torch.ones_like(uth_s1),
        create_graph=True
    )
    (uth_rs,) = torch.autograd.grad(
        uth_r1,
        sigma1,
        grad_outputs=torch.ones_like(uth_r1),
        create_graph=True
    )

    del_aH = uth_r1 - uth_rr * rho_phys - uth_rs * sigma_phys
    del_aS = uth_s1 - uth_ss * sigma_phys - uth_rs * rho_phys

    # for thermal correction safe
    rho_cut = 0.05
    sigma_cut = 0.05
    temp = 2000.0

    #rho_cut = 0.15
    #sigma_cut = 0.15
    #temp = 200.0

    mask_rhoT = torch.sigmoid(temp * (rho1 - rho_cut))
    mask_sigmaT = torch.sigmoid(temp * (sigma1 - sigma_cut))
    soft_maskT = mask_rhoT * mask_sigmaT

    aH_run = aH_run + del_aH.detach()
    aS_run = aS_run + del_aS.detach()


    # quadratic-like terms (if the form of the potential is close to the renormalizable one)
    aH_NN = u_rho - u_rhorho * rho_phys - u_rhosigma * sigma_phys
    aS_NN = u_sigma - u_sigmasigma * sigma_phys - u_rhosigma * rho_phys

    lamH_tree = lamH
    lamS_tree = lamS
    lamHS_tree = lamHS

    D_H_target = (lamH_run_b) - lamH_tree
    D_S_target = (lamS_run_b) - lamS_tree
    D_HS_target = (lamHS_run_b) - lamHS_tree

    D_H_NN = nn_H - lamH_tree
    D_S_NN = nn_S - lamS_tree
    D_HS_NN = nn_HS - lamHS_tree

    aH_tree = aH * torch.exp(-2.0*t_phys0)
    aS_tree = aS * torch.exp(-2.0*t_phys0)
    
    F_H_target = (aH_run) - aH_tree
    F_S_target = (aS_run) - aS_tree

    # --- 追加: 0付近を無視するためのマスク ---
    f_threshold = 1e-2  # 値のスケールに合わせて適宜調整してください
    mask_F_H = (torch.abs(F_H_target/aH_tree) > f_threshold).float()
    mask_F_S = (torch.abs(F_S_target/aS_tree) > f_threshold).float()

    #mask_F_H = 1.0
    #mask_F_S = 1.0
    # ----------------------------------------

    F_H_NN = aH_NN - aH_tree
    F_S_NN = aS_NN - aS_tree

    z = torch.tensor(0.0, device=device)
    sign_terms = []

    eps = 1e-12
    

    # ==========================================
    # ソフトマスクの計算 (rho_phys, sigma_phys < x0 の領域に限定)
    # ==========================================
    mask_rho = torch.sigmoid(mask_temp * (rho_cw_cut - rho_phys))
    mask_sigma = torch.sigmoid(mask_temp * (sigma_cw_cut - sigma_phys))
    soft_mask = mask_rho * mask_sigma

    soft_mask = soft_mask * soft_maskT

    def masked_mean(loss_tensor, extra_mask=1.0):
        """マスクによる加重平均を計算し、有効なサンプルの領域のみでLossを評価する"""
        final_mask = soft_mask * extra_mask
        return (loss_tensor * final_mask).sum() / (final_mask.sum() + eps)

    # ==========================================
    # rho > rho_cw_cut / sigma > sigma_cw_cut 側のソフトマスク
    # (mask_rho / mask_sigma の相補マスク、field ごとに独立)
    # これらの領域では対応する場の quartic self-coupling (D_H_NN / D_S_NN)
    # のみを sign+magnitude で縛る。mass-term や Higgs-singlet mixing
    # (D_HS) はここでは触らない。
    # ==========================================
    mask_rho_hi = torch.sigmoid(mask_temp * (rho_phys - rho_cw_cut))
    mask_sigma_hi = torch.sigmoid(mask_temp * (sigma_phys - sigma_cw_cut))
    soft_mask_rho_hi = mask_rho_hi * soft_maskT
    soft_mask_sigma_hi = mask_sigma_hi * soft_maskT

    def masked_mean_rho_hi(loss_tensor, extra_mask=1.0):
        final_mask = soft_mask_rho_hi * extra_mask
        return (loss_tensor * final_mask).sum() / (final_mask.sum() + eps)

    def masked_mean_sigma_hi(loss_tensor, extra_mask=1.0):
        final_mask = soft_mask_sigma_hi * extra_mask
        return (loss_tensor * final_mask).sum() / (final_mask.sum() + eps)


    norm_lamH = float(abs(lamH_tree)) + eps
    norm_lamS = float(abs(lamS_tree)) + eps
    norm_lamHS = float(abs(lamHS_tree)) + eps

    d_threshold = 1e-2
    mask_D_H = (torch.abs(D_H_target / norm_lamH) > d_threshold).float()
    mask_D_S = (torch.abs(D_S_target / norm_lamS) > d_threshold).float()

    # ==========================================
    # ペナルティ項の計算 (masked_mean を使用)
    # ==========================================
    '''
    sH = torch.sign(D_H_target)
    sign_terms.append(w_sign_r2*masked_mean(_hinge(-(sH * D_H_NN / norm_lamH) + margin)))

    sS = torch.sign(D_S_target)
    sign_terms.append(w_sign_s2 * masked_mean(_hinge(-(sS * D_S_NN / norm_lamS) + margin)))

    sHS = torch.sign(D_HS_target)
    sign_terms.append(w_sign_mix * masked_mean(_hinge(-(sHS * D_HS_NN / norm_lamHS) + margin)))
    '''

    #
    # mass terms
    #
    norm_aH = torch.abs(aH_tree) + eps
    norm_aS = torch.abs(aS_tree) + eps

    fH = torch.sign(F_H_target)
    fS = torch.sign(F_S_target)

    loss_H = _hinge(-(fH * F_H_NN / norm_aH) + margin)
    loss_S = _hinge(-(fS * F_S_NN / norm_aS) + margin)

    sign_terms.append(w_mass_rho * masked_mean(loss_H, extra_mask=mask_F_H))
    sign_terms.append(w_mass_sigma * masked_mean(loss_S, extra_mask=mask_F_S))

    # rho > rho_cw_cut: Higgs quartic self-coupling sign constraint
    # (mass-term and mixing are intentionally left unconstrained here)
    sH2 = torch.sign(D_H_target)
    loss_D_H_sign = _hinge(-(sH2 * D_H_NN / norm_lamH) + margin)
    sign_terms.append(w_sign_r2 * masked_mean_rho_hi(loss_D_H_sign, extra_mask=mask_D_H))

    # sigma > sigma_cw_cut: singlet quartic self-coupling sign constraint
    sS2 = torch.sign(D_S_target)
    loss_D_S_sign = _hinge(-(sS2 * D_S_NN / norm_lamS) + margin)
    sign_terms.append(w_sign_s2 * masked_mean_sigma_hi(loss_D_S_sign, extra_mask=mask_D_S))

    L_sign = torch.stack(sign_terms).mean() if len(sign_terms) > 0 else z
    
    #############################
    #
    #   magnitude loss
    #   For the mass-like combination we use the same logic as that of the zero T
    #
    #############################

    #L_mag = z

    mag_H_target = torch.abs(F_H_target)
    mag_S_target = torch.abs(F_S_target)

    mag_H_NN = torch.abs(F_H_NN)
    mag_S_NN = torch.abs(F_S_NN)

    c_lower_H = hp.c_mag_lower_rho
    c_upper_H = hp.c_mag_upper_rho
    c_lower_S = hp.c_mag_lower_sigma
    c_upper_S = hp.c_mag_upper_sigma

    #print(c_lower_H,c_upper_H,c_lower_S,c_upper_S)

    L_mag_H_lower = w_mag_mass_rho*masked_mean(_hinge((c_lower_H * mag_H_target - mag_H_NN)/norm_aH), extra_mask=mask_F_H)
    L_mag_H_upper = w_mag_mass_rho*masked_mean(_hinge((mag_H_NN - c_upper_H * mag_H_target)/norm_aH), extra_mask=mask_F_H)

    L_mag_S_lower = w_mag_mass_sigma*masked_mean(_hinge((c_lower_S * mag_S_target - mag_S_NN)/norm_aS), extra_mask=mask_F_S)
    L_mag_S_upper = w_mag_mass_sigma*masked_mean(_hinge((mag_S_NN - c_upper_S * mag_S_target)/norm_aS), extra_mask=mask_F_S)


    # self coupling magnitude
    # uppper lower bounds are taken loosely
    # self couplings we use the same weights as those for the mass terms

    #c_lower = 0.1
    c_upper = 0.99

    #mag_H_target2 = torch.abs(D_H_target)
    #mag_S_target2 = torch.abs(D_S_target)
    #mag_HS_target2 = torch.abs(D_HS_target)

    mag_H_NN2 = torch.abs(D_H_NN/lamH_tree)
    mag_S_NN2 = torch.abs(D_S_NN/lamS_tree)
    mag_HS_NN2 = torch.abs(D_HS_NN/lamHS_tree)


    L_mag_H_lower2 = z
    L_mag_H_upper2 = hp.w_mag_r2*masked_mean(_hinge(mag_H_NN2 - c_upper))

    L_mag_S_lower2 = z
    L_mag_S_upper2 = hp.w_mag_s2*masked_mean(_hinge(mag_S_NN2 - c_upper))

    L_mag_HS_lower2 = z
    L_mag_HS_upper2 = hp.w_mag_mix*masked_mean(_hinge(mag_HS_NN2 - c_upper))

    # rho > rho_cw_cut / sigma > sigma_cw_cut: quartic self-coupling magnitude
    # band vs the RGE-running target (D_H_target / D_S_target), field-independent
    # masks. Mixing (D_HS) is left untouched.
    mag_D_H_target_hi = torch.abs(D_H_target)
    mag_D_H_NN_hi = torch.abs(D_H_NN)
    L_mag_D_H_lower_hi = hp.w_mag_r2_hi * masked_mean_rho_hi(
        _hinge((c_lower_H * mag_D_H_target_hi - mag_D_H_NN_hi) / norm_lamH),
        extra_mask=mask_D_H
    )
    L_mag_D_H_upper_hi = hp.w_mag_r2_hi * masked_mean_rho_hi(
        _hinge((mag_D_H_NN_hi - c_upper_H * mag_D_H_target_hi) / norm_lamH),
        extra_mask=mask_D_H
    )

    mag_D_S_target_hi = torch.abs(D_S_target)
    mag_D_S_NN_hi = torch.abs(D_S_NN)
    L_mag_D_S_lower_hi = hp.w_mag_s2_hi * masked_mean_sigma_hi(
        _hinge((c_lower_S * mag_D_S_target_hi - mag_D_S_NN_hi) / norm_lamS),
        extra_mask=mask_D_S
    )
    L_mag_D_S_upper_hi = hp.w_mag_s2_hi * masked_mean_sigma_hi(
        _hinge((mag_D_S_NN_hi - c_upper_S * mag_D_S_target_hi) / norm_lamS),
        extra_mask=mask_D_S
    )

    '''
    L_mag_H_lower2 = w_mag_mass_rho*masked_mean(_hinge((c_lower * mag_H_target2 - mag_H_NN2)/norm_lamH))
    L_mag_H_upper2 = w_mag_mass_rho*masked_mean(_hinge((mag_H_NN2 - c_upper * mag_H_target2)/norm_lamH))

    L_mag_S_lower2 = w_mag_mass_sigma*masked_mean(_hinge((c_lower * mag_S_target2 - mag_S_NN2)/norm_lamS))
    L_mag_S_upper2 = w_mag_mass_sigma*masked_mean(_hinge((mag_S_NN2 - c_upper * mag_S_target2)/norm_lamS))

    L_mag_HS_lower2 = hp.w_mod_mix*masked_mean(_hinge((c_lower * mag_HS_target2 - mag_HS_NN2)/norm_lamHS))
    L_mag_HS_upper2 = hp.w_mod_mix*masked_mean(_hinge((mag_HS_NN2 - c_upper * mag_HS_target2)/norm_lamHS))
    '''

    L_mag = (
            L_mag_H_lower + L_mag_H_upper + L_mag_S_lower + L_mag_S_upper
            + L_mag_H_lower2 + L_mag_H_upper2
            + L_mag_S_lower2 + L_mag_S_upper2
            + L_mag_HS_lower2 + L_mag_HS_upper2
            + L_mag_D_H_lower_hi + L_mag_D_H_upper_hi
            + L_mag_D_S_lower_hi + L_mag_D_S_upper_hi
            )

    return L_sign, L_mag

