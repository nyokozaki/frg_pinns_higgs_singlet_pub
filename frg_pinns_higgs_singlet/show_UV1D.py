import gc
import torch
import torch.nn as nn
import numpy as np
import torch.autograd as autograd
import matplotlib.pyplot as plt

# ============================================================
# device / dtype / seed
# ============================================================
import hyperparams as hp
device = hp.device

torch.set_default_dtype(torch.float32)

def set_seed(seed=0):
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed(seed)
        torch.cuda.manual_seed_all(seed)

set_seed(1234)


# ============================================================
# UV parameters / running couplings (新学習コードと完全一致版)
# ============================================================

from perturbation.config_params import tau_uv, finite_T, k_IR, t_range, fixed_tau
from thermal_functions import get_u_ring
from running_couplings import tree_params, get_running_couplings, get_running_masses, get_running_quartics
[aH,aS,lamH,lamS,lamHS] = tree_params

from my_networks import u_tree_exact, BaseNet, WrappedPotentialNet

kloop = 1.0 / (12.0 * torch.pi**2)
NGS = 3.0


def get_eta_rho(t):
    with torch.no_grad():
        g1, g2, yt, _, _ = get_running_couplings(t)

    gamma = (1.0 / (16.0 * torch.pi**2)) * (
        3.0 * yt**2
        - 3.0 * g1**2 / 4.0
        - 9.0 * g2**2 / 4.0
    )
    return 2.0 * gamma


def get_eta_sigma(t):
    with torch.no_grad():
        _, _, _, lamS_t, lamHS_t = get_running_couplings(t)

    # gamma_S from PyR@TE3 for this model (matches frg_pinn.tex Sec. 2.3):
    #   gamma_S = (16 pi^2)^-2 (lambda_HS^2 + 3 lambda_S^2),  eta_S = 2 gamma_S.
    # Pre-2026-09 runs used (18 lamS^2 + 2 lamHS^2) (1-loop beta_lamS coeff as a
    # proxy); eta_S ~ 1e-4 and its flow effect ~1e-5, so snapshots are unaffected.
    gamma = (1.0 / (16.0 * torch.pi**2))**2 * (
        3.0 * lamS_t**2 + 1.0 * lamHS_t**2
    )
    return 2.0 * gamma


######################################################################

# ============================================================
# Time block mapping (2 blocks)
# TIME CONVENTION (high energy -> low energy):
#   block1: t in [-1.0,  0.0]   (UV side)
#   block2: t in [-2.0, -1.0]
# ============================================================

def get_time_block_info(t_val):
    """2-block structure"""
    if -1.0 <= t_val <= 0.0:
        return 1, -1.0, 0.0
    elif -2.0 <= t_val < -1.0:
        return 2, -2.0, -1.0
    else:
        raise ValueError("t_val must be in [-2.0, 0.0].")


# ============================================================
# Plotting helpers
# ============================================================

TORCH_REAL_DTYPE = torch.get_default_dtype()

def my_to(x):
    if torch.is_tensor(x):
        return x.to(dtype=TORCH_REAL_DTYPE, device=device)
    else:
        return torch.as_tensor(x, dtype=TORCH_REAL_DTYPE, device=device)


def make_model_for_range(model_path, t0, t1, rho0, rho1, sigma0, sigma1):
    base_net = BaseNet().to(device)
    model = WrappedPotentialNet(
        base_net,
        T_min=t0, T_max=t1,
        Rho_min=rho0, Rho_max=rho1,
        Sigma_min=sigma0, Sigma_max=sigma1
    ).to(device)
    model.load_state_dict(torch.load(model_path, map_location=device))
    model.eval()
    return model


def eval_model_slice_sigma_fixed(model, t_val, rho_np, sigma_val):
    rho_t = my_to(rho_np.reshape(-1, 1))
    t_t = my_to(np.full((len(rho_np), 1), t_val))
    sigma_t = my_to(np.full((len(rho_np), 1), sigma_val))

    t_in = model.scale_t(t_t)
    rho_in = model.scale_rho(rho_t)
    sigma_in = model.scale_sigma(sigma_t)

    with torch.no_grad():
        u_pred = model(t_in, rho_in, sigma_in)[0].detach().cpu().numpy().flatten()

    with torch.no_grad():
        u_tree = u_tree_exact(t_t, rho_t, sigma_t).detach().cpu().numpy().flatten()

    return u_tree, u_pred


def eval_model_point(model, t_val, rho_val, sigma_val):
    t_t = my_to([[t_val]])
    rho_t = my_to([[rho_val]])
    sigma_t = my_to([[sigma_val]])

    t_in = model.scale_t(t_t)
    rho_in = model.scale_rho(rho_t)
    sigma_in = model.scale_sigma(sigma_t)

    with torch.no_grad():
        u_pred = model(t_in, rho_in, sigma_in)[0].item()

    return u_pred


# ============================================================
# Stitched global rho-slice (5 rho blocks)
# ============================================================

def stitched_slice_sigma_fixed_matched(
    t_val,
    sigma_fixed=0.0,
    sigma0=0.0,
    sigma1=0.350,
    n_per_block=200
):
    """5 rho blocks with drho=0.35, overlap=0.02"""
    block_id, t0, t1 = get_time_block_info(t_val)

    n_rho_blocks = 5
    drho = 0.350
    rho_overlap = 0.02

    rho_blocks = []
    u_tree_blocks = []
    u_pred_blocks = []

    model0_for_anchor = None

    for irho in range(n_rho_blocks):
        rho_start = irho * drho
        if irho < n_rho_blocks - 1:
            rho_end = (irho + 1) * drho + rho_overlap
            rho_end0 = (irho + 1) * drho
        else:
            rho_end = (irho + 1) * drho
            rho_end0 = (irho + 1) * drho

        file_model = f"./data_UV/model_higgs_singlet_u_rho{irho}_block{block_id}.pt"

        model = make_model_for_range(
            file_model,
            t0, t1, rho_start, rho_end, sigma0, sigma1
        )

        if irho == 0:
            model0_for_anchor = model

        rho_np = np.linspace(rho_start, rho_end0, n_per_block)
        u_tree, u_pred = eval_model_slice_sigma_fixed(model, t_val, rho_np, sigma_fixed)

        rho_blocks.append(rho_np)
        u_tree_blocks.append(u_tree)
        u_pred_blocks.append(u_pred)

        if irho != 0:
            del model
            gc.collect()
            if torch.cuda.is_available():
                torch.cuda.empty_cache()

    # Anchor condition: u(t, rho=0, sigma=0) = 0
    u_anchor = eval_model_point(model0_for_anchor, t_val, rho_val=0.0, sigma_val=0.0)
    shifts = [-u_anchor]

    del model0_for_anchor
    gc.collect()
    if torch.cuda.is_available():
        torch.cuda.empty_cache()

    # Continuity shifts
    for irho in range(1, n_rho_blocks):
        left_val = u_pred_blocks[irho - 1][-1] + shifts[irho - 1]
        right_val = u_pred_blocks[irho][0]
        c_i = left_val - right_val
        shifts.append(c_i)

    # Concatenate
    rho_all_list = []
    u_tree_all_list = []
    u_pred_all_list = []

    for irho in range(n_rho_blocks):
        rho_np = rho_blocks[irho]
        u_tree = u_tree_blocks[irho]
        u_pred = u_pred_blocks[irho] + shifts[irho]

        if irho == 0:
            rho_all_list.append(rho_np)
            u_tree_all_list.append(u_tree)
            u_pred_all_list.append(u_pred)
        else:
            rho_all_list.append(rho_np[1:])
            u_tree_all_list.append(u_tree[1:])
            u_pred_all_list.append(u_pred[1:])

    rho_all = np.concatenate(rho_all_list)
    u_tree_all = np.concatenate(u_tree_all_list)
    u_pred_all = np.concatenate(u_pred_all_list)

    return rho_all, u_tree_all, u_pred_all, np.array(shifts)

# ============================================================
# Plot stitched t-slices (修正版)
# ============================================================

def plot_u(sigma_fixed=0.0, t_list=None, ncols=None):
    # t_list: RG-times to show (default [0, -2], the draft two-panel layout).
    # ncols : panels per row (default min(len(t_list), 3)).
    if t_list is None:
        t_list = [0.0, -2.0]
    n_t = len(t_list)
    if ncols is None:
        ncols = min(n_t, 3)
    nrows = -(-n_t // ncols)

    fig, axes = plt.subplots(nrows, ncols,
                             figsize=(6.0 * ncols, 4.5 * nrows),
                             sharex=True, sharey=False, squeeze=False)
    axes = axes.flatten()
    
    import thermal_np as thermal
    from thermal_functions import u_CW_zeroT

    for i, t_val in enumerate(t_list):
        rho_all, u_tree_all, u_pred_all, shifts = stitched_slice_sigma_fixed_matched(
            t_val=t_val,
            sigma_fixed=sigma_fixed,
            sigma0=0.0,
            sigma1=0.350,
            n_per_block=200
        )

        rho_np = rho_all
        t_np = np.full_like(rho_np, t_val)
        sigma_np = np.full_like(rho_np, sigma_fixed)

        #finite_T = True
        # --- リング項の計算と加算 ---
        if finite_T:
            # NumPy配列をPyTorchテンソルに変換
            t_tensor = my_to(t_np)
            rho_tensor = my_to(rho_np)
            rho_zero_tensor = my_to(np.zeros_like(rho_np)) # rho=0 のテンソル
            
            with torch.no_grad():
                u_ring_tensor = get_u_ring(t_tensor, rho_tensor)
                u_ring_zero_tensor = get_u_ring(t_tensor, rho_zero_tensor)
                
                # rho=0 での定数項を差し引く
                u_ring_net = u_ring_tensor - u_ring_zero_tensor
                u_ring_np = u_ring_net.detach().cpu().numpy().flatten()
            
            # NNの予言にリング項を足し合わせる
            u_pred_with_ring = u_pred_all + u_ring_np
        else:
            # 有限温度でない場合はリング項を加算しない
            u_pred_with_ring = u_pred_all
        # ---------------------------

        u_pert_vals = thermal.u_pert(t_np, rho_np, sigma_np)
        u_pert_offset = thermal.u_pert(t_np, np.zeros_like(rho_np), np.zeros_like(sigma_np))
        u_pert_vals = u_pert_vals - u_pert_offset

        u_finiteT_vals = thermal.u_seed_finiteT(t_np, rho_np, sigma_np)


        rho_tor=my_to(rho_np)
        sigma_tor=my_to(sigma_np)
        t_tor = my_to(t_np)

        rho0_tor=my_to(rho_np*0)
        sigma0_tor=my_to(sigma_np*0)
        t_tor = my_to(t_np)

        u_cw = u_CW_zeroT(t_tor,rho_tor,sigma_tor).detach().cpu().numpy().flatten()
        u_cw0 = u_CW_zeroT(t_tor,rho0_tor,sigma0_tor).detach().cpu().numpy().flatten()

        u_pert_vals = u_pert_vals + (u_cw - u_cw0)*np.heaviside(-(t_np + 2.0)+1e-5, 1.0)



        ax = axes[i]
        ax.plot(rho_all, u_tree_all, 'k-', lw=1.5, label='tree')
        # ここを u_pred_with_ring に変更
        ax.plot(rho_all, u_pred_with_ring, 'r--', lw=1.5, label='NN+ring stitched')
        ax.plot(rho_all, u_pert_vals,      color='tab:blue',  linestyle='-.', lw=1.5, label='RGE (T=0)')
        ax.plot(rho_all, u_finiteT_vals,  color='tab:green', linestyle=':',  lw=1.5, label='finite T')
        
        # Show rho block boundaries
        _dx = 0.350
        _xtot = 0.350 * 5
        for rb in np.arange(_dx, _xtot, _dx):
            ax.axvline(rb, color='gray', lw=0.8, ls=':', alpha=0.5)

        block_id, _, _ = get_time_block_info(t_val)
        ax.set_title(f't={t_val:.2f}, sigma={sigma_fixed:.2f}, block={block_id}')
        ax.set_xlabel('rho')
        ax.set_xlim(0.0, _xtot)
        ax.grid(alpha=0.3)

        if i % ncols == 0:
            ax.set_ylabel('u')

        print(f"t={t_val:.2f}, block={block_id}, shifts={shifts}")

    for j in range(len(t_list), len(axes)):
        fig.delaxes(axes[j])

    axes[0].legend()
    fig.tight_layout()
    plt.show()

# ============================================================
# Training history plots (5 rho blocks, 2 time blocks)
# ============================================================

def plot_training_histories_all_rho():
    n_rho_blocks = 5
    fig, axes = plt.subplots(3, 2, figsize=(10, 15), sharey=True)
    axes = axes.flatten()

    for irho in range(n_rho_blocks):
        file_hist_1 = f"./data_UV/history_higgs_singlet_u_rho{irho}_block1.npy"
        file_hist_2 = f"./data_UV/history_higgs_singlet_u_rho{irho}_block2.npy"

        hist1 = np.load(file_hist_1)
        hist2 = np.load(file_hist_2)

        ax = axes[irho]
        # History columns: [loss, Lpde, Lbc, Loverlap, Lsign, Lmag]
        ax.plot(hist1[:, 1], label='pde b1 [-1.0, 0.0]')
        ax.plot(hist1[:, 2], label='bc  b1 [-1.0, 0.0]')

        ax.plot(hist2[:, 1], '--', label='pde b2 [-2.0,-1.0]')
        ax.plot(hist2[:, 2], '--', label='bc  b2 [-2.0,-1.0]')

        ax.set_yscale('log')
        ax.set_title(f'irho={irho}')
        ax.set_xlabel('epoch')
        if irho % 2 == 0:
            ax.set_ylabel('loss')
        ax.grid(alpha=0.3)
        ax.set_ylim(1e-8, 0.1) 

    if n_rho_blocks < len(axes):
        fig.delaxes(axes[-1])

    axes[0].legend(fontsize=8)
    fig.suptitle(
        'Training losses for all rho-blocks (UV->IR, block1=UV, block2=IR)',
        y=1.00
    )
    fig.tight_layout()
    plt.show()


# ============================================================
# Helper functions for residual check
# ============================================================

def coth(x, eps=1e-10):
    return torch.cosh(x) / (torch.sinh(x) + eps)


def get_rho_block_index(rho_val):
    """5 blocks with drho=0.350"""
    irho = int(np.floor(rho_val / 0.350))
    if irho < 0:
        irho = 0
    if irho > 4:
        irho = 4
    return irho


# ============================================================
# Load models for residual check (5 rho blocks, 2 time blocks)
# ============================================================

def load_models_for_sigma_fixed_blockwise(rho0, rho1, sigma0, sigma1):
    """5 rho blocks, 2 time blocks"""
    models_block1 = {}
    models_block2 = {}

    n_rho_blocks = 5
    drho = 0.350
    rho_overlap = 0.02

    for irho in range(n_rho_blocks):
        rho_start = irho * drho
        if irho < n_rho_blocks - 1:
            rho_end = (irho + 1) * drho + rho_overlap
        else:
            rho_end = (irho + 1) * drho

        file1 = f"./data_UV/model_higgs_singlet_u_rho{irho}_block1.pt"
        file2 = f"./data_UV/model_higgs_singlet_u_rho{irho}_block2.pt"

        models_block1[irho] = make_model_for_range(
            file1,
            -1.0, 0.0, rho_start, rho_end, sigma0, sigma1
        )
        models_block2[irho] = make_model_for_range(
            file2,
            -2.0, -1.0, rho_start, rho_end, sigma0, sigma1
        )

    return models_block1, models_block2


# ============================================================
# Residual check (2 blocks, running couplings)
# ============================================================

LPAp = 1.0


def residual_check_sigma_fixed_blockwise(
    models_block1,
    models_block2,
    t0,
    t1,
    rho0,
    rho1,
    sigma_fixed=0.2,
    invert_t_axis=False
):
    Nt_check = 121
    Nrho_check = 161

    t_vals = np.linspace(t0, t1, Nt_check)
    rho_vals = np.linspace(rho0, rho1, Nrho_check)

    u_np = np.zeros((Nt_check, Nrho_check))
    u_t_np = np.zeros_like(u_np)
    rhs_np = np.zeros_like(u_np)
    res_np = np.zeros_like(u_np)
    loop_np = np.zeros_like(u_np)

    for it, t_val in enumerate(t_vals):
        if -1.0 <= t_val <= 0.0:
            t_block_id = 1
        elif -2.0 <= t_val < -1.0:
            t_block_id = 2
        else:
            raise ValueError("t_val must be in [-2.0, 0]")

        for ir, rho_val in enumerate(rho_vals):
            irho = get_rho_block_index(rho_val)

            if t_block_id == 1:
                model = models_block1[irho]
            else:
                model = models_block2[irho]

            t_grid = my_to([[t_val]])
            rho_grid = my_to([[rho_val]])
            sigma_grid = my_to([[sigma_fixed]])

            t_in = model.scale_t(t_grid)
            rho_in = model.scale_rho(rho_grid)
            sigma_in = model.scale_sigma(sigma_grid)

            t_in.requires_grad_(True)
            rho_in.requires_grad_(True)
            sigma_in.requires_grad_(True)

            u_tuple = model(t_in, rho_in, sigma_in)
            u = u_tuple[0]

            u_t_in = autograd.grad(
                outputs=u,
                inputs=t_in,
                grad_outputs=torch.ones_like(u),
                create_graph=True,
                retain_graph=True,
            )[0]

            u_rho_in = autograd.grad(
                outputs=u,
                inputs=rho_in,
                grad_outputs=torch.ones_like(u),
                create_graph=True,
                retain_graph=True,
            )[0]

            u_sigma_in = autograd.grad(
                outputs=u,
                inputs=sigma_in,
                grad_outputs=torch.ones_like(u),
                create_graph=True,
                retain_graph=True,
            )[0]

            u_rhorho_in = autograd.grad(
                outputs=u_rho_in,
                inputs=rho_in,
                grad_outputs=torch.ones_like(u_rho_in),
                create_graph=True,
                retain_graph=True,
            )[0]

            u_sigmasigma_in = autograd.grad(
                outputs=u_sigma_in,
                inputs=sigma_in,
                grad_outputs=torch.ones_like(u_sigma_in),
                create_graph=True,
                retain_graph=True,
            )[0]

            u_rhosigma_in = autograd.grad(
                outputs=u_rho_in,
                inputs=sigma_in,
                grad_outputs=torch.ones_like(u_rho_in),
                create_graph=True,
                retain_graph=True,
            )[0]

            A_T = model.A_T
            A_RHO = model.A_RHO
            A_SIGMA = model.A_SIGMA

            u_t = A_T * u_t_in
            u_rho = A_RHO * u_rho_in
            u_sigma = A_SIGMA * u_sigma_in
            u_rhorho = A_RHO * A_RHO * u_rhorho_in
            u_sigmasigma = A_SIGMA * A_SIGMA * u_sigmasigma_in
            u_rhosigma = A_RHO * A_SIGMA * u_rhosigma_in

            rho_phys = model.unscale_rho(rho_in)
            sigma_phys = model.unscale_sigma(sigma_in)

            eps = 1e-10

            mG2 = u_rho

            M11 = u_rho + 2.0 * rho_phys * u_rhorho
            M22 = u_sigma + 2.0 * sigma_phys * u_sigmasigma
            
            M12 = 2.0 * torch.sqrt(torch.clamp(rho_phys * sigma_phys, min=eps)) * u_rhosigma

            trM2 = M11 + M22
            detM2 = M11 * M22 - M12 * M12

            disc = trM2 * trM2 - 4.0 * detM2
            sqrt_disc = torch.sqrt(torch.clamp(disc, min=eps))

            m1_sq = 0.5 * (trM2 - sqrt_disc)
            m2_sq = 0.5 * (trM2 + sqrt_disc)

            termG_0 = 1.0 / torch.sqrt(torch.clamp(1.0 + mG2, min=eps))
            term1_0 = 1.0 / torch.sqrt(torch.clamp(1.0 + m1_sq, min=eps))
            term2_0 = 1.0 / torch.sqrt(torch.clamp(1.0 + m2_sq, min=eps))

            # Running couplings を使用
            t_phys_lp = model.unscale_t(t_in).detach()
            with torch.no_grad():
                g1, g2, yt, lamS_t, lamHS_t = get_running_couplings(t_phys_lp)

            mW2 = 0.5 * g2**2 * rho_phys
            mZ2 = 0.5 * (g2**2 + g1**2) * rho_phys
            mt2 = yt**2 * rho_phys

            termW0 = 1.0 / torch.sqrt(torch.clamp(1.0 + mW2, min=eps))
            termZ0 = 1.0 / torch.sqrt(torch.clamp(1.0 + mZ2, min=eps))
            termT0 = 1.0 / torch.sqrt(torch.clamp(1.0 + mt2, min=eps))

            #print(finite_T)
            if finite_T:
                t_phys = model.unscale_t(t_in)
                tau_t = tau_uv * torch.exp(-t_phys)

                EG = torch.sqrt(torch.clamp(1.0 + mG2, min=eps))
                E1 = torch.sqrt(torch.clamp(1.0 + m1_sq, min=eps))
                E2 = torch.sqrt(torch.clamp(1.0 + m2_sq, min=eps))

                E_W = torch.sqrt(torch.clamp(1.0 + mW2, min=eps))
                E_Z = torch.sqrt(torch.clamp(1.0 + mZ2, min=eps))
                E_T = torch.sqrt(torch.clamp(1.0 + mt2, min=eps))

                factorG = coth(EG / (2.0 * tau_t), eps=0.0)
                factor1 = coth(E1 / (2.0 * tau_t), eps=0.0)
                factor2 = coth(E2 / (2.0 * tau_t), eps=0.0)

                factor_W = coth(E_W / (2.0 * tau_t), eps=0.0)
                factor_Z = coth(E_Z / (2.0 * tau_t), eps=0.0)
                factor_T = torch.tanh(E_T / (2.0 * tau_t))

                termG = termG_0 * factorG
                term1 = term1_0 * factor1
                term2 = term2_0 * factor2

                termW = termW0 * factor_W
                termZ = termZ0 * factor_Z
                termT = termT0 * factor_T

            else:
                termG, term1, term2 = termG_0, term1_0, term2_0
                termW, termZ, termT = termW0, termZ0, termT0

            loop_sum = (
                3.0 * termG
                + term1
                + term2
                + 6.0 * termW
                + 3.0 * termZ
                - 12.0 * termT
            )

            rloop = kloop * loop_sum
            loop_term = torch.clamp(rloop, min=-2.0, max=2.0)

            # Running couplings を使った eta
            with torch.no_grad():
                eta_rho = get_eta_rho(t_phys_lp)
                eta_sigma = get_eta_sigma(t_phys_lp)

            rhs = (
                -4.0 * u
                + (2.0 + eta_rho * LPAp) * rho_phys * u_rho
                + (2.0 + eta_sigma * LPAp) * sigma_phys * u_sigma
                + loop_term
            )
            res = u_t - rhs

            u_np[it, ir] = u.detach().cpu().item()
            u_t_np[it, ir] = u_t.detach().cpu().item()
            rhs_np[it, ir] = rhs.detach().cpu().item()
            res_np[it, ir] = res.detach().cpu().item()
            loop_np[it, ir] = loop_term.detach().cpu().item()

    abs_res = np.abs(res_np)
    abs_loop = np.abs(loop_np)

    print(f"sigma_fixed = {sigma_fixed}")
    print("max |res|   =", np.max(abs_res))
    print("mean |res|  =", np.mean(abs_res))
    print("median |res|=", np.median(abs_res))
    print("Max u:", np.max(np.abs(u_np)))

    rel_res_loop = abs_res / (abs_loop + 1e-12)

    fig, axes = plt.subplots(1, 3, figsize=(18, 4.5))

    pcm0 = axes[0].pcolormesh(rho_vals, t_vals, abs_loop, shading='auto')
    fig.colorbar(pcm0, ax=axes[0], label='|loop|')
    axes[0].set_title('|loop(t,rho)|')
    axes[0].set_xlabel('rho')
    axes[0].set_ylabel('t')

    pcm1 = axes[1].pcolormesh(rho_vals, t_vals, abs_res, shading='auto')
    fig.colorbar(pcm1, ax=axes[1], label='|res|')
    axes[1].set_title('|residual|')
    axes[1].set_xlabel('rho')
    axes[1].set_ylabel('t')

    pcm2 = axes[2].pcolormesh(rho_vals, t_vals, rel_res_loop, shading='auto', vmax=1e-1)
    fig.colorbar(pcm2, ax=axes[2], label='|res| / |loop|')
    axes[2].set_title('relative residual vs loop')
    axes[2].set_xlabel('rho')
    axes[2].set_ylabel('t')

    for ax in axes:
        # Block boundary at t = -1.0
        ax.axhline(-1.0, color='w', lw=1.0, ls='--', alpha=0.8)

    if invert_t_axis:
        for ax in axes:
            ax.invert_yaxis()

    fig.tight_layout()
    plt.show()

    # Line plots
    irho_list = [0, Nrho_check // 4, Nrho_check // 2, 3 * Nrho_check // 4, int(3.9 * Nrho_check / 4.0)]
    fig, ax = plt.subplots(figsize=(6.5, 4.5))
    for ir in irho_list:
        ax.plot(t_vals, rel_res_loop[:, ir], label=f"rho={rho_vals[ir]:.2f}")
    ax.axhline(1e-1, color='k', lw=1, ls='--')
    ax.axvline(-1.0, color='gray', lw=1, ls=':')
    ax.set_yscale('log')
    ax.set_title(f'relative residual vs t (sigma={sigma_fixed:.2f})')
    ax.set_xlabel('t')
    ax.set_ylabel('|res| / |loop|')
    ax.legend()
    ax.grid(alpha=0.3)
    plt.show()


# ============================================================
# Example usage of residual check
# ============================================================
def plot_residual():

    _dx = 0.350
    _xtot = 0.350*5
    models_b1, models_b2 = load_models_for_sigma_fixed_blockwise(0.0, _xtot, 0.0, _dx)

    residual_check_sigma_fixed_blockwise(
        models_b1, models_b2,
        t0=-2.0, t1=0.0,
        rho0=0.0, rho1=_xtot,
        sigma_fixed=0.0,
        invert_t_axis=False
    )