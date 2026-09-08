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
# UV parameters / running couplings (exactly matching the new training code)
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
# 2D Stitched Grid (5x5 blocks)
# ============================================================

def stitched_grid_matched_2d(
    t_val,
    n_rho_blocks=5,
    n_sigma_blocks=5,
    drho=0.350,
    dsigma=0.350,
    rho_overlap=0.02,
    sigma_overlap=0.02,
    n_points=40 # resolution per block
):
    """
    Load the models of all blocks on the 2D grid, compute the shifts that
    guarantee continuity at the boundaries, and merge them into one large 2D
    array (RHO, SIGMA, U_PRED).
    """
    block_id, t0, t1 = get_time_block_info(t_val)
    shifts = np.zeros((n_rho_blocks, n_sigma_blocks))
    models = [[None]*n_sigma_blocks for _ in range(n_rho_blocks)]

    # 1. load the model of each block
    for irho in range(n_rho_blocks):
        rho_start = irho * drho
        rho_end = (irho + 1) * drho + (rho_overlap if irho < n_rho_blocks - 1 else 0.0)
        for isigma in range(n_sigma_blocks):
            sigma_start = isigma * dsigma
            sigma_end = (isigma + 1) * dsigma + (sigma_overlap if isigma < n_sigma_blocks - 1 else 0.0)
            
            file_model = f"./data_UV_2D/model_higgs_singlet_u_rho{irho}_sig{isigma}_block{block_id}.pt"
            models[irho][isigma] = make_model_for_range(
                file_model, t0, t1, rho_start, rho_end, sigma_start, sigma_end
            )

    # 2. compute the shift (constant term) that keeps continuity at the boundaries
    # anchor condition: u(t, rho=0, sigma=0) = 0
    u_anchor = eval_model_point(models[0][0], t_val, rho_val=0.0, sigma_val=0.0)
    shifts[0, 0] = -u_anchor

    # compute the shift along the rho axis (sigma=0)
    for irho in range(1, n_rho_blocks):
        rho_bound = irho * drho
        left_val = eval_model_point(models[irho-1][0], t_val, rho_bound, 0.0) + shifts[irho-1, 0]
        right_val = eval_model_point(models[irho][0], t_val, rho_bound, 0.0)
        shifts[irho, 0] = left_val - right_val

    # for each rho block, compute the shift going up along the sigma axis
    for irho in range(n_rho_blocks):
        rho_mid = irho * drho + drho / 2.0  # evaluate at the rho of the block center
        for isigma in range(1, n_sigma_blocks):
            sigma_bound = isigma * dsigma
            bot_val = eval_model_point(models[irho][isigma-1], t_val, rho_mid, sigma_bound) + shifts[irho, isigma-1]
            top_val = eval_model_point(models[irho][isigma], t_val, rho_mid, sigma_bound)
            shifts[irho, isigma] = bot_val - top_val

    # 3. build the global grid for plotting (a clean mesh, assuming overlap regions get overwritten)
    rho_global = np.linspace(0, n_rho_blocks * drho, n_rho_blocks * n_points)
    sigma_global = np.linspace(0, n_sigma_blocks * dsigma, n_sigma_blocks * n_points)
    RHO, SIGMA = np.meshgrid(rho_global, sigma_global, indexing='ij')
    
    U_PRED = np.zeros_like(RHO)
    U_TREE = np.zeros_like(RHO)

    # fill in the values over each block's region
    for irho in range(n_rho_blocks):
        for isigma in range(n_sigma_blocks):
            r_mask = (rho_global >= irho * drho) & (rho_global <= (irho + 1) * drho)
            s_mask = (sigma_global >= isigma * dsigma) & (sigma_global <= (isigma + 1) * dsigma)
            
            # prevent dropping the trailing boundary
            if irho == n_rho_blocks - 1: r_mask[-1] = True
            if isigma == n_sigma_blocks - 1: s_mask[-1] = True

            idx_r = np.where(r_mask)[0]
            idx_s = np.where(s_mask)[0]

            if len(idx_r) == 0 or len(idx_s) == 0:
                continue

            r_sub, s_sub = np.meshgrid(rho_global[idx_r], sigma_global[idx_s], indexing='ij')
            
            # evaluate all at once with PyTorch
            r_t = my_to(r_sub.flatten().reshape(-1, 1))
            s_t = my_to(s_sub.flatten().reshape(-1, 1))
            t_t = my_to(np.full((len(r_t), 1), t_val))

            model = models[irho][isigma]
            t_in = model.scale_t(t_t)
            rho_in = model.scale_rho(r_t)
            sigma_in = model.scale_sigma(s_t)

            from my_networks import u_tree_exact # just in case
            with torch.no_grad():
                u_pred_flat = model(t_in, rho_in, sigma_in)[0].detach().cpu().numpy().flatten()
                u_tree_flat = u_tree_exact(t_t, r_t, s_t).detach().cpu().numpy().flatten()

            U_PRED[np.ix_(idx_r, idx_s)] = u_pred_flat.reshape(len(idx_r), len(idx_s)) + shifts[irho, isigma]
            U_TREE[np.ix_(idx_r, idx_s)] = u_tree_flat.reshape(len(idx_r), len(idx_s))

            del model # free memory
            
    gc.collect()
    if torch.cuda.is_available():
        torch.cuda.empty_cache()

    return RHO, SIGMA, U_TREE, U_PRED, shifts


# ============================================================
# 2D Colormap Visualization
# ============================================================

def plot_u_2d_colormap(t_vals=[0.0, -1.0, -2.0]):
    """Plot a 2D color map of U(rho, sigma) at a specific time t."""
    n_plots = len(t_vals)
    fig, axes = plt.subplots(1, n_plots, figsize=(6 * n_plots, 5), sharey=True)
    if n_plots == 1: axes = [axes]

    for i, t_val in enumerate(t_vals):
        RHO, SIGMA, U_TREE, U_PRED, shifts = stitched_grid_matched_2d(t_val)
        
        ax = axes[i]
        # plot the NN prediction
        pcm = ax.pcolormesh(RHO, SIGMA, U_PRED, shading='auto', cmap='viridis')
        fig.colorbar(pcm, ax=ax, label='U_pred')
        
        _dx = 0.350
        _xtot = 0.350*5
        # draw the block boundary lines
        for rb in np.arange(_dx, _xtot, _dx):
            ax.axvline(rb, color='white', lw=1.0, ls=':', alpha=0.7)
        for sb in np.arange(_dx, _xtot, _dx):
            ax.axhline(sb, color='white', lw=1.0, ls=':', alpha=0.7)

        block_id, _, _ = get_time_block_info(t_val)
        ax.set_title(f'U_pred at t={t_val:.2f} (Block {block_id})')
        ax.set_xlabel('rho')
        if i == 0: ax.set_ylabel('sigma')

        ax.set_xlim(0.0, _xtot)
        ax.set_ylim(0.0, _xtot)

    fig.tight_layout()
    plt.show()


# ============================================================
# 1D Slice Plots from 2D Grid
# ============================================================

def plot_u_1d_cuts(t_vals=[0.0, -1.0, -2.0]):
    """Extract a specific slice (sigma=0 fixed, rho=0 fixed, etc.) from the 2D prediction and make a 1D plot."""
    import thermal_np as thermal # for the RGE/finiteT comparison, same as in 1D

    fig, axes = plt.subplots(2, len(t_vals), figsize=(6 * len(t_vals), 8))
    if len(t_vals) == 1: axes = axes[:, None] # match the shape

    for i, t_val in enumerate(t_vals):
        # get the 2D grid data
        RHO, SIGMA, U_TREE, U_PRED, _ = stitched_grid_matched_2d(t_val)
        
        # --- (A) Cut along sigma = 0 ---
        ax_rho = axes[0, i]
        # extract index 0 of sigma_global
        idx_sigma0 = 0 
        rho_1d = RHO[:, idx_sigma0]
        u_pred_1d_rho = U_PRED[:, idx_sigma0]
        u_tree_1d_rho = U_TREE[:, idx_sigma0]
        
        sigma_np_zeros = np.zeros_like(rho_1d)
        t_np_rho = np.full_like(rho_1d, t_val)

        # compute the ring term and perturbation (same as in 1D)
        if finite_T:
            from thermal_functions import get_u_ring
            t_tensor = my_to(t_np_rho)
            rho_tensor = my_to(rho_1d)
            rho_zero_tensor = my_to(np.zeros_like(rho_1d))
            with torch.no_grad():
                u_ring = (get_u_ring(t_tensor, rho_tensor) - get_u_ring(t_tensor, rho_zero_tensor)).detach().cpu().numpy().flatten()
            u_pred_1d_rho += u_ring

        u_pert_rho = thermal.u_pert(t_np_rho, rho_1d, sigma_np_zeros) - thermal.u_pert(t_np_rho, np.zeros_like(rho_1d), sigma_np_zeros)
        u_finiteT_rho = thermal.u_seed_finiteT(t_np_rho, rho_1d, sigma_np_zeros)

        ax_rho.plot(rho_1d, u_tree_1d_rho, 'k-', lw=1.5, label='Tree')
        ax_rho.plot(rho_1d, u_pred_1d_rho, 'r--', lw=1.5, label='NN+ring (Stitched)')
        ax_rho.plot(rho_1d, u_pert_rho, color='tab:blue', ls='-.', lw=1.5, label='RGE (T=0)')
        ax_rho.plot(rho_1d, u_finiteT_rho, color='tab:green', ls=':', lw=1.5, label='Finite T')
        
        _dx = 0.350
        _xtot = 0.350*5
        for rb in np.arange(_dx, _xtot, _dx): ax_rho.axvline(rb, color='gray', lw=0.8, ls=':', alpha=0.5)
        ax_rho.set_title(f't={t_val:.2f} (Cut at sigma=0.0)')
        ax_rho.set_xlabel('rho')
        if i == 0: ax_rho.set_ylabel('u')
        ax_rho.grid(alpha=0.3)
        if i == 0: ax_rho.legend()

        ax_rho.set_xlim(0.0, _xtot)


        # --- (B) Cut along rho = 0 ---
        ax_sig = axes[1, i]
        # extract index 0 of rho_global
        idx_rho0 = 0
        sigma_1d = SIGMA[idx_rho0, :]
        u_pred_1d_sig = U_PRED[idx_rho0, :]
        u_tree_1d_sig = U_TREE[idx_rho0, :]

        rho_np_zeros = np.zeros_like(sigma_1d)
        t_np_sig = np.full_like(sigma_1d, t_val)

        # the ring term is basically zero at rho=0 because it cancels, but include it formally anyway
        if finite_T:
            # at rho=0, U_ring(0) - U_ring(0) = 0, so it is fine to skip
            pass
        
        u_pert_sig = thermal.u_pert(t_np_sig, rho_np_zeros, sigma_1d) - thermal.u_pert(t_np_sig, rho_np_zeros, np.zeros_like(sigma_1d))
        u_finiteT_sig = thermal.u_seed_finiteT(t_np_sig, rho_np_zeros, sigma_1d)

        ax_sig.plot(sigma_1d, u_tree_1d_sig, 'k-', lw=1.5, label='Tree')
        ax_sig.plot(sigma_1d, u_pred_1d_sig, 'r--', lw=1.5, label='NN (Stitched)')
        ax_sig.plot(sigma_1d, u_pert_sig, color='tab:blue', ls='-.', lw=1.5, label='RGE (T=0)')
        ax_sig.plot(sigma_1d, u_finiteT_sig, color='tab:green', ls=':', lw=1.5, label='Finite T')
        
        _dx = 0.350
        _xtot = 0.350*5
        for sb in np.arange(_dx, _xtot, _dx): ax_sig.axvline(sb, color='gray', lw=0.8, ls=':', alpha=0.5)
        ax_sig.set_title(f't={t_val:.2f} (Cut at rho=0.0)')
        ax_sig.set_xlabel('sigma')
        if i == 0: ax_sig.set_ylabel('u')
        ax_sig.grid(alpha=0.3)

        ax_sig.set_xlim(0.0, _xtot)

    fig.tight_layout()
    plt.show()


# ============================================================
# Potential Along the Straight Line Connecting the Two Vacua
# (0, sigma_vs) -- (rho_v, 0), typical of two-step EWPT barrier plots
# ============================================================
def plot_potential_vacuum_line(t_val=-2.0, n_points=40, n_line=200):
    """
    From the 2D stitched grid of the NN(+gauge/top ring) prediction, find the
    vacuum in the rho direction on sigma=0 (rho_v, 0) and the vacuum in the sigma
    direction on rho=0 (0, sigma_vs), and plot the potential along the
    straight-line path connecting the two points.
    """
    import thermal_np as thermal
    from scipy.interpolate import RegularGridInterpolator

    RHO, SIGMA, U_TREE, U_PRED, shifts = stitched_grid_matched_2d(t_val, n_points=n_points)
    rho_axis = RHO[:, 0]
    sigma_axis = SIGMA[0, :]

    # Gauge/top ring correction depends on rho only (sigma=0 baseline cancels)
    if finite_T:
        t_tensor = my_to(np.full_like(rho_axis, t_val))
        rho_tensor = my_to(rho_axis)
        rho_zero_tensor = my_to(np.zeros_like(rho_axis))
        with torch.no_grad():
            u_ring_1d = (
                get_u_ring(t_tensor, rho_tensor) - get_u_ring(t_tensor, rho_zero_tensor)
            ).detach().cpu().numpy().flatten()
        U_FULL = U_PRED + u_ring_1d[:, None]
    else:
        U_FULL = U_PRED

    # Locate the two axis vacua
    irho_min = np.argmin(U_FULL[:, 0])
    isig_min = np.argmin(U_FULL[0, :])
    rho_v = rho_axis[irho_min]
    sigma_vs = sigma_axis[isig_min]
    print(f"[t={t_val:.2f}] rho_v    = {rho_v:.4f}  (h_v = {np.sqrt(2.0*rho_v):.4f})")
    print(f"[t={t_val:.2f}] sigma_vs = {sigma_vs:.4f}  (s_v = {np.sqrt(2.0*sigma_vs):.4f})")

    # Straight line in (rho, sigma): x=0 -> (0, sigma_vs), x=1 -> (rho_v, 0)
    x = np.linspace(0.0, 1.0, n_line)
    rho_line = x * rho_v
    sigma_line = (1.0 - x) * sigma_vs
    pts = np.stack([rho_line, sigma_line], axis=-1)

    interp_pred = RegularGridInterpolator((rho_axis, sigma_axis), U_FULL)
    interp_tree = RegularGridInterpolator((rho_axis, sigma_axis), U_TREE)
    u_line_pred = interp_pred(pts)
    u_line_tree = interp_tree(pts)

    t_np_line = np.full_like(rho_line, t_val)
    u_line_pert = thermal.u_pert(t_np_line, rho_line, sigma_line) - thermal.u_pert(
        t_np_line, np.zeros_like(rho_line), np.zeros_like(sigma_line)
    )
    u_line_finiteT = thermal.u_seed_finiteT(t_np_line, rho_line, sigma_line)

    fig, ax = plt.subplots(figsize=(7, 5))
    ax.plot(x, u_line_tree, 'k-', lw=1.5, label='Tree')
    ax.plot(x, u_line_pred, 'r--', lw=1.5, label='NN+ring (Stitched)')
    ax.plot(x, u_line_pert, color='tab:blue', ls='-.', lw=1.5, label='RGE (T=0)')
    ax.plot(x, u_line_finiteT, color='tab:green', ls=':', lw=1.5, label='Finite T')

    ax.set_xlabel(r'path fraction $x$   [$x{=}0:(\rho,\sigma)=(0,\sigma_{vs})$'
                  r'  $\to$  $x{=}1:(\rho,\sigma)=(\rho_v,0)$]')
    ax.set_ylabel('u')
    ax.set_title(f'Potential along the vacuum-connecting line, t={t_val:.2f}')
    ax.grid(alpha=0.3)
    ax.legend()
    fig.tight_layout()
    savefig(fig, f"potential_vacuum_line_t{t_val:.2f}")
    plt.show()

    return rho_v, sigma_vs


# ============================================================
# Helper: Block Indexing
# ============================================================
def get_grid_block_index(val, dval=0.350, max_blocks=5):
    """Get the corresponding block index from a value (0 to max_blocks-1)."""
    idx = int(np.floor(val / dval))
    return max(0, min(idx, max_blocks - 1))

# ============================================================
# Core: Local Residual Computation
# (the inner loop of the 1D code factored out as a shared function)
# ============================================================
def compute_local_residual(model, t_val, rho_val, sigma_val, LPAp=1.0):
    """
    Function that computes the partial derivatives and the PDE residual at one
    point (t, rho, sigma) using autograd. Does exactly the same processing as the
    original residual computation logic.
    """
    t_grid = my_to([[t_val]])
    rho_grid = my_to([[rho_val]])
    sigma_grid = my_to([[sigma_val]])

    t_in = model.scale_t(t_grid)
    rho_in = model.scale_rho(rho_grid)
    sigma_in = model.scale_sigma(sigma_grid)

    t_in.requires_grad_(True)
    rho_in.requires_grad_(True)
    sigma_in.requires_grad_(True)

    u_tuple = model(t_in, rho_in, sigma_in)
    u = u_tuple[0]

    # --- 1st derivatives ---
    u_t_in = autograd.grad(outputs=u, inputs=t_in, grad_outputs=torch.ones_like(u), create_graph=True, retain_graph=True)[0]
    u_rho_in = autograd.grad(outputs=u, inputs=rho_in, grad_outputs=torch.ones_like(u), create_graph=True, retain_graph=True)[0]
    u_sigma_in = autograd.grad(outputs=u, inputs=sigma_in, grad_outputs=torch.ones_like(u), create_graph=True, retain_graph=True)[0]

    # --- 2nd derivatives ---
    u_rhorho_in = autograd.grad(outputs=u_rho_in, inputs=rho_in, grad_outputs=torch.ones_like(u_rho_in), create_graph=True, retain_graph=True)[0]
    u_sigmasigma_in = autograd.grad(outputs=u_sigma_in, inputs=sigma_in, grad_outputs=torch.ones_like(u_sigma_in), create_graph=True, retain_graph=True)[0]
    u_rhosigma_in = autograd.grad(outputs=u_rho_in, inputs=sigma_in, grad_outputs=torch.ones_like(u_rho_in), create_graph=True, retain_graph=True)[0]

    # --- Unscaling ---
    u_t = model.A_T * u_t_in
    u_rho = model.A_RHO * u_rho_in
    u_sigma = model.A_SIGMA * u_sigma_in
    u_rhorho = model.A_RHO * model.A_RHO * u_rhorho_in
    u_sigmasigma = model.A_SIGMA * model.A_SIGMA * u_sigmasigma_in
    u_rhosigma = model.A_RHO * model.A_SIGMA * u_rhosigma_in

    rho_phys = model.unscale_rho(rho_in)
    sigma_phys = model.unscale_sigma(sigma_in)
    t_phys_lp = model.unscale_t(t_in).detach()

    # --- Physics / Loops ---
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

    with torch.no_grad():
        g1, g2, yt, lamS_t, lamHS_t = get_running_couplings(t_phys_lp)

    mW2 = 0.5 * g2**2 * rho_phys
    mZ2 = 0.5 * (g2**2 + g1**2) * rho_phys
    mt2 = yt**2 * rho_phys

    termW0 = 1.0 / torch.sqrt(torch.clamp(1.0 + mW2, min=eps))
    termZ0 = 1.0 / torch.sqrt(torch.clamp(1.0 + mZ2, min=eps))
    termT0 = 1.0 / torch.sqrt(torch.clamp(1.0 + mt2, min=eps))

    def coth(x, eps=1e-10):
        return torch.cosh(x) / (torch.sinh(x) + eps)


    if finite_T:
        tau_t = tau_uv * torch.exp(-t_phys_lp)
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

    loop_sum = (3.0 * termG + term1 + term2 + 6.0 * termW + 3.0 * termZ - 12.0 * termT)
    rloop = kloop * loop_sum
    loop_term = torch.clamp(rloop, min=-2.0, max=2.0)

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

    return u.item(), u_t.item(), rhs.item(), res.item(), loop_term.item()


# ============================================================
# Plot 1: 2D Plane Residual Map (rho vs sigma at fixed t)
# ============================================================
def plot_residual_2d_plane(t_val, rho_max=1.75, sigma_max=1.75, n_pts=61):
    """Plot the residual map over the whole (rho, sigma) plane at a specific time t."""
    block_id, t0, t1 = get_time_block_info(t_val)
    
    n_rho_blocks, n_sigma_blocks = 5, 5
    drho, dsigma = 0.350, 0.350
    
    # load all models of the relevant block
    models = [[None]*n_sigma_blocks for _ in range(n_rho_blocks)]
    for irho in range(n_rho_blocks):
        for isigma in range(n_sigma_blocks):
            r_end = (irho + 1) * drho + (0.02 if irho < n_rho_blocks - 1 else 0.0)
            s_end = (isigma + 1) * dsigma + (0.02 if isigma < n_sigma_blocks - 1 else 0.0)
            f_model = f"./data_UV_2D/model_higgs_singlet_u_rho{irho}_sig{isigma}_block{block_id}.pt"
            models[irho][isigma] = make_model_for_range(f_model, t0, t1, irho*drho, r_end, isigma*dsigma, s_end)

    rho_vals = np.linspace(0.0, rho_max, n_pts)
    sigma_vals = np.linspace(0.0, sigma_max, n_pts)
    
    res_np = np.zeros((n_pts, n_pts))
    loop_np = np.zeros((n_pts, n_pts))

    # evaluate each grid point
    for ir, r_val in enumerate(rho_vals):
        irho = get_grid_block_index(r_val, drho, n_rho_blocks)
        for isig, s_val in enumerate(sigma_vals):
            isigma = get_grid_block_index(s_val, dsigma, n_sigma_blocks)
            model = models[irho][isigma]
            
            _, _, _, res_val, loop_val = compute_local_residual(model, t_val, r_val, s_val)
            res_np[ir, isig] = res_val
            loop_np[ir, isig] = loop_val

    abs_res = np.abs(res_np)
    abs_loop = np.abs(loop_np)
    rel_res = abs_res / (abs_loop + 1e-12)

    print(f"[2D Plane Map at t={t_val}] max |res| = {np.max(abs_res):.2e}, mean |res| = {np.mean(abs_res):.2e}")

    RHO, SIGMA = np.meshgrid(rho_vals, sigma_vals, indexing='ij')

    fig, axes = plt.subplots(1, 3, figsize=(18, 5))
    
    # |loop|
    pcm0 = axes[0].pcolormesh(RHO, SIGMA, abs_loop, shading='auto')
    fig.colorbar(pcm0, ax=axes[0])
    axes[0].set_title(f'|loop| at t={t_val}')
    
    # |residual|
    pcm1 = axes[1].pcolormesh(RHO, SIGMA, abs_res, shading='auto')
    fig.colorbar(pcm1, ax=axes[1])
    axes[1].set_title(f'|residual| at t={t_val}')
    
    # Relative residual
    pcm2 = axes[2].pcolormesh(RHO, SIGMA, rel_res, shading='auto', vmax=1e-1)
    fig.colorbar(pcm2, ax=axes[2])
    axes[2].set_title(f'|res| / |loop| at t={t_val}')

    for ax in axes:
        ax.set_xlabel('rho')
        ax.set_ylabel('sigma')
        # Draw block boundaries
        _dx = 0.350
        _xtot = 0.350*5
        for b in np.arange(_dx, _xtot, _dx):
            ax.axvline(b, color='white', lw=1.0, ls='--', alpha=0.5)
            ax.axhline(b, color='white', lw=1.0, ls='--', alpha=0.5)

    fig.tight_layout()
    plt.show()


# ============================================================
# Plot 2: Time-Evolution Slice Residual Map (t vs rho at fixed sigma)
# ============================================================
def plot_residual_t_rho_slice(sigma_fixed=0.0, t0=-2.0, t1=0.0, rho_max=1.75, Nt=41, Nrho=61):
    """Residual map of the (t, rho) slice on a specific sigma plane (an extension of the 1D version)."""
    t_vals = np.linspace(t0, t1, Nt)
    rho_vals = np.linspace(0.0, rho_max, Nrho)
    
    n_rho_blocks, n_sigma_blocks = 5, 5
    drho, dsigma = 0.350, 0.350
    
    isigma = get_grid_block_index(sigma_fixed, dsigma, n_sigma_blocks)
    
    res_np = np.zeros((Nt, Nrho))
    loop_np = np.zeros((Nt, Nrho))
    
    # loop over t, rho
    for it, t_val in enumerate(t_vals):
        block_id, tb_start, tb_end = get_time_block_info(t_val)
        for ir, r_val in enumerate(rho_vals):
            irho = get_grid_block_index(r_val, drho, n_rho_blocks)
            
            # load from the file path of the required model
            r_end = (irho + 1) * drho + (0.02 if irho < n_rho_blocks - 1 else 0.0)
            s_end = (isigma + 1) * dsigma + (0.02 if isigma < n_sigma_blocks - 1 else 0.0)
            f_model = f"./data_UV_2D/model_higgs_singlet_u_rho{irho}_sig{isigma}_block{block_id}.pt"
            
            model = make_model_for_range(f_model, tb_start, tb_end, irho*drho, r_end, isigma*dsigma, s_end)
            
            _, _, _, res_val, loop_val = compute_local_residual(model, t_val, r_val, sigma_fixed)
            res_np[it, ir] = res_val
            loop_np[it, ir] = loop_val
            
            # save memory
            del model
            
    abs_res = np.abs(res_np)
    abs_loop = np.abs(loop_np)
    rel_res = abs_res / (abs_loop + 1e-12)

    print(f"[Time Slice Map at sigma={sigma_fixed}] max |res| = {np.max(abs_res):.2e}, mean |res| = {np.mean(abs_res):.2e}")

    RHO, T_GRID = np.meshgrid(rho_vals, t_vals, indexing='xy')

    fig, axes = plt.subplots(1, 3, figsize=(18, 4.5))
    
    pcm0 = axes[0].pcolormesh(RHO, T_GRID, abs_loop, shading='auto')
    fig.colorbar(pcm0, ax=axes[0])
    axes[0].set_title('|loop|')
    
    pcm1 = axes[1].pcolormesh(RHO, T_GRID, abs_res, shading='auto')
    fig.colorbar(pcm1, ax=axes[1])
    axes[1].set_title('|residual|')
    
    pcm2 = axes[2].pcolormesh(RHO, T_GRID, rel_res, shading='auto', vmax=1e-1)
    fig.colorbar(pcm2, ax=axes[2])
    axes[2].set_title(f'|res| / |loop| (sigma={sigma_fixed})')

    for ax in axes:
        ax.set_xlabel('rho')
        ax.set_ylabel('t')
        ax.axhline(-1.0, color='white', lw=1.0, ls='--', alpha=0.8) # time block boundary
        _dx = 0.350
        _xtot = 0.350*5
        for b in np.arange(_dx,_xtot,_dx):
            ax.axvline(b, color='white', lw=1.0, ls=':', alpha=0.5) # rho block boundary

    fig.tight_layout()
    plt.show()

# ============================================================
# example test run
# ============================================================
if __name__ == "__main__":
    # 1. check the smoothness of the boundaries on the 2D spatial plane (rho, sigma)
    # draw the 2D color map
    plot_u_2d_colormap(t_vals=[0.0, -1.0, -2.0])
    
    # 1D comparison plot against the analytic solution
    plot_u_1d_cuts(t_vals=[0.0, -1.0, -2.0])

    #plot_residual_2d_plane(t_val=-0.5, n_pts=61)
    #plot_residual_2d_plane(t_val=-1.5, n_pts=61)
    
    # 2. check how the residual evolves in time (t, rho) (e.g. sigma=0.0 and sigma=0.5)
    plot_residual_t_rho_slice(sigma_fixed=0.0)
    plot_residual_t_rho_slice(sigma_fixed=0.350)