"""
data_UV_ext/ (loss_extensionあり) と data_UV_noext/ (loss_extensionなし) の
学習済みモデルを、stitched_slice_matched (show_UV1D.py) と同じロジックで
5 rhoブロックを継ぎ合わせて評価し、t断面ごとに並べて比較する。
"""

import gc

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import torch

from perturbation.config_params import finite_T
from thermal_functions import get_u_ring, u_CW_zeroT
import thermal_np as thermal
from show_UV1D import (
    make_model_for_range,
    eval_model_slice,
    eval_model_point,
    get_time_block_info,
    my_to,
)


def stitched_slice(data_dir, t_val, n_per_block=200):
    block_id, t0, t1 = get_time_block_info(t_val)
    n_rho_blocks = 5
    drho = 0.350
    rho_overlap = 0.02

    rho_blocks, u_tree_blocks, u_pred_blocks = [], [], []
    model0_for_anchor = None

    for irho in range(n_rho_blocks):
        rho_start = irho * drho
        if irho < n_rho_blocks - 1:
            rho_end = (irho + 1) * drho + rho_overlap
            rho_end0 = (irho + 1) * drho
        else:
            rho_end = (irho + 1) * drho
            rho_end0 = (irho + 1) * drho

        file_model = f"{data_dir}/model_higgs_only_u_rho{irho}_block{block_id}.pt"
        model = make_model_for_range(file_model, t0, t1, rho_start, rho_end)

        if irho == 0:
            model0_for_anchor = model

        rho_np = np.linspace(rho_start, rho_end0, n_per_block)
        u_tree, u_pred = eval_model_slice(model, t_val, rho_np)

        rho_blocks.append(rho_np)
        u_tree_blocks.append(u_tree)
        u_pred_blocks.append(u_pred)

        if irho != 0:
            del model
            gc.collect()
            if torch.cuda.is_available():
                torch.cuda.empty_cache()

    u_anchor = eval_model_point(model0_for_anchor, t_val, rho_val=0.0)
    shifts = [-u_anchor]
    del model0_for_anchor
    gc.collect()
    if torch.cuda.is_available():
        torch.cuda.empty_cache()

    for irho in range(1, n_rho_blocks):
        left_val = u_pred_blocks[irho - 1][-1] + shifts[irho - 1]
        right_val = u_pred_blocks[irho][0]
        shifts.append(left_val - right_val)

    rho_all_list, u_tree_all_list, u_pred_all_list = [], [], []
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

    return (
        np.concatenate(rho_all_list),
        np.concatenate(u_tree_all_list),
        np.concatenate(u_pred_all_list),
    )


def add_ring(t_val, rho_all, u_pred_all):
    if not finite_T:
        return u_pred_all
    t_np = np.full_like(rho_all, t_val)
    t_tensor = my_to(t_np)
    rho_tensor = my_to(rho_all)
    rho_zero_tensor = my_to(np.zeros_like(rho_all))
    with torch.no_grad():
        u_ring = get_u_ring(t_tensor, rho_tensor) - get_u_ring(t_tensor, rho_zero_tensor)
    return u_pred_all + u_ring.detach().cpu().numpy().flatten()


def main():
    t_list = [0.0, -2.0]
    tags = [
        ("data_UV_ext", "with loss_ext (w_sign=1000, w_mag=10)", "tab:red"),
        ("data_UV_noext", "without loss_ext (w_sign=w_mag=0)", "tab:blue"),
    ]

    fig, axes = plt.subplots(1, len(t_list), figsize=(6.5 * len(t_list), 5.0), sharey=False)

    for i, t_val in enumerate(t_list):
        ax = axes[i]
        rho_all = None
        for data_dir, label, color in tags:
            rho_all, _u_tree_all, u_pred_all = stitched_slice(data_dir, t_val)
            u_final = add_ring(t_val, rho_all, u_pred_all)
            ax.plot(rho_all, u_final, lw=1.5, color=color, label=label)

        # Physical reference curves (same construction as show_UV1D.plot_u):
        # RGE-improved tree (running couplings) + finite-T (Arnold-Espinosa
        # resummed thermal) + zero-T Coleman-Weinberg (gated to contribute
        # only at the deep-IR endpoint t=-2, matching plot_u()'s heaviside
        # gating so the Wetterich flow's own 1-loop resummation at
        # intermediate t is not double counted).
        t_np = np.full_like(rho_all, t_val)

        u_pert_vals = thermal.u_pert(t_np, rho_all)
        u_pert_offset = thermal.u_pert(t_np, np.zeros_like(rho_all))
        u_pert_vals = u_pert_vals - u_pert_offset

        u_finiteT_vals = thermal.u_seed_finiteT(t_np, rho_all)

        rho_tor = my_to(rho_all)
        t_tor = my_to(t_np)
        rho0_tor = my_to(rho_all * 0)

        u_cw = u_CW_zeroT(t_tor, rho_tor).detach().cpu().numpy().flatten()
        u_cw0 = u_CW_zeroT(t_tor, rho0_tor).detach().cpu().numpy().flatten()
        cw_gate = np.heaviside(-(t_np + 2.0) + 1e-5, 1.0)

        u_pert_plus_cw = u_pert_vals + (u_cw - u_cw0) * cw_gate
        u_full_ref = u_finiteT_vals + (u_cw - u_cw0) * cw_gate  # tree(running)+CW+thermal

        ax.plot(rho_all, u_pert_plus_cw, color='0.45', ls='-.', lw=1.2,
                label='RGE-run tree + CW (T=0)')
        ax.plot(rho_all, u_full_ref, ls=':', color='tab:green', lw=2.5,
                label='RGE-run tree + CW + thermal + ring (Arnold-Espinosa) -- physical reference')

        _dx = 0.350
        for rb in np.arange(_dx, _dx * 5, _dx):
            ax.axvline(rb, color='gray', lw=0.6, ls=':', alpha=0.4)

        ax.set_title(f"t = {t_val:.2f}")
        ax.set_xlabel("rho")
        ax.set_ylabel("u")
        ax.grid(alpha=0.3)
        if i == 0:
            ax.legend(fontsize=8)

    fig.suptitle("Higgs-only PINN: with vs without loss_extension", y=1.0)
    fig.tight_layout()

    out_path = "plots_pinn/compare_ext_vs_noext.png"
    fig.savefig(out_path, dpi=150)
    print(f"Saved {out_path}")


if __name__ == "__main__":
    main()
