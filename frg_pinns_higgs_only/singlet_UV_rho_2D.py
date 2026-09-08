import gc
import os
import torch
import torch.nn as nn
import torch.optim as optim
import numpy as np
import torch.autograd as autograd

from torch.optim.lr_scheduler import CosineAnnealingLR
from savon import SOAP

import functools
print = functools.partial(print, flush=True)
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

model_compile = False

from perturbation.config_params import finite_T
from running_couplings import tree_params

from my_networks import BaseNet, WrappedPotentialNet
from loss_basics import loss_bc, loss_bc0, loss_interface_overlap, loss_pde
from loss_extensions import loss_sign_and_mag_couplings


_epoch_ratio = 0.0
# ============================================================
# training helpers
# ============================================================
def train_one_block_joint(
    model,
    n_epochs,
    n_res,
    base_lr,
    loop=1.0,
    model_prev=None,
    model_left=None,
    rho_ov_min=None,
    rho_ov_max=None,
    t_min=None,
    t_max=None,
    n_bc=0,
    n_interface=500,
    w_pde=1.0,
    w_bc=1.0,
    w_overlap=1.0,
    w_weak=1.0,
    w_consist=1.0,  # weight of the consistency loss
    w_sign=1000.0,
    save_name_model="model.pt",
    save_name_hist="history.npy",
    save_name_ckpt="checkpoint.pt",
    resume_ckpt=None,
    ckpt_every=200
):

    history = []
    start_epoch = 0

    optimizer = SOAP(model.parameters(), lr=base_lr)
    scheduler = CosineAnnealingLR(
        optimizer,
        T_max=n_epochs,
        eta_min=hp.eta_min
    )
    #eta_min = 3e-8

    if hp.non_resume:
        resume_ckpt = None

    if resume_ckpt is not None and os.path.exists(resume_ckpt):
        ckpt = torch.load(resume_ckpt, map_location=device)
        model.load_state_dict(ckpt["model_state_dict"])
        optimizer.load_state_dict(ckpt["optimizer_state_dict"])
        scheduler.load_state_dict(ckpt["scheduler_state_dict"])
        start_epoch = int(ckpt["epoch"]) + 1
        history = ckpt.get("history", [])

        if "n_epochs" in ckpt and ckpt["n_epochs"] != n_epochs:
            print(f"Warning: checkpoint n_epochs={ckpt['n_epochs']} != current n_epochs={n_epochs}")

        print(f"Resume from epoch {start_epoch}/{n_epochs}: {resume_ckpt}")

    #warmup_epochs = max(1, int(0.03 * n_epochs))

    def compute_losses():

        L_pde, L_consist = loss_pde(model, n_res, loop, _epoch_ratio)

        if model_prev is None:
            L_bc = loss_bc0(model, int(1.0 * n_bc))
        else:
            L_bc = loss_bc(model, model_prev, n_bc)

        if model_left is not None and rho_ov_min is not None and rho_ov_max is not None:
            L_overlap = loss_interface_overlap(
                model_left, model, rho_ov_min, rho_ov_max, t_min, t_max, n_interface
            )
        else:
            L_overlap = torch.tensor(0.0, device=device)

        L_sign, L_mag = loss_sign_and_mag_couplings(model, n_samp=int(1024), rho_cut=None, margin=1e-8)
        #L_sign = torch.tensor(0.0, device=device)
        #L_mag = torch.tensor(0.0, device=device)

        #w_weak = 0
        L_weak = torch.tensor(0.0, device=device)

        loss = (
            w_pde * L_pde
            + w_consist * L_consist
            + w_bc * L_bc
            + w_overlap * L_overlap
            + hp.w_sign * L_sign
            + hp.w_mag * L_mag
            + w_weak * L_weak
        )

        return loss, L_pde, L_consist, L_bc, L_overlap, L_sign, L_mag, L_weak

    for epoch in range(start_epoch, n_epochs):
        global _epoch_ratio
        _epoch_ratio = epoch / n_epochs

        model.train()
        optimizer.zero_grad()

        lr = optimizer.param_groups[0]["lr"]

        loss, L_pde, L_consist, L_bc, L_overlap, L_sign, L_mag, L_weak = compute_losses()
        loss.backward()
        optimizer.step()
        scheduler.step()

        history.append([
            loss.item(),
            L_pde.item(),
            L_consist.item(),
            L_bc.item(),
            L_overlap.item(),
            L_sign.item(),
            L_mag.item(),
            L_weak.item()
        ])

        if (epoch + 1) % 200 == 0 or epoch == start_epoch:
            print(
                f"Epoch {epoch+1}, loss = {loss.item():.1e}, "
                f"loss_pde = {L_pde.item():.1e}, "
                f"loss_consist = {L_consist.item():.1e}, "
                f"loss_bc = {L_bc.item():.1e}, "
                f"loss_ov = {L_overlap.item():.1e}, "
                f"loss_sign = {L_sign.item():.1e}, "
                f"loss_mag = {L_mag.item():.1e}, "
                #f"loss_weak = {L_weak.item():.1e}, "
                f"lr={lr:.1e}"
            )

            #from check_model_predictions import check_model_predictions
            #check_model_predictions(model, device, tree_params)

        if ((epoch + 1) % ckpt_every == 0) or (epoch + 1 == n_epochs):
            ckpt = {
                "epoch": epoch,
                "model_state_dict": model.state_dict(),
                "optimizer_state_dict": optimizer.state_dict(),
                "scheduler_state_dict": scheduler.state_dict(),
                "history": history,
                "base_lr": base_lr,
                "n_epochs": n_epochs,
            }
            ckpt_dir = os.path.dirname(save_name_ckpt) or "."
            hist_dir = os.path.dirname(save_name_hist) or "."
            model_dir = os.path.dirname(save_name_model) or "."

            os.makedirs(ckpt_dir, exist_ok=True)
            os.makedirs(hist_dir, exist_ok=True)
            os.makedirs(model_dir, exist_ok=True)

            torch.save(ckpt, save_name_ckpt)
            np.save(save_name_hist, np.array(history))
            print(f"Saved checkpoint to {save_name_ckpt}")



    mag_lower_org = hp.c_mag_lower
    mag_upper_org = hp.c_mag_upper
    w_sign_org = hp.w_sign

    #hp.w_sign = 1.0


    # freeze the lr at the value it had when the scheduled epochs finished
    final_lr = optimizer.param_groups[0]["lr"]
    for g in optimizer.param_groups:
        g["lr"] = final_lr

    extra_block_size = 100
    max_extra_blocks = 25
    threshold = 1e-9

     # this is safety for the case where the main loop is already finished
    model.train()
    loss, L_pde, L_consist, L_bc, L_overlap, L_sign, L_mag, L_weak = compute_losses()

    loss_sm = L_sign.item() + L_mag.item()
    print(f"After base training: loss_sign+loss_mag = {loss_sm:.3e}, final_lr={final_lr:.1e}")

    if loss_sm > threshold:
        print("Start extra training blocks (fixed lr, no scheduler, no checkpoints).")

        for iblock in range(max_extra_blocks):
            print(f"  Extra block {iblock+1} / {max_extra_blocks}")

            for i in range(extra_block_size):
                # the epoch ratio is for logging only
                _epoch_ratio = (n_epochs + iblock * extra_block_size + i) / (
                    n_epochs + max_extra_blocks * extra_block_size
                )

                model.train()
                optimizer.zero_grad()

                lr = optimizer.param_groups[0]["lr"]  # fixed, but still logged

                loss, L_pde, L_consist, L_bc, L_overlap, L_sign, L_mag, L_weak = compute_losses()
                loss.backward()
                optimizer.step()
                # do not call scheduler.step()

                history.append([
                    loss.item(),
                    L_pde.item(),
                    L_consist.item(),
                    L_bc.item(),
                    L_overlap.item(),
                    L_sign.item(),
                    L_mag.item(),
                    L_weak.item()
                ])

            # evaluate after running the block for the full 100 steps
            if hp.w_sign == 0:
                loss_sm = L_mag.item()
            else:
                loss_sm = L_sign.item() + L_mag.item()

            print(
                f"  After block {iblock+1}, "
                f"loss_sign = {L_sign.item():.3e}, "
                f"loss_mag = {L_mag.item():.3e}, lr={lr:.1e}"
            )

            #from check_model_predictions import check_model_predictions
            #check_model_predictions(model, device, tree_params)

            if loss_sm < threshold:
                print("  Threshold reached after this block, stop extra training.")
                break

            hp.c_mag_lower = hp.c_mag_lower * 0.8
            hp.c_mag_upper = hp.c_mag_upper * 1.1

            #if hp.c_mag_upper > 5.0:
            #    hp.c_mag_upper = 5.0



        print("Extra training blocks finished.")
    else:
        print("loss_sign+loss_mag already below threshold, skip extra training.")

    hp.c_mag_lower = mag_lower_org
    hp.c_mag_upper = mag_upper_org
    hp.w_sign = w_sign_org

    os.makedirs(os.path.dirname(save_name_model) or ".", exist_ok=True)
    torch.save(model.state_dict(), save_name_model)
    print(f"Saved model to {save_name_model}")

    hist_arr = np.array(history)
    np.save(save_name_hist, hist_arr)
    print(f"Saved history to {save_name_hist}")

    del optimizer
    del scheduler
    gc.collect()
    if torch.cuda.is_available():
        torch.cuda.empty_cache()

    return hist_arr

def load_trained_model(file_model, t0, t1, rho0, rho1):
    base = BaseNet().to(device)
    model = WrappedPotentialNet(
        base,
        T_min=t0, T_max=t1,
        Rho_min=rho0, Rho_max=rho1,
    ).to(device)
    model.load_state_dict(torch.load(file_model, map_location=device))
    model.eval()
    for p in model.parameters():
        p.requires_grad_(False)

    if model_compile:
        model = torch.compile(model, mode="default")

    return model


_n_epochs_base=None

def train_rho_window_joint(
    irho,
    rho_min,
    rho_max,
    rho_ov_min=None,
    rho_ov_max=None,
    model_left_blocks=None,
    w_overlap=1.0,
    tag="default"
):

    w_bc1 = hp.w_bc1
    w_bc2 = hp.w_bc2
    w_consist = hp.w_consist
    w_pde = hp.w_pde
    w_sign = hp.w_sign
    w_weak = hp.w_weak
    w_overlap = hp.w_overlap
    epoch_1to2 = hp.epoch_1to2


    if model_left_blocks is None:
        model_left_blocks = [None, None]

    has_left_interface = (rho_ov_min is not None) and all(m is not None for m in model_left_blocks)
    w_ov_actual = w_overlap if has_left_interface else 0.0

    t_blocks = [
        (-1.0, 0.0),
        (-2.0, -1.0),
    ]

    n_epochs_blocks = [
        int(_n_epochs_base*epoch_1to2),
        int(_n_epochs_base),
    ]

    w_bc_blocks = [
        w_bc1,
        w_bc2,
    ]

    trained_models = []
    all_hist_local = []

    model_prev = None
    prev_model_file = None

    for iblock, ((t0, t1), n_epochs, w_bc_now) in enumerate(
        zip(t_blocks, n_epochs_blocks, w_bc_blocks), start=1
    ):

        if (t0,t1)==(-2.0, -1.0) and hp.skip_2nd == True:
            continue

        base = BaseNet().to(device)
        model = WrappedPotentialNet(
            base,
            T_min=t0, T_max=t1,
            Rho_min=rho_min, Rho_max=rho_max,
        ).to(device)

        if prev_model_file is not None:
            model.load_state_dict(torch.load(prev_model_file, map_location=device))

        if model_compile:
            model = torch.compile(model, mode="default")

        file_model = f"./data_UV_{tag}/model_higgs_only_u_rho{irho}_block{iblock}.pt"
        file_hist = f"./data_UV_{tag}/history_higgs_only_u_rho{irho}_block{iblock}.npy"
        file_ckpt = f"./data_UV_{tag}/checkpoint_higgs_only_u_rho{irho}_block{iblock}.pt"

        hist = train_one_block_joint(
            model=model,
            n_epochs=n_epochs,
            n_res=int(7168),
            base_lr=3e-4,
            loop=1.0,
            model_prev=model_prev,
            model_left=model_left_blocks[iblock - 1] if has_left_interface else None,
            rho_ov_min=rho_ov_min,
            rho_ov_max=rho_ov_max,
            t_min=t0,
            t_max=t1,
            n_bc=int(4096),
            n_interface=int(2048),
            w_pde=w_pde,
            w_bc=w_bc_now,
            w_overlap=w_ov_actual,
            w_weak=w_weak,
            w_consist=w_consist,
            w_sign=w_sign,
            save_name_model=file_model,
            save_name_hist=file_hist,
            save_name_ckpt=file_ckpt,
            resume_ckpt=file_ckpt if os.path.exists(file_ckpt) else None,
            ckpt_every=200
        )

        all_hist_local.append(hist)

        loaded_model = load_trained_model(
            file_model=file_model,
            t0=t0, t1=t1,
            rho0=rho_min, rho1=rho_max,
        )
        trained_models.append(loaded_model)

        model_prev = loaded_model
        prev_model_file = file_model

        gc.collect()
        if torch.cuda.is_available():
            torch.cuda.empty_cache()

    return all_hist_local, trained_models


# ============================================================
# main loop (supports split execution)
# ============================================================
def train(n_epochs=1000, rho_blocks=None, tag="default"):
    all_hist = {}

    global _n_epochs_base
    _n_epochs_base = int(n_epochs)

    n_rho_blocks = 5
    d_rho = 0.350
    rho_overlap = 0.02

    # run all blocks [0, 1, 2, 3, 4] if none specified
    if rho_blocks is None:
        rho_blocks = list(range(n_rho_blocks))

    prev_block_models = [None, None]

    # --- when starting partway through, load the previous (irho - 1) model ---
    first_rho = rho_blocks[0]
    if first_rho > 0:
        prev_rho = first_rho - 1
        prev_rho_start = prev_rho * d_rho
        if prev_rho < n_rho_blocks - 1:
            prev_rho_end = (prev_rho + 1) * d_rho + rho_overlap
        else:
            prev_rho_end = (prev_rho + 1) * d_rho


        print(f"--- resuming partway: loading the trained model of rho block {prev_rho} ---")
        t_blocks = [(-1.0, 0.0), (-2.0, -1.0)]

        for iblock, (t0, t1) in enumerate(t_blocks, start=1):
            file_model = f"./data_UV_{tag}/model_higgs_only_u_rho{prev_rho}_block{iblock}.pt"

            if os.path.exists(file_model):
                prev_block_models[iblock - 1] = load_trained_model(
                    file_model=file_model,
                    t0=t0, t1=t1,
                    rho0=prev_rho_start, rho1=prev_rho_end,
                )
                print("--- load complete ---")
            else:
                print(f"previous block model not found. irho={prev_rho} has not been trained yet: {file_model}")

    # -------------------------------------------------------------------------

    for irho in rho_blocks:
        # safeguard for when a block index outside the specified range is passed
        if irho >= n_rho_blocks:
            print(f"Warning: irho={irho} exceeds the maximum number of blocks ({n_rho_blocks}). Skipping.")
            continue

        rho_start = irho * d_rho

        if irho < n_rho_blocks - 1:
            rho_end = (irho + 1) * d_rho + rho_overlap
        else:
            rho_end = (irho + 1) * d_rho

        if irho > 0:
            rho_ov_min = rho_start
            rho_ov_max = rho_start + rho_overlap
        else:
            rho_ov_min = None
            rho_ov_max = None

        print(f"\n===== rho block {irho}: [{rho_start:.3f}, {rho_end:.3f}] =====")
        if rho_ov_min is not None:
            print(f"      Overlap with previous: [{rho_ov_min:.3f}, {rho_ov_max:.3f}]")

        hist_list, trained_models = train_rho_window_joint(
            irho=irho,
            rho_min=rho_start,
            rho_max=rho_end,
            rho_ov_min=rho_ov_min,
            rho_ov_max=rho_ov_max,
            model_left_blocks=prev_block_models,
            w_overlap=1e-2,
            tag=tag
        )

        all_hist[irho] = hist_list

        # update the model for the next loop (the right-neighbor block)
        prev_block_models = trained_models

    return all_hist
