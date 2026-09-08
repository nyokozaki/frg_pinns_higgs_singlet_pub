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
    sigma_min=0.0,
    sigma_max=0.5,
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

    '''
    import weekpinns as wpn
    vpinn_deg = 10
    N_t, N_r, N_s = 2, 2, 2 

    vpinn_configs = []
    for nt in range(N_t + 1):
        for nr in range(N_r + 1):
            for ns in range(N_s + 1):
                # whether to include (0,0,0) is a matter of taste
                if (nt, nr, ns) != (0, 0, 0):
                    vpinn_configs.append((nt, nr, ns))

    t_in_w, rho_in_w, sig_in_w, W_3d_w, Fa_list_w, dFa_dt_w, dFa_drho_w, dFa_dsigma_w = wpn.setup_weak_pinn(
        deg=vpinn_deg,
        t_min=model.T_min, t_max=model.T_max,
        rho_min=model.Rho_min, rho_max=model.Rho_max,
        sigma_min=model.Sigma_min, sigma_max=model.Sigma_max,
        configs=vpinn_configs,
        device=device
    )
    '''

    def compute_losses():

        L_pde, L_consist = loss_pde(model, n_res, loop, _epoch_ratio)
        
        if model_prev is None:
            L_bc = loss_bc0(model, int(1.0 * n_bc))
        else:
            L_bc = loss_bc(model, model_prev, n_bc)

        if model_left is not None and rho_ov_min is not None and rho_ov_max is not None:
            L_overlap = loss_interface_overlap(
                model_left, model, rho_ov_min, rho_ov_max, t_min, t_max, sigma_min, sigma_max, n_interface
            )
        else:
            L_overlap = torch.tensor(0.0, device=device)

        L_sign, L_mag = loss_sign_and_mag_couplings(model, n_samp=int(1024), rho_cut=None, sigma_cut=None, margin=1e-8)
        #L_sign = torch.tensor(0.0, device=device)
        #L_mag = torch.tensor(0.0, device=device)
        
        #w_weak = 0
        L_weak = torch.tensor(0.0, device=device)
        '''
        if w_weak > 0.0:
            L_weak = wpn.loss_weak(model, t_in_w, rho_in_w, sig_in_w, W_3d_w, Fa_list_w, dFa_dt_w, dFa_drho_w, dFa_dsigma_w, loop, epoch_ratio)
        else:
            L_weak = torch.tensor(0.0, device=device)
        '''
        
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

        #if epoch < warmup_epochs:
        #    scheduler.step()
        #    lr = base_lr * float(epoch + 1) / float(warmup_epochs)
        #    for g in optimizer.param_groups:
        #        g["lr"] = lr
        #else:
        #    scheduler.step()
        #    lr = optimizer.param_groups[0]["lr"]

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
                
    


    mag_lower_org_rho = hp.c_mag_lower_rho
    mag_upper_org_rho = hp.c_mag_upper_rho
    mag_lower_org_sigma = hp.c_mag_lower_sigma
    mag_upper_org_sigma = hp.c_mag_upper_sigma
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

            hp.c_mag_lower_rho = hp.c_mag_lower_rho * 0.8
            hp.c_mag_upper_rho = hp.c_mag_upper_rho * 1.1
            hp.c_mag_lower_sigma = hp.c_mag_lower_sigma * 0.8
            hp.c_mag_upper_sigma = hp.c_mag_upper_sigma * 1.1
            
            #if hp.c_mag_upper > 5.0:
            #    hp.c_mag_upper = 5.0



        print("Extra training blocks finished.")
    else:
        print("loss_sign+loss_mag already below threshold, skip extra training.")

    hp.c_mag_lower_rho = mag_lower_org_rho
    hp.c_mag_upper_rho = mag_upper_org_rho
    hp.c_mag_lower_sigma = mag_lower_org_sigma
    hp.c_mag_upper_sigma = mag_upper_org_sigma
    hp.w_sign = w_sign_org

    os.makedirs("./data_UV", exist_ok=True)
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

def load_trained_model(file_model, t0, t1, rho0, rho1, sigma0, sigma1):
    base = BaseNet().to(device)
    model = WrappedPotentialNet(
        base,
        T_min=t0, T_max=t1,
        Rho_min=rho0, Rho_max=rho1,
        Sigma_min=sigma0, Sigma_max=sigma1
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
    sigma0=0.0,
    sigma1=0.350,
    model_left_blocks=None,
    w_overlap=1.0
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
            Sigma_min=sigma0, Sigma_max=sigma1
        ).to(device)

        if prev_model_file is not None:
            model.load_state_dict(torch.load(prev_model_file, map_location=device))

        if model_compile:
            model = torch.compile(model, mode="default")

        file_model = f"./data_UV/model_higgs_singlet_u_rho{irho}_block{iblock}.pt"
        file_hist = f"./data_UV/history_higgs_singlet_u_rho{irho}_block{iblock}.npy"
        file_ckpt = f"./data_UV/checkpoint_higgs_singlet_u_rho{irho}_block{iblock}.pt"

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
            sigma_min=sigma0,
            sigma_max=sigma1,
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
            sigma0=sigma0, sigma1=sigma1
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
def train(n_epochs=1000, rho_blocks=None):
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
            file_model = f"./data_UV/model_higgs_singlet_u_rho{prev_rho}_block{iblock}.pt"

            if os.path.exists(file_model):
                prev_block_models[iblock - 1] = load_trained_model(
                    file_model=file_model,
                    t0=t0, t1=t1,
                    rho0=prev_rho_start, rho1=prev_rho_end,
                    sigma0=0.0, sigma1=0.350
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
            sigma0=0.0,
            sigma1=0.350,
            model_left_blocks=prev_block_models,
            w_overlap=1e-2
        )

        all_hist[irho] = hist_list

        # update the model for the next loop (the right-neighbor block)
        prev_block_models = trained_models
        
    return all_hist











    
# ============================================================
# 2D Grid Extension (rho & sigma)
# ============================================================

def train_one_block_joint_2d(
    model,
    n_epochs,
    n_res,
    base_lr,
    loop=1.0,
    model_prev=None,
    model_left=None,       # neighboring model in the rho direction
    rho_ov_min=None,
    rho_ov_max=None,
    model_bottom=None,     # neighboring model in the sigma direction
    sigma_ov_min=None,
    sigma_ov_max=None,
    model_right=None,      # right-neighbor model in the rho direction (used from round 2 on)
    rho_ov_min2=None,
    rho_ov_max2=None,
    model_top=None,        # top-neighbor model in the sigma direction (used from round 2 on)
    sigma_ov_min2=None,
    sigma_ov_max2=None,
    rho_min=0.0,
    rho_max=1.0,
    t_min=None,
    t_max=None,
    sigma_min=0.0,
    sigma_max=0.5,
    n_bc=0,
    n_interface=500,
    w_pde=1.0,
    w_bc=1.0,
    w_overlap=1.0,
    w_weak=1.0,       
    w_consist=1.0,  
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
    '''
    import weekpinns as wpn
    vpinn_deg = 10
    N_t, N_r, N_s = 2, 2, 2 

    vpinn_configs = []
    for nt in range(N_t + 1):
        for nr in range(N_r + 1):
            for ns in range(N_s + 1):
                if (nt, nr, ns) != (0, 0, 0):
                    vpinn_configs.append((nt, nr, ns))

    t_in_w, rho_in_w, sig_in_w, W_3d_w, Fa_list_w, dFa_dt_w, dFa_drho_w, dFa_dsigma_w = wpn.setup_weak_pinn(
        deg=vpinn_deg,
        t_min=model.T_min, t_max=model.T_max,
        rho_min=model.Rho_min, rho_max=model.Rho_max,
        sigma_min=model.Sigma_min, sigma_max=model.Sigma_max,
        configs=vpinn_configs,
        device=device
    )
    '''

    def compute_losses():
        L_pde, L_consist = loss_pde(model, n_res, loop, _epoch_ratio)
        
        if model_prev is None:
            L_bc = loss_bc0(model, int(1.0 * n_bc))
        else:
            L_bc = loss_bc(model, model_prev, n_bc)

        # ====== 2D Overlap Loss (matching the first derivatives) ======
        L_overlap = torch.tensor(0.0, device=device)
        
        # Left (rho) Overlap
        if model_left is not None and rho_ov_min is not None and rho_ov_max is not None:
            L_overlap += loss_interface_overlap(
                model_left, model, rho_ov_min, rho_ov_max, t_min, t_max, sigma_min, sigma_max, n_interface
            )
            
        # Bottom (sigma) Overlap
        if model_bottom is not None and sigma_ov_min is not None and sigma_ov_max is not None:
            L_overlap += loss_interface_overlap(
                model_bottom, model, rho_min, rho_max, t_min, t_max, sigma_ov_min, sigma_ov_max, n_interface
            )

        # Right (rho) Overlap (round 2 on)
        if model_right is not None and rho_ov_min2 is not None and rho_ov_max2 is not None:
            L_overlap += loss_interface_overlap(
                model, model_right, rho_ov_min2, rho_ov_max2, t_min, t_max, sigma_min, sigma_max, n_interface
            )

        # Top (sigma) Overlap (round 2 on)
        if model_top is not None and sigma_ov_min2 is not None and sigma_ov_max2 is not None:
            L_overlap += loss_interface_overlap(
                model, model_top, rho_min, rho_max, t_min, t_max, sigma_ov_min2, sigma_ov_max2, n_interface
            )
        # ===================================================

        L_sign, L_mag = loss_sign_and_mag_couplings(model, n_samp=int(1024), rho_cut=None, sigma_cut=None, margin=1e-8)
        
        L_weak = torch.tensor(0.0, device=device)
        '''
        if w_weak > 0.0:
            L_weak = wpn.loss_weak(model, t_in_w, rho_in_w, sig_in_w, W_3d_w, Fa_list_w, dFa_dt_w, dFa_drho_w, dFa_dsigma_w, loop, epoch_ratio)
        else:
            L_weak = torch.tensor(0.0, device=device)
        '''
      
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
        global _epoch_ratio # keep the existing declaration
        _epoch_ratio = epoch / n_epochs

        model.train()
        optimizer.zero_grad()

        lr = optimizer.param_groups[0]["lr"]
        loss, L_pde, L_consist, L_bc, L_overlap, L_sign, L_mag, L_weak = compute_losses()
        loss.backward()
        optimizer.step()
        scheduler.step()

        history.append([
            loss.item(), L_pde.item(), L_consist.item(),
            L_bc.item(), L_overlap.item(), L_sign.item(),
            L_mag.item(), L_weak.item() 
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


    mag_lower_org_rho = hp.c_mag_lower_rho
    mag_upper_org_rho = hp.c_mag_upper_rho
    mag_lower_org_sigma = hp.c_mag_lower_sigma
    mag_upper_org_sigma = hp.c_mag_upper_sigma
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

    do_extra = getattr(hp, "do_extra", True)

    if loss_sm > threshold and do_extra == True:
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
                f"loss_sign = {L_sign.item():.3e},"
                f"loss_mag = {L_mag.item():.3e}, lr={lr:.1e}"
           )

            #from check_model_predictions import check_model_predictions
            #check_model_predictions(model, device, tree_params)

            if loss_sm < threshold:
                print("  Threshold reached after this block, stop extra training.")
                break

            hp.c_mag_lower_rho = hp.c_mag_lower_rho * 0.8
            hp.c_mag_upper_rho = hp.c_mag_upper_rho * 1.1
            hp.c_mag_lower_sigma = hp.c_mag_lower_sigma * 0.8
            hp.c_mag_upper_sigma = hp.c_mag_upper_sigma * 1.1

            #if hp.c_mag_upper > 5.0:
            #    hp.c_mag_upper = 5.0



        print("Extra training blocks finished.")
    else:
        print("loss_sign+loss_mag already below threshold, skip extra training.")

    
    hp.c_mag_lower_rho = mag_lower_org_rho
    hp.c_mag_upper_rho = mag_upper_org_rho
    hp.c_mag_lower_sigma = mag_lower_org_sigma
    hp.c_mag_upper_sigma = mag_upper_org_sigma
    hp.w_sign = w_sign_org



    # create the directory for 2D
    os.makedirs("./data_UV_2D", exist_ok=True)
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
######################################
#
# 2D grid
#
######################################

def train_rho_sigma_window_joint(
    irho,
    isigma,
    rho_min,
    rho_max,
    sigma_min,
    sigma_max,
    rho_ov_min=None,
    rho_ov_max=None,
    sigma_ov_min=None,
    sigma_ov_max=None,
    model_left_blocks=None,
    model_bottom_blocks=None,
    w_overlap=1.0
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
    if model_bottom_blocks is None:
        model_bottom_blocks = [None, None]

    has_left_interface = (rho_ov_min is not None) and all(m is not None for m in model_left_blocks)
    has_bottom_interface = (sigma_ov_min is not None) and all(m is not None for m in model_bottom_blocks)
    
    # enable the overlap loss if either interface is present
    w_ov_actual = w_overlap if (has_left_interface or has_bottom_interface) else 0.0

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
        base = BaseNet().to(device)
        model = WrappedPotentialNet(
            base,
            T_min=t0, T_max=t1,
            Rho_min=rho_min, Rho_max=rho_max,
            Sigma_min=sigma_min, Sigma_max=sigma_max
        ).to(device)

        if prev_model_file is not None:
            model.load_state_dict(torch.load(prev_model_file, map_location=device))

        if model_compile:
            model = torch.compile(model, mode="default")

        file_model = f"./data_UV_2D/model_higgs_singlet_u_rho{irho}_sig{isigma}_block{iblock}.pt"
        file_hist = f"./data_UV_2D/history_higgs_singlet_u_rho{irho}_sig{isigma}_block{iblock}.npy"
        file_ckpt = f"./data_UV_2D/checkpoint_higgs_singlet_u_rho{irho}_sig{isigma}_block{iblock}.pt"

        hist = train_one_block_joint_2d(
            model=model,
            n_epochs=n_epochs,
            n_res=int(7168),
            base_lr=3e-4,
            loop=1.0,
            model_prev=model_prev,
            model_left=model_left_blocks[iblock - 1] if has_left_interface else None,
            rho_ov_min=rho_ov_min,
            rho_ov_max=rho_ov_max,
            model_bottom=model_bottom_blocks[iblock - 1] if has_bottom_interface else None,
            sigma_ov_min=sigma_ov_min,
            sigma_ov_max=sigma_ov_max,
            rho_min=rho_min,     # needed when computing bottom
            rho_max=rho_max,
            t_min=t0,
            t_max=t1,
            sigma_min=sigma_min,
            sigma_max=sigma_max,
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
            sigma0=sigma_min, sigma1=sigma_max
        )
        trained_models.append(loaded_model)
        model_prev = loaded_model
        prev_model_file = file_model

        gc.collect()
        if torch.cuda.is_available():
            torch.cuda.empty_cache()

    return all_hist_local, trained_models

# ============================================================
# Main 2D Loop (supports partway restart / resume)
# ============================================================
def train_rho_sigma(n_epochs=1000, rho_blocks=None, sigma_blocks=None):
    all_hist = {}

    global _n_epochs_base
    _n_epochs_base = int(n_epochs)

    # Rho grids
    n_rho_blocks = 5
    d_rho = 0.350
    rho_overlap = 0.02
    
    # Sigma grids
    n_sigma_blocks = 5
    d_sigma = 0.350
    sigma_overlap = 0.02

    # run all blocks if none specified in the arguments
    if rho_blocks is None:
        rho_blocks = list(range(n_rho_blocks))
    if sigma_blocks is None:
        sigma_blocks = list(range(n_sigma_blocks))

    # list holding the trained 2D models [irho][isigma]
    trained_grid_models = [[None for _ in range(n_sigma_blocks)] for _ in range(n_rho_blocks)]

    # --- helper: load the trained model of the given (irho, isigma) if it exists ---
    def try_load_models(r_idx, s_idx):
        r_start = r_idx * d_rho
        r_end = (r_idx + 1) * d_rho + (rho_overlap if r_idx < n_rho_blocks - 1 else 0.0)
        s_start = s_idx * d_sigma
        s_end = (s_idx + 1) * d_sigma + (sigma_overlap if s_idx < n_sigma_blocks - 1 else 0.0)

        t_blocks = [(-1.0, 0.0), (-2.0, -1.0)]
        loaded = [None, None]

        for iblock, (t0, t1) in enumerate(t_blocks, start=1):
            # skip the second block if hp.skip_2nd is True
            if (t0, t1) == (-2.0, -1.0) and getattr(hp, 'skip_2nd', False):
                continue
            
            file_model = f"./data_UV_2D/model_higgs_singlet_u_rho{r_idx}_sig{s_idx}_block{iblock}.pt"
            if os.path.exists(file_model):
                loaded[iblock - 1] = load_trained_model(
                    file_model=file_model,
                    t0=t0, t1=t1,
                    rho0=r_start, rho1=r_end,
                    sigma0=s_start, sigma1=s_end
                )
            else:
                # if even one required block is missing, treat the load as incomplete
                return [None, None]
        return loaded
    # --------------------------------------------------------------------------

    for irho in rho_blocks:
        for isigma in sigma_blocks:
            # skip out-of-range specifications
            if irho >= n_rho_blocks or isigma >= n_sigma_blocks:
                continue

            rho_start = irho * d_rho
            rho_end = (irho + 1) * d_rho + (rho_overlap if irho < n_rho_blocks - 1 else 0.0)
            rho_ov_min = rho_start if irho > 0 else None
            rho_ov_max = rho_start + rho_overlap if irho > 0 else None

            sigma_start = isigma * d_sigma
            sigma_end = (isigma + 1) * d_sigma + (sigma_overlap if isigma < n_sigma_blocks - 1 else 0.0)
            sigma_ov_min = sigma_start if isigma > 0 else None
            sigma_ov_max = sigma_start + sigma_overlap if isigma > 0 else None

            # 1. skip if this block itself is already trained
            current_models = try_load_models(irho, isigma)
            if current_models[0] is not None:
                print(f"\n--- rho block {irho} | sigma block {isigma} already trained; loading and skipping ---")
                trained_grid_models[irho][isigma] = current_models
                continue

            print(f"\n===== rho block {irho}: [{rho_start:.3f}, {rho_end:.3f}] | sigma block {isigma}: [{sigma_start:.3f}, {sigma_end:.3f}] =====")

            # 2. obtain the left (irho-1) and bottom (isigma-1) boundary models, loading as needed
            left_models = [None, None]
            if irho > 0:
                left_models = trained_grid_models[irho - 1][isigma]
                if left_models is None or left_models[0] is None:
                    print(f"  --> loading the trained model of the left boundary (rho={irho-1}, sig={isigma}) from disk")
                    left_models = try_load_models(irho - 1, isigma)
                    trained_grid_models[irho - 1][isigma] = left_models
                    if left_models[0] is None:
                        print(f"  Warning: left-boundary model not found (rho={irho-1}, sig={isigma})")

            bottom_models = [None, None]
            if isigma > 0:
                bottom_models = trained_grid_models[irho][isigma - 1]
                if bottom_models is None or bottom_models[0] is None:
                    print(f"  --> loading the trained model of the bottom boundary (rho={irho}, sig={isigma-1}) from disk")
                    bottom_models = try_load_models(irho, isigma - 1)
                    trained_grid_models[irho][isigma - 1] = bottom_models
                    if bottom_models[0] is None:
                        print(f"  Warning: bottom-boundary model not found (rho={irho}, sig={isigma-1})")

            if rho_ov_min is not None:
                print(f"      Overlap Left (rho) : [{rho_ov_min:.3f}, {rho_ov_max:.3f}]")
            if sigma_ov_min is not None:
                print(f"      Overlap Bottom (sigma) : [{sigma_ov_min:.3f}, {sigma_ov_max:.3f}]")

            hist_list, trained_models = train_rho_sigma_window_joint(
                irho=irho,
                isigma=isigma,
                rho_min=rho_start,
                rho_max=rho_end,
                sigma_min=sigma_start,
                sigma_max=sigma_end,
                rho_ov_min=rho_ov_min,
                rho_ov_max=rho_ov_max,
                sigma_ov_min=sigma_ov_min,
                sigma_ov_max=sigma_ov_max,
                model_left_blocks=left_models,
                model_bottom_blocks=bottom_models,
                w_overlap=1e-2
            )

            all_hist[f"rho{irho}_sig{isigma}"] = hist_list
            trained_grid_models[irho][isigma] = trained_models

    return all_hist

# ============================================================
# Main 2D Loop (round 2: a smoothing pass that also includes interface continuity on the right/top boundaries)
# ============================================================

def train_rho_sigma_window_joint_round2(
    irho,
    isigma,
    rho_min,
    rho_max,
    sigma_min,
    sigma_max,
    src_dir,
    out_dir,
    rho_ov_min=None,
    rho_ov_max=None,
    sigma_ov_min=None,
    sigma_ov_max=None,
    rho_ov_min2=None,
    rho_ov_max2=None,
    sigma_ov_min2=None,
    sigma_ov_max2=None,
    model_left_blocks=None,
    model_bottom_blocks=None,
    model_right_blocks=None,
    model_top_blocks=None,
    w_overlap=1.0
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
    if model_bottom_blocks is None:
        model_bottom_blocks = [None, None]
    if model_right_blocks is None:
        model_right_blocks = [None, None]
    if model_top_blocks is None:
        model_top_blocks = [None, None]

    has_left_interface = (rho_ov_min is not None) and all(m is not None for m in model_left_blocks)
    has_bottom_interface = (sigma_ov_min is not None) and all(m is not None for m in model_bottom_blocks)
    has_right_interface = (rho_ov_min2 is not None) and all(m is not None for m in model_right_blocks)
    has_top_interface = (sigma_ov_min2 is not None) and all(m is not None for m in model_top_blocks)

    # enable the overlap loss if any interface is present
    w_ov_actual = w_overlap if (
        has_left_interface or has_bottom_interface or has_right_interface or has_top_interface
    ) else 0.0

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
        base = BaseNet().to(device)
        model = WrappedPotentialNet(
            base,
            T_min=t0, T_max=t1,
            Rho_min=rho_min, Rho_max=rho_max,
            Sigma_min=sigma_min, Sigma_max=sigma_max
        ).to(device)

        if prev_model_file is not None:
            model.load_state_dict(torch.load(prev_model_file, map_location=device))
        else:
            # warm-start from the trained weights of the same cell and same block from round 1 (src_dir)
            warm_file = f"{src_dir}/model_higgs_singlet_u_rho{irho}_sig{isigma}_block{iblock}.pt"
            if os.path.exists(warm_file):
                model.load_state_dict(torch.load(warm_file, map_location=device))

        if model_compile:
            model = torch.compile(model, mode="default")

        file_model = f"{out_dir}/model_higgs_singlet_u_rho{irho}_sig{isigma}_block{iblock}.pt"
        file_hist = f"{out_dir}/history_higgs_singlet_u_rho{irho}_sig{isigma}_block{iblock}.npy"
        file_ckpt = f"{out_dir}/checkpoint_higgs_singlet_u_rho{irho}_sig{isigma}_block{iblock}.pt"

        hist = train_one_block_joint_2d(
            model=model,
            n_epochs=n_epochs,
            n_res=int(7168),
            base_lr=3e-4,
            loop=1.0,
            model_prev=model_prev,
            model_left=model_left_blocks[iblock - 1] if has_left_interface else None,
            rho_ov_min=rho_ov_min,
            rho_ov_max=rho_ov_max,
            model_bottom=model_bottom_blocks[iblock - 1] if has_bottom_interface else None,
            sigma_ov_min=sigma_ov_min,
            sigma_ov_max=sigma_ov_max,
            model_right=model_right_blocks[iblock - 1] if has_right_interface else None,
            rho_ov_min2=rho_ov_min2,
            rho_ov_max2=rho_ov_max2,
            model_top=model_top_blocks[iblock - 1] if has_top_interface else None,
            sigma_ov_min2=sigma_ov_min2,
            sigma_ov_max2=sigma_ov_max2,
            rho_min=rho_min,     # needed when computing bottom
            rho_max=rho_max,
            t_min=t0,
            t_max=t1,
            sigma_min=sigma_min,
            sigma_max=sigma_max,
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
            sigma0=sigma_min, sigma1=sigma_max
        )
        trained_models.append(loaded_model)
        model_prev = loaded_model
        prev_model_file = file_model

        gc.collect()
        if torch.cuda.is_available():
            torch.cuda.empty_cache()

    return all_hist_local, trained_models


def train_rho_sigma_round2(
    n_epochs=1000,
    rho_blocks=None,
    sigma_blocks=None,
    src_dir="./data_UV_2D",
    out_dir="./data_UV_2D_round2",
):
    """Round-2 pass: warm-start from the 25-cell x 2-block solution completed in
    round 1 (src_dir) as the initial guess, and retrain with the previously
    unimposed right/top interface-continuity loss added. The output goes to
    out_dir, separate from src_dir; the contents of src_dir are never modified.
    """
    all_hist = {}

    global _n_epochs_base
    _n_epochs_base = int(n_epochs)

    os.makedirs(out_dir, exist_ok=True)

    # Rho grids
    n_rho_blocks = 5
    d_rho = 0.350
    rho_overlap = 0.02

    # Sigma grids
    n_sigma_blocks = 5
    d_sigma = 0.350
    sigma_overlap = 0.02

    if rho_blocks is None:
        rho_blocks = list(range(n_rho_blocks))
    if sigma_blocks is None:
        sigma_blocks = list(range(n_sigma_blocks))

    # list holding the round-2 trained models [irho][isigma]
    trained_grid_models = [[None for _ in range(n_sigma_blocks)] for _ in range(n_rho_blocks)]

    def try_load_models_from(r_idx, s_idx, data_dir):
        r_start = r_idx * d_rho
        r_end = (r_idx + 1) * d_rho + (rho_overlap if r_idx < n_rho_blocks - 1 else 0.0)
        s_start = s_idx * d_sigma
        s_end = (s_idx + 1) * d_sigma + (sigma_overlap if s_idx < n_sigma_blocks - 1 else 0.0)

        t_blocks = [(-1.0, 0.0), (-2.0, -1.0)]
        loaded = [None, None]

        for iblock, (t0, t1) in enumerate(t_blocks, start=1):
            if (t0, t1) == (-2.0, -1.0) and getattr(hp, 'skip_2nd', False):
                continue

            file_model = f"{data_dir}/model_higgs_singlet_u_rho{r_idx}_sig{s_idx}_block{iblock}.pt"
            if os.path.exists(file_model):
                loaded[iblock - 1] = load_trained_model(
                    file_model=file_model,
                    t0=t0, t1=t1,
                    rho0=r_start, rho1=r_end,
                    sigma0=s_start, sigma1=s_end
                )
            else:
                # if even one required block is missing, treat the load as incomplete
                return [None, None]
        return loaded

    for irho in rho_blocks:
        for isigma in sigma_blocks:
            if irho >= n_rho_blocks or isigma >= n_sigma_blocks:
                continue

            rho_start = irho * d_rho
            rho_end = (irho + 1) * d_rho + (rho_overlap if irho < n_rho_blocks - 1 else 0.0)
            rho_ov_min = rho_start if irho > 0 else None
            rho_ov_max = rho_start + rho_overlap if irho > 0 else None
            rho_ov_min2 = rho_end - rho_overlap if irho < n_rho_blocks - 1 else None
            rho_ov_max2 = rho_end if irho < n_rho_blocks - 1 else None

            sigma_start = isigma * d_sigma
            sigma_end = (isigma + 1) * d_sigma + (sigma_overlap if isigma < n_sigma_blocks - 1 else 0.0)
            sigma_ov_min = sigma_start if isigma > 0 else None
            sigma_ov_max = sigma_start + sigma_overlap if isigma > 0 else None
            sigma_ov_min2 = sigma_end - sigma_overlap if isigma < n_sigma_blocks - 1 else None
            sigma_ov_max2 = sigma_end if isigma < n_sigma_blocks - 1 else None

            # 1. if this cell is already trained in round 2 (out_dir), load and skip
            current_models = try_load_models_from(irho, isigma, out_dir)
            if current_models[0] is not None:
                print(f"\n--- [round2] rho block {irho} | sigma block {isigma} already trained; loading and skipping ---")
                trained_grid_models[irho][isigma] = current_models
                continue

            # 2. check that the warm-start source (round 1 src_dir) exists; skip this cell if not
            warm_file0 = f"{src_dir}/model_higgs_singlet_u_rho{irho}_sig{isigma}_block1.pt"
            if not os.path.exists(warm_file0):
                print(f"  Warning: round-1 model not found (rho={irho}, sig={isigma}). "
                      f"Skipping this cell: {warm_file0}")
                continue

            print(f"\n===== [round2] rho block {irho}: [{rho_start:.3f}, {rho_end:.3f}] | "
                  f"sigma block {isigma}: [{sigma_start:.3f}, {sigma_end:.3f}] =====")

            # 3. the left (irho-1) and bottom (isigma-1) cells should already be retrained in this round-2 pass, so take them from out_dir
            left_models = [None, None]
            if irho > 0:
                left_models = trained_grid_models[irho - 1][isigma]
                if left_models is None or left_models[0] is None:
                    print(f"  --> loading the round-2 model of the left boundary (rho={irho-1}, sig={isigma}) from disk")
                    left_models = try_load_models_from(irho - 1, isigma, out_dir)
                    trained_grid_models[irho - 1][isigma] = left_models
                    if left_models[0] is None:
                        print(f"  Warning: round-2 left-boundary model not found (rho={irho-1}, sig={isigma})")

            bottom_models = [None, None]
            if isigma > 0:
                bottom_models = trained_grid_models[irho][isigma - 1]
                if bottom_models is None or bottom_models[0] is None:
                    print(f"  --> loading the round-2 model of the bottom boundary (rho={irho}, sig={isigma-1}) from disk")
                    bottom_models = try_load_models_from(irho, isigma - 1, out_dir)
                    trained_grid_models[irho][isigma - 1] = bottom_models
                    if bottom_models[0] is None:
                        print(f"  Warning: round-2 bottom-boundary model not found (rho={irho}, sig={isigma-1})")

            # 4. the right (irho+1) and top (isigma+1) cells are not yet processed in round 2 in this sweep order, so reference the fixed round-1 (src_dir) versions
            right_models = [None, None]
            if irho < n_rho_blocks - 1:
                right_models = try_load_models_from(irho + 1, isigma, src_dir)
                if right_models[0] is None:
                    print(f"  Warning: round-1 right-boundary model not found (rho={irho+1}, sig={isigma})")

            top_models = [None, None]
            if isigma < n_sigma_blocks - 1:
                top_models = try_load_models_from(irho, isigma + 1, src_dir)
                if top_models[0] is None:
                    print(f"  Warning: round-1 top-boundary model not found (rho={irho}, sig={isigma+1})")

            if rho_ov_min is not None:
                print(f"      Overlap Left   (rho)   : [{rho_ov_min:.3f}, {rho_ov_max:.3f}]")
            if sigma_ov_min is not None:
                print(f"      Overlap Bottom (sigma) : [{sigma_ov_min:.3f}, {sigma_ov_max:.3f}]")
            if rho_ov_min2 is not None:
                print(f"      Overlap Right  (rho)   : [{rho_ov_min2:.3f}, {rho_ov_max2:.3f}]")
            if sigma_ov_min2 is not None:
                print(f"      Overlap Top    (sigma) : [{sigma_ov_min2:.3f}, {sigma_ov_max2:.3f}]")

            hist_list, trained_models = train_rho_sigma_window_joint_round2(
                irho=irho,
                isigma=isigma,
                rho_min=rho_start,
                rho_max=rho_end,
                sigma_min=sigma_start,
                sigma_max=sigma_end,
                src_dir=src_dir,
                out_dir=out_dir,
                rho_ov_min=rho_ov_min,
                rho_ov_max=rho_ov_max,
                sigma_ov_min=sigma_ov_min,
                sigma_ov_max=sigma_ov_max,
                rho_ov_min2=rho_ov_min2,
                rho_ov_max2=rho_ov_max2,
                sigma_ov_min2=sigma_ov_min2,
                sigma_ov_max2=sigma_ov_max2,
                model_left_blocks=left_models,
                model_bottom_blocks=bottom_models,
                model_right_blocks=right_models,
                model_top_blocks=top_models,
                w_overlap=1e-2
            )

            all_hist[f"rho{irho}_sig{isigma}"] = hist_list
            trained_grid_models[irho][isigma] = trained_models

    return all_hist