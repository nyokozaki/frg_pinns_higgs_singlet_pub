#!/usr/bin/env python
"""
frg_pinns_higgs_singlet: orchestrator that runs rho/sigma training + image
generation into plots_pinn/ in one shot.

- completed blocks (model_*.pt with both block1 and block2 present for
  rho{i}/sigma{i}) are skipped automatically, resuming from the first
  incomplete block.
  (train_networks.py's own in-block checkpoint resume still works, so an
  interruption partway through a block resumes at the epoch level.)
- if Paperspace restarts, just re-run the same command.
- pass --fresh explicitly only when you want to delete the existing
  data_UV/data_sigma and retrain from scratch (nothing is deleted by default).

Usage:
    python run_pinn_pipeline.py                       # resume/train rho+sigma for 10000 epochs, then generate plots when done
    python run_pinn_pipeline.py --epochs 6000
    python run_pinn_pipeline.py --modes rho
    python run_pinn_pipeline.py --fresh                # delete data_UV/data_sigma, then retrain
    python run_pinn_pipeline.py --plot-only --tag T200_h12_
    python run_pinn_pipeline.py --sequential           # run rho then sigma in order (not in parallel)
"""
import argparse
import os
import shutil
import subprocess
import sys
import time
from pathlib import Path

BASE = Path(__file__).resolve().parent

N_BLOCKS = 5
T_BLOCKS = (1, 2)

DATA_DIRS = {
    "rho": BASE / "data_UV",
    "sigma": BASE / "data_sigma",
}


def completed_range(mode):
    """Return the 'N-4' string from the first incomplete block to the last block.
    Return None if all blocks are complete."""
    d = DATA_DIRS[mode]
    for i in range(N_BLOCKS):
        done = all(
            (d / f"model_higgs_singlet_u_{mode}{i}_block{b}.pt").exists()
            for b in T_BLOCKS
        )
        if not done:
            return f"{i}-{N_BLOCKS - 1}"
    return None


def block_status_table(mode):
    d = DATA_DIRS[mode]
    rows = []
    for i in range(N_BLOCKS):
        marks = []
        for b in T_BLOCKS:
            f = d / f"model_higgs_singlet_u_{mode}{i}_block{b}.pt"
            marks.append("done" if f.exists() else "----")
        rows.append(f"  {mode}{i}: block1={marks[0]} block2={marks[1]}")
    return "\n".join(rows)


def do_fresh(modes):
    for m in modes:
        d = DATA_DIRS[m]
        if d.exists():
            print(f"[fresh] removing {d}")
            shutil.rmtree(d)
        else:
            print(f"[fresh] {d} does not exist, nothing to remove")


def launch_training(mode, epochs, rng):
    blockflag = f"--{mode}-blocks"
    logpath = BASE / f"train_{mode}_{epochs}.log"
    cmd = [sys.executable, "-u", "train_networks.py", mode,
           "--epochs", str(epochs), blockflag, rng]
    print(f"[{mode}] launching: {' '.join(cmd)}  (resume range {rng})")
    print(f"[{mode}] log -> {logpath}")
    logf = open(logpath, "a")
    logf.write(f"\n=== run_pinn_pipeline.py relaunch {time.ctime()} range={rng} epochs={epochs} ===\n")
    logf.flush()
    proc = subprocess.Popen(cmd, cwd=str(BASE), stdout=logf, stderr=subprocess.STDOUT)
    return proc, logf


def run_training(modes, epochs, sequential):
    procs = {}
    logs = {}
    for m in modes:
        rng = completed_range(m)
        if rng is None:
            print(f"[{m}] already fully trained ({N_BLOCKS * len(T_BLOCKS)}/{N_BLOCKS * len(T_BLOCKS)} blocks) -- skip")
            continue
        proc, logf = launch_training(m, epochs, rng)
        procs[m] = proc
        logs[m] = logf
        if sequential:
            rc = proc.wait()
            print(f"[{m}] training process exited with code {rc}")

    if not sequential:
        for m, proc in procs.items():
            rc = proc.wait()
            print(f"[{m}] training process exited with code {rc}")

    for logf in logs.values():
        logf.close()


def plottable_modes(modes):
    ok = []
    for m in modes:
        if completed_range(m) is None:
            ok.append(m)
        else:
            print(f"[{m}] WARNING: still incomplete after training run -- skipping its plots\n{block_status_table(m)}")
    return ok


def run_and_save(fn, names, outdir, tag, **kwargs):
    import matplotlib.pyplot as plt
    plt.close("all")
    fn(**kwargs)
    fignums = plt.get_fignums()
    if len(fignums) != len(names):
        print(f"WARNING: expected {len(names)} figures from {fn.__name__}, got {len(fignums)}")
    for num, name in zip(fignums, names):
        path = outdir / f"{tag}{name}"
        plt.figure(num).savefig(path, dpi=150, bbox_inches="tight")
        print("saved", path)
    plt.close("all")


def generate_plots(modes, tag):
    os.chdir(BASE)
    sys.path.insert(0, str(BASE))

    import matplotlib
    matplotlib.use("Agg")

    outdir = BASE / "plots_pinn"
    outdir.mkdir(exist_ok=True)

    if "rho" in modes:
        import show_UV1D
        print("=== rho: plot_u ===")
        run_and_save(show_UV1D.plot_u, ["rho_u_vs_reference.png"], outdir, tag)
        print("=== rho: plot_training_histories_all_rho ===")
        run_and_save(show_UV1D.plot_training_histories_all_rho, ["rho_training_history.png"], outdir, tag)
        print("=== rho: plot_residual ===")
        run_and_save(show_UV1D.plot_residual, ["rho_residual_heatmap.png", "rho_residual_vs_t.png"], outdir, tag)

    if "sigma" in modes:
        import show_1D_sigma
        print("=== sigma: plot_u ===")
        run_and_save(show_1D_sigma.plot_u, ["sigma_u_vs_reference.png"], outdir, tag)
        print("=== sigma: plot_training_histories_all_sigma ===")
        run_and_save(show_1D_sigma.plot_training_histories_all_sigma, ["sigma_training_history.png"], outdir, tag)
        print("=== sigma: plot_residual ===")
        run_and_save(show_1D_sigma.plot_residual, ["sigma_residual_heatmap.png", "sigma_residual_vs_t.png"], outdir, tag)


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--epochs", type=int, default=10000, help="epochs per block (default: 10000)")
    ap.add_argument("--modes", default="rho,sigma", help="comma-separated subset of rho,sigma (default: rho,sigma)")
    ap.add_argument("--fresh", action="store_true",
                     help="delete data_UV/data_sigma for the selected modes before training (irreversible)")
    ap.add_argument("--sequential", action="store_true",
                     help="train modes one after another instead of concurrently")
    ap.add_argument("--plot-only", action="store_true",
                     help="skip training entirely, only (re)generate plots from whatever is already trained")
    ap.add_argument("--tag", default="", help="prefix for output PNG filenames, e.g. T200_h12_")
    args = ap.parse_args()

    modes = [m.strip() for m in args.modes.split(",") if m.strip()]
    for m in modes:
        if m not in DATA_DIRS:
            ap.error(f"unknown mode {m!r}, must be one of {list(DATA_DIRS)}")

    if args.fresh:
        do_fresh(modes)

    if not args.plot_only:
        run_training(modes, args.epochs, args.sequential)

    modes_for_plot = plottable_modes(modes)
    if not modes_for_plot:
        print("No mode is fully trained yet -- nothing to plot.")
        return

    generate_plots(modes_for_plot, args.tag)
    print("DONE.")


if __name__ == "__main__":
    main()
