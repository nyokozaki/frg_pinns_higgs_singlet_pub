#!/usr/bin/env python3
"""
The w-residual version of relax_full.py (Higgs-only, 1D rho).

Write u(t,rho) = u_tree_exact(t,rho) + w(t,rho) and make the Newton free
variable w rather than u. Since u_tree_exact is the exact solution of the free
flow (rloop=0, eta=0) with zero discretization error, this split means the
equation satisfied by w (flow_equation_higgs_only.flow_rhs_w) has the canonical
scaling "-4u+2rho u_rho" cancelled exactly on the u_tree side, so only w has to
be discretized.

Boundary condition: w(t=0) = u_thermal_finiteT(t=0,...) (since
u_seed = u_tree_exact + u_thermal_finiteT, subtracting u_tree_exact(t=0) leaves
exactly w(t=0)).

Usage:
    python relax_full_w.py --n-rho 21 --n-t 40
"""

import argparse
import os
import time

import numpy as np
from scipy.optimize import newton_krylov

import _pathsetup  # noqa: F401
from perturbation.config_params import t_range, tau_uv
from grid_fd_higgs_only import Grid1D
from seed_potential_higgs_only import u_tree_exact, u_thermal_finiteT
from flow_equation_higgs_only import flow_rhs_w


_KH_THERMAL = 0.2


def thermal_seed_ansatz(t_phys, rho_phys):
    tau_t = tau_uv * np.exp(-t_phys)
    return _KH_THERMAL * rho_phys * tau_t ** 2


def parse_args():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--n-rho", type=int, default=21)
    p.add_argument("--n-t", type=int, default=40)
    p.add_argument("--rho-max", type=float, default=1.75)
    p.add_argument("--t0", type=float, default=0.0)
    p.add_argument("--t-end", type=float, default=None)
    p.add_argument("--f-tol", type=float, default=1e-8)
    p.add_argument("--maxiter", type=int, default=100)
    p.add_argument("--inner-maxiter", type=int, default=30)
    p.add_argument("--method", type=str, default="lgmres")
    p.add_argument("--scheme", type=str, default="cn", choices=["cn", "be"])
    p.add_argument("--warm-start", type=str, default=None,
                   help="use a previous relax_full_w.py output npz (W_all_raw) as the initial guess.")
    p.add_argument("--out", type=str, default="results/relax_full_w.npz")
    return p.parse_args()


def main():
    args = parse_args()
    t_end = args.t_end if args.t_end is not None else t_range

    grid = Grid1D(rho_max=args.rho_max, n_rho=args.n_rho)
    n_rho = grid.n_rho

    t_grid = np.linspace(args.t0, t_end, args.n_t + 1)
    dt = t_grid[1] - t_grid[0]

    W0 = u_thermal_finiteT(args.t0, grid.RHO)
    print(f"Grid: {n_rho} spatial (rho), {args.n_t} time steps (t: {args.t0} -> {t_end}) [w-residual formulation]")
    print(f"max|W0| (boundary, t={args.t0}) = {np.abs(W0).max():.6e}")

    W_ansatz_all = np.stack([thermal_seed_ansatz(tn, grid.RHO) for tn in t_grid])

    if args.warm_start is not None:
        print(f"Loading warm start from {args.warm_start} (W_all_raw)...")
        prev = np.load(args.warm_start)
        W_init = prev["W_all_raw"].reshape(args.n_t, n_rho).copy()
    else:
        print("Building warm start from thermal_seed_ansatz (closed form)...")
        W_init = W_ansatz_all[1:].copy()

    def rhs_at(n, W):
        return flow_rhs_w(t_grid[n], W, grid, scheme="central")

    def residual(W_free):
        W_all = W_free.reshape(args.n_t, n_rho)
        R = np.empty_like(W_all)
        W_prev = W0
        rhs_prev = rhs_at(0, W_prev)
        for n in range(args.n_t):
            W_curr = W_all[n]
            rhs_curr = rhs_at(n + 1, W_curr)
            if args.scheme == "cn":
                R[n] = (W_curr - W_prev) - 0.5 * dt * (rhs_prev + rhs_curr)
            else:
                R[n] = (W_curr - W_prev) - dt * rhs_curr
            W_prev = W_curr
            rhs_prev = rhs_curr
        return R.ravel()

    W_init_flat = W_init.ravel()
    r0 = residual(W_init_flat)
    print(f"initial residual norm: {np.linalg.norm(r0):.6e} (max abs {np.max(np.abs(r0)):.6e})")

    start = time.time()
    try:
        W_sol_flat = newton_krylov(
            residual, W_init_flat,
            method=args.method, f_tol=args.f_tol, maxiter=args.maxiter,
            inner_maxiter=args.inner_maxiter, verbose=True,
        )
        success = True
        message = "converged"
    except Exception as e:
        print(f"[WARNING] newton_krylov did not converge cleanly: {e}")
        if e.args and hasattr(e.args[0], "reshape"):
            W_sol_flat = e.args[0]
        else:
            W_sol_flat = W_init_flat
        success = False
        message = str(e)
    elapsed = time.time() - start
    r_final = residual(W_sol_flat)
    print(f"final residual norm: {np.linalg.norm(r_final):.6e} "
          f"(max abs {np.max(np.abs(r_final)):.6e}), elapsed {elapsed:.1f}s")

    W_all = W_sol_flat.reshape(args.n_t, n_rho)
    W_final = W_all[-1]
    U_tree_final = u_tree_exact(t_end, grid.RHO)
    U_final = U_tree_final + W_final

    origin = U_final[0]
    U_final_norm = U_final - origin
    print(f"U(t_end) [w-formulation, origin-normalized] range: "
          f"[{U_final_norm.min():.6e}, {U_final_norm.max():.6e}]")
    print(f"U(rho_max, t_end) = {U_final_norm[-1]:.6f}")

    U_tree_all = np.stack([u_tree_exact(tn, grid.RHO) for tn in t_grid])
    U_all_raw = U_tree_all[1:] + W_all

    out_dir = os.path.dirname(args.out)
    if out_dir:
        os.makedirs(out_dir, exist_ok=True)
    np.savez(
        args.out,
        rho=grid.rho, t=t_grid,
        U_final=U_final_norm, U_all=U_all_raw - origin,
        U_all_raw=U_all_raw,
        W_all_raw=W_all,
        success=success, message=message,
        final_residual_norm=np.linalg.norm(r_final), elapsed=elapsed,
    )
    print(f"Saved to {args.out}")


if __name__ == "__main__":
    main()
