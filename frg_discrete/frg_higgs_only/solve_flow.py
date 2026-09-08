#!/usr/bin/env python3
"""
Singlet-free version of frg_discrete/solve_flow.py (Higgs-only, 1D rho).

Solves the FRG flow equation the "conventional" way, with method of lines +
scipy.integrate.solve_ivp (actually forward-integrating from t=0 to t_end), as a
reference for comparison with relax_full.py (the Newton-Krylov all-times
simultaneous relaxation method).

Usage:
    python solve_flow.py --n-rho 21 --out results/solve_flow.npz
"""

import argparse
import os
import time

import numpy as np
import scipy.sparse as sp
from scipy.integrate import solve_ivp

import _pathsetup  # noqa: F401
from perturbation.config_params import t_range
from grid_fd_higgs_only import Grid1D
from seed_potential_higgs_only import u_seed, u_tree_exact
from flow_equation_higgs_only import flow_rhs, flow_rhs_w


def build_jac_sparsity(n_rho, bandwidth=3):
    n = n_rho
    idx = np.arange(n)
    rows, cols = [], []
    for d in range(-bandwidth, bandwidth + 1):
        i_src = np.arange(n)
        i_dst = i_src + d
        valid = (i_dst >= 0) & (i_dst < n)
        rows.append(idx[i_src[valid]])
        cols.append(idx[i_dst[valid]])
    rows = np.concatenate(rows)
    cols = np.concatenate(cols)
    data = np.ones_like(rows, dtype=np.float64)
    return sp.coo_matrix((data, (rows, cols)), shape=(n, n)).tocsr()


def parse_args():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--n-rho", type=int, default=21, help="number of grid points in rho")
    p.add_argument("--rho-max", type=float, default=1.75)
    p.add_argument("--t0", type=float, default=0.0)
    p.add_argument("--t-end", type=float, default=None)
    p.add_argument("--n-save", type=int, default=21)
    p.add_argument("--method", type=str, default="BDF",
                   choices=["RK45", "RK23", "DOP853", "Radau", "BDF", "LSODA"])
    p.add_argument("--scheme", type=str, default="upwind", choices=["upwind", "central"])
    p.add_argument("--formulation", type=str, default="residual", choices=["residual", "full"])
    p.add_argument("--rtol", type=float, default=1e-6)
    p.add_argument("--atol", type=float, default=1e-8)
    p.add_argument("--max-step", type=float, default=np.inf)
    p.add_argument("--out", type=str, default="results/solve_flow.npz")
    p.add_argument("--normalize-origin", action=argparse.BooleanOptionalAction, default=True)
    return p.parse_args()


def main():
    args = parse_args()
    t_end = args.t_end if args.t_end is not None else t_range

    grid = Grid1D(rho_max=args.rho_max, n_rho=args.n_rho)

    print(f"Grid: n_rho={grid.n_rho}, drho={grid.drho:.4e}")
    print(f"Integrating t: {args.t0} -> {t_end}  (method={args.method}, scheme={args.scheme})")

    U0 = u_seed(args.t0, grid.RHO)
    print(f"U0 range: [{U0.min():.6e}, {U0.max():.6e}]")
    print(f"formulation={args.formulation}")

    t_eval = np.linspace(args.t0, t_end, args.n_save)

    if args.formulation == "residual":
        Y0 = U0 - u_tree_exact(args.t0, grid.RHO)

        def rhs(t, y):
            return flow_rhs_w(t, y, grid, scheme=args.scheme)
    else:
        Y0 = U0

        def rhs(t, y):
            return flow_rhs(t, y, grid, scheme=args.scheme)

    extra_kwargs = {}
    if args.method in ("Radau", "BDF"):
        extra_kwargs["jac_sparsity"] = build_jac_sparsity(grid.n_rho, bandwidth=3)

    start = time.time()
    sol = solve_ivp(
        rhs, (args.t0, t_end), Y0,
        method=args.method, t_eval=t_eval,
        rtol=args.rtol, atol=args.atol, max_step=args.max_step,
        **extra_kwargs,
    )
    elapsed = time.time() - start

    if not sol.success:
        print(f"[WARNING] Integration did not fully succeed: {sol.message}")
    else:
        print(f"Integration finished in {elapsed:.1f}s, {sol.nfev} RHS evaluations.")

    Y_t = sol.y.T  # shape (n_saved, n_rho)
    t_saved = sol.t

    if args.formulation == "residual":
        U_tree_saved = np.stack([u_tree_exact(tk, grid.RHO) for tk in t_saved], axis=0)
        U_t = U_tree_saved + Y_t
    else:
        U_t = Y_t

    n_nan = np.sum(~np.isfinite(U_t))
    if n_nan > 0:
        print(f"[WARNING] {n_nan} non-finite values found in the solution.")

    if args.normalize_origin:
        origin = U_t[:, 0:1]
        print(f"Normalizing origin: U(0,t) was in [{origin.min():.6e}, {origin.max():.6e}], now set to 0.")
        U_t = U_t - origin

    U_final = U_t[-1]
    print(f"U(t_end) range: [{np.nanmin(U_final):.6e}, {np.nanmax(U_final):.6e}]")
    print(f"U(rho_max, t_end) = {U_final[-1]:.6f}")

    out_dir = os.path.dirname(args.out)
    if out_dir:
        os.makedirs(out_dir, exist_ok=True)

    np.savez(
        args.out,
        rho=grid.rho, t=t_saved, U=U_t,
        t0=args.t0, t_end=t_end,
        success=sol.success, nfev=sol.nfev, elapsed=elapsed,
        normalize_origin=args.normalize_origin, formulation=args.formulation,
    )
    print(f"Saved to {args.out}")


if __name__ == "__main__":
    main()
