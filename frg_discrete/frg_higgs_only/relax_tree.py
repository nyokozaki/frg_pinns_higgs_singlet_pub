#!/usr/bin/env python3
"""
Solve the tree-only (rloop=eta=0) free flow with a relaxation method
(discretize space x time together and solve one global system) -- the
Higgs-only, 1D rho version.

Equation: du/dt = -4u + 2*rho*u_rho  (t: 0 -> t_range, t_range<0)

A linear PDE, so no Newton iteration is needed (one linear solve converges
exactly). Space derivatives use a central difference, time integration uses
Crank-Nicolson, and all time steps are solved as one block-bidiagonal linear
system.

Usage:
    python relax_tree.py --n-rho 41 --n-t 100
"""

import argparse
import os
import time

import numpy as np
import scipy.sparse as sp
import scipy.sparse.linalg as spla

import _pathsetup  # noqa: F401
from perturbation.config_params import t_range
from seed_potential_higgs_only import u_seed, u_tree_exact


def _fd_first_deriv_coeffs(offsets):
    offsets = np.asarray(offsets, dtype=float)
    m = len(offsets)
    M = np.vstack([offsets ** p for p in range(m)])
    rhs = np.zeros(m)
    rhs[1] = 1.0
    return np.linalg.solve(M, rhs)


def build_D1_central(n, h, order=2):
    """Return the sparse first-derivative matrix (same stencil as frg_discrete/grid_fd.py Grid2D._first_deriv_1d)."""
    rows, cols, vals = [], [], []

    def add(i, j, v):
        rows.append(i); cols.append(j); vals.append(v)

    def add_stencil(i, offsets):
        c = _fd_first_deriv_coeffs(offsets) / h
        for off, cv in zip(offsets, c):
            add(i, i + int(off), cv)

    if order == 2:
        for i in range(1, n - 1):
            add(i, i - 1, -1.0 / (2 * h))
            add(i, i + 1, 1.0 / (2 * h))
        add(0, 0, -3.0 / (2 * h)); add(0, 1, 4.0 / (2 * h)); add(0, 2, -1.0 / (2 * h))
        add(n - 1, n - 1, 3.0 / (2 * h)); add(n - 1, n - 2, -4.0 / (2 * h)); add(n - 1, n - 3, 1.0 / (2 * h))
    elif order == 4:
        if n < 5:
            raise ValueError("order=4 requires at least 5 grid points")
        add_stencil(0, [0, 1, 2, 3, 4])
        add_stencil(1, [-1, 0, 1, 2, 3])
        for i in range(2, n - 2):
            add_stencil(i, [-2, -1, 0, 1, 2])
        add_stencil(n - 2, [-3, -2, -1, 0, 1])
        add_stencil(n - 1, [-4, -3, -2, -1, 0])
    else:
        raise ValueError(f"unsupported order={order} (2 or 4)")

    return sp.csr_matrix((vals, (rows, cols)), shape=(n, n))


def build_spatial_operator(rho, order=2):
    """Build A = -4*I + 2*rho*d/drho as an n_rho x n_rho sparse matrix."""
    n_rho = len(rho)
    drho = rho[1] - rho[0]

    D1_rho = build_D1_central(n_rho, drho, order=order)
    Rho_diag = sp.diags(rho)

    A = -4.0 * sp.identity(n_rho, format="csr") + 2.0 * (Rho_diag @ D1_rho)
    return A.tocsr()


def solve_relaxation(rho, U0, t0, t_end, n_t):
    """
    Relaxation method: discretize the t direction into Nt+1 points with
    Crank-Nicolson and solve the block-bidiagonal global linear system.

    (I - 0.5*dt*A) U^n = (I + 0.5*dt*A) U^{n-1},  n=1..Nt
    U^0 = U0 (UV boundary condition, fixed)

    Since A and dt do not depend on t (tree-only), the LHS matrix is the same
    for all steps. Do one LU factorization and solve all steps by block
    elimination.
    """
    N = len(rho)
    A = build_spatial_operator(rho)

    t_grid = np.linspace(t0, t_end, n_t + 1)
    dt = t_grid[1] - t_grid[0]

    LHS = (sp.identity(N, format="csc") - 0.5 * dt * A).tocsc()
    RHSmat = (sp.identity(N, format="csr") + 0.5 * dt * A).tocsr()

    lu = spla.splu(LHS)

    U = U0.copy()
    U_all = np.empty((n_t + 1, N))
    U_all[0] = U
    for n in range(1, n_t + 1):
        b = RHSmat @ U
        U = lu.solve(b)
        U_all[n] = U

    return t_grid, U_all


def analytic_tree_solution(t_end, RHO, use_full_seed):
    """Evaluate the exact solution along the characteristics rho(0)=rho(t_end)*exp(2*t_end)."""
    scale = np.exp(2.0 * t_end)
    rho0 = RHO * scale
    if use_full_seed:
        residual0 = u_seed(0.0, rho0) - u_tree_exact(0.0, rho0)
    else:
        residual0 = np.zeros_like(rho0)
    return u_tree_exact(t_end, RHO) + residual0 * np.exp(-4.0 * t_end)


def parse_args():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--n-rho", type=int, default=41, help="number of grid points in rho")
    p.add_argument("--n-t", type=int, default=100, help="number of time steps")
    p.add_argument("--rho-max", type=float, default=1.75)
    p.add_argument("--t0", type=float, default=0.0)
    p.add_argument("--t-end", type=float, default=None)
    p.add_argument("--full-seed", action="store_true",
                    help="use u_seed (tree+thermal) as the initial condition, not just the tree polynomial")
    p.add_argument("--out", type=str, default="results/relax_tree.npz")
    return p.parse_args()


def main():
    args = parse_args()
    t_end = args.t_end if args.t_end is not None else t_range

    rho = np.linspace(0.0, args.rho_max, args.n_rho)

    if args.full_seed:
        U0 = u_seed(args.t0, rho)
    else:
        U0 = u_tree_exact(args.t0, rho)

    print(f"Grid: {args.n_rho} spatial (rho), {args.n_t} time steps "
          f"(t: {args.t0} -> {t_end}), seed={'full (tree+thermal)' if args.full_seed else 'pure tree'}")

    start = time.time()
    t_grid, U_all = solve_relaxation(rho, U0, args.t0, t_end, args.n_t)
    elapsed = time.time() - start
    print(f"Solved in {elapsed:.2f}s ({args.n_t} block-elimination steps, 1 LU factorization)")

    U_final = U_all[-1]

    U_exact = analytic_tree_solution(t_end, rho, use_full_seed=args.full_seed)

    err = U_final - U_exact
    max_abs_err = np.max(np.abs(err))
    denom = np.max(np.abs(U_exact))
    rel_err = max_abs_err / denom
    print(f"U_num(t_end)  range: [{U_final.min():.6e}, {U_final.max():.6e}]")
    print(f"U_exact(t_end) range: [{U_exact.min():.6e}, {U_exact.max():.6e}]")
    print(f"max abs error = {max_abs_err:.6e}")
    print(f"relative error (max_abs / max|U_exact|) = {rel_err:.6e} ({rel_err*100:.4f}%)")

    print(f"U_num(rho_max, t_end) = {U_final[-1]:.6f}, U_exact = {U_exact[-1]:.6f}")

    out_dir = os.path.dirname(args.out)
    if out_dir:
        os.makedirs(out_dir, exist_ok=True)
    np.savez(
        args.out,
        rho=rho, t=t_grid,
        U_final=U_final, U_exact=U_exact,
        max_abs_err=max_abs_err, rel_err=rel_err,
        full_seed=args.full_seed, elapsed=elapsed,
    )
    print(f"Saved to {args.out}")


if __name__ == "__main__":
    main()
