#!/usr/bin/env python3
"""
Solve the tree-only (rloop=eta=0) free flow with a relaxation method
(discretize space x time together and solve one global system), rather than by
"shooting" (forward integration as an initial-value problem from t=0).

Equation: du/dt = -4u + 2*rho*u_rho + 2*sigma*u_sigma  (t: 0 -> t_range, t_range<0)

This is a linear PDE, so no Newton iteration is needed (one linear solve
converges exactly). Space derivatives use a central difference (the same
second-order stencil as the "central" scheme in grid_fd.Grid2D), time
integration uses Crank-Nicolson (central, second order), and all time steps are
solved as one block-bidiagonal linear system.

Solving this block-bidiagonal system "all at once" is mathematically exactly
equivalent to CN time marching with the same discretization (block elimination
= sequential marching, because of the lower-triangular block structure). So for
the tree-only (linear) case the benefit specific to a relaxation method (that
Newton's global search for a self-consistent solution avoids the divergence-
prone forward shooting) does not yet appear. The real aims here are:

  1. Check whether a central (second-order, symmetric) difference + a
     non-adaptive global linear solve improves on the 27% error of the previous
     upwind (first-order) + adaptive-step BDF.
  2. Build the framework of "holding the whole trajectory as one system", which
     is used later when rloop is added (rloop makes the system nonlinear, and
     then Newton iteration genuinely matters).

Usage:
    python relax_tree.py --n-rho 41 --n-sigma 41 --n-t 100
"""

import argparse
import os
import time

import numpy as np
import scipy.sparse as sp
import scipy.sparse.linalg as spla

import _pathsetup  # noqa: F401
from perturbation.config_params import t_range
from seed_potential import u_seed, u_tree_exact


def _fd_first_deriv_coeffs(offsets):
    """Finite-difference coefficients for a first derivative on integer offsets
    (positions relative to the grid point), obtained exactly by the Fornberg
    method (method of undetermined coefficients).

    Returns c such that sum_k c_k * f(x+offsets_k*h) = h*f'(x) + O(h^len(offsets))
    (an m=len(offsets)-point stencil, accurate to order m-1).
    """
    offsets = np.asarray(offsets, dtype=float)
    m = len(offsets)
    M = np.vstack([offsets ** p for p in range(m)])
    rhs = np.zeros(m)
    rhs[1] = 1.0
    return np.linalg.solve(M, rhs)


def build_D1_central(n, h, order=2):
    """Return the sparse first-derivative matrix.

    order=2: second-order central difference (3-point) in the interior,
        second-order one-sided difference at the boundaries (the same stencil as
        frg_discrete/grid_fd.py Grid2D._first_deriv_1d).
    order=4: fourth-order central difference (5-point) in the interior, a
        fourth-order asymmetric 5-point stencil for the near-boundary point, and
        a fourth-order one-sided 5-point difference at the edge.
        All coefficients are derived by the Fornberg method, so raising the order
        does not introduce hand-typed coefficient errors.
    """
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
            raise ValueError("order=4 requires at least 5 grid points per direction")
        add_stencil(0, [0, 1, 2, 3, 4])
        add_stencil(1, [-1, 0, 1, 2, 3])
        for i in range(2, n - 2):
            add_stencil(i, [-2, -1, 0, 1, 2])
        add_stencil(n - 2, [-3, -2, -1, 0, 1])
        add_stencil(n - 1, [-4, -3, -2, -1, 0])
    else:
        raise ValueError(f"unsupported order={order} (2 or 4)")

    return sp.csr_matrix((vals, (rows, cols)), shape=(n, n))


def build_spatial_operator(rho, sigma, order=2):
    """Build A = -4*I + 2*rho*d/drho + 2*sigma*d/dsigma as an (n_rho*n_sigma)^2
    sparse matrix. The flatten order is U.ravel() (rho-major, C order)."""
    n_rho, n_sigma = len(rho), len(sigma)
    drho = rho[1] - rho[0]
    dsigma = sigma[1] - sigma[0]

    D1_rho = build_D1_central(n_rho, drho, order=order)
    D1_sigma = build_D1_central(n_sigma, dsigma, order=order)

    I_nr = sp.identity(n_rho, format="csr")
    I_ns = sp.identity(n_sigma, format="csr")

    Drho2D = sp.kron(D1_rho, I_ns, format="csr")
    Dsigma2D = sp.kron(I_nr, D1_sigma, format="csr")

    rho_flat = np.repeat(rho, n_sigma)
    sigma_flat = np.tile(sigma, n_rho)

    Rho_diag = sp.diags(rho_flat)
    Sigma_diag = sp.diags(sigma_flat)

    N = n_rho * n_sigma
    A = -4.0 * sp.identity(N, format="csr") + 2.0 * (Rho_diag @ Drho2D) + 2.0 * (Sigma_diag @ Dsigma2D)
    return A.tocsr()


def solve_relaxation(rho, sigma, U0_flat, t0, t_end, n_t):
    """
    Relaxation method: discretize the t direction into Nt+1 points with
    Crank-Nicolson and solve the block-bidiagonal global linear system.

    (I - 0.5*dt*A) U^n = (I + 0.5*dt*A) U^{n-1},  n=1..Nt
    U^0 = U0 (UV boundary condition, fixed)

    Since A and dt do not depend on t (tree-only), the LHS matrix is the same for
    all steps. Do one LU factorization, then solve all steps by block
    elimination (mathematically equivalent to solving this block-bidiagonal
    global system exactly).
    """
    N = len(rho) * len(sigma)
    A = build_spatial_operator(rho, sigma)

    t_grid = np.linspace(t0, t_end, n_t + 1)
    dt = t_grid[1] - t_grid[0]

    LHS = (sp.identity(N, format="csc") - 0.5 * dt * A).tocsc()
    RHSmat = (sp.identity(N, format="csr") + 0.5 * dt * A).tocsr()

    lu = spla.splu(LHS)

    U = U0_flat.copy()
    U_all = np.empty((n_t + 1, N))
    U_all[0] = U
    for n in range(1, n_t + 1):
        b = RHSmat @ U
        U = lu.solve(b)
        U_all[n] = U

    return t_grid, U_all


def analytic_tree_solution(t_end, RHO, SIGMA, use_full_seed):
    """Evaluate the exact solution along the characteristics rho(0)=rho(t_end)*exp(2*t_end)."""
    scale = np.exp(2.0 * t_end)
    rho0 = RHO * scale
    sigma0 = SIGMA * scale
    if use_full_seed:
        residual0 = u_seed(0.0, rho0, sigma0) - u_tree_exact(0.0, rho0, sigma0)
    else:
        residual0 = np.zeros_like(rho0)
    return u_tree_exact(t_end, RHO, SIGMA) + residual0 * np.exp(-4.0 * t_end)


def parse_args():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--n-rho", type=int, default=41, help="number of grid points in rho")
    p.add_argument("--n-sigma", type=int, default=41, help="number of grid points in sigma")
    p.add_argument("--n-t", type=int, default=100, help="number of time steps")
    p.add_argument("--rho-max", type=float, default=1.75)
    p.add_argument("--sigma-max", type=float, default=1.75)
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
    sigma = np.linspace(0.0, args.sigma_max, args.n_sigma)
    RHO, SIGMA = np.meshgrid(rho, sigma, indexing="ij")

    if args.full_seed:
        U0 = u_seed(args.t0, RHO, SIGMA)
    else:
        U0 = u_tree_exact(args.t0, RHO, SIGMA)

    print(f"Grid: {args.n_rho}x{args.n_sigma} spatial, {args.n_t} time steps "
          f"(t: {args.t0} -> {t_end}), seed={'full (tree+thermal)' if args.full_seed else 'pure tree'}")

    start = time.time()
    t_grid, U_all_flat = solve_relaxation(rho, sigma, U0.ravel(), args.t0, t_end, args.n_t)
    elapsed = time.time() - start
    print(f"Solved in {elapsed:.2f}s ({args.n_t} block-elimination steps, 1 LU factorization)")

    U_final = U_all_flat[-1].reshape(args.n_rho, args.n_sigma)

    U_exact = analytic_tree_solution(t_end, RHO, SIGMA, use_full_seed=args.full_seed)

    err = U_final - U_exact
    max_abs_err = np.max(np.abs(err))
    denom = np.max(np.abs(U_exact))
    rel_err = max_abs_err / denom
    print(f"U_num(t_end)  range: [{U_final.min():.6e}, {U_final.max():.6e}]")
    print(f"U_exact(t_end) range: [{U_exact.min():.6e}, {U_exact.max():.6e}]")
    print(f"max abs error = {max_abs_err:.6e}")
    print(f"relative error (max_abs / max|U_exact|) = {rel_err:.6e} ({rel_err*100:.4f}%)")

    i_edge = args.n_rho - 1
    print(f"U_num(rho_max, sigma=0, t_end) = {U_final[i_edge, 0]:.6f}, "
          f"U_exact = {U_exact[i_edge, 0]:.6f}")

    out_dir = os.path.dirname(args.out)
    if out_dir:
        os.makedirs(out_dir, exist_ok=True)
    np.savez(
        args.out,
        rho=rho, sigma=sigma, t=t_grid,
        U_final=U_final, U_exact=U_exact,
        max_abs_err=max_abs_err, rel_err=rel_err,
        full_seed=args.full_seed, elapsed=elapsed,
    )
    print(f"Saved to {args.out}")


if __name__ == "__main__":
    main()
