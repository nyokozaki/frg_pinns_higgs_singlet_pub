#!/usr/bin/env python3
"""
tree-only (rloop=eta=0) の自由フローを、"shooting" (t=0からの初期値問題として
forward積分) ではなく、リラクゼーション法 (space x time をまとめて離散化し、
1つの大域的な連立方程式として解く) で解く。

方程式: du/dt = -4u + 2*rho*u_rho + 2*sigma*u_sigma  (t: 0 -> t_range, t_range<0)

これは線形PDEなので、Newton反復は不要 (1回の線形ソルブで厳密に収束する)。
空間微分は中心差分 (grid_fd.Grid2D の "central" スキームと同じ2次精度
ステンシル)、時間積分は Crank-Nicolson (中心差分、2次精度) で離散化し、
全時刻ステップをまとめた1つのブロック双対角線形系として解く。

このブロック双対角系を「まとめて」解くのは、数学的には同じ離散化での
CN時間マーチングと厳密に同値 (下三角ブロック構造なので、ブロック消去 =
逐次マーチング)。したがって tree-only (線形) の場合、リラクゼーション法
固有の御利益 (Newtonによる大域的な自己無撞着解の探索が、発散しがちな
forward shootingを回避できること) はまだ現れない。ここでの本当の狙いは:

  1. central (2次精度, 対称) 差分 + 非適応的な大域線形ソルブが、
     以前の upwind (1次精度) + 適応刻み幅BDFでの27%誤差を改善するか。
  2. rloopを後で足したときに使う「全軌道を1つの連立方程式として持つ」
     という枠組みの基盤を作ること (rloopが入ると非線形になり、Newton
     反復が本当に意味を持つようになる)。

使い方:
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
    """整数offsets (格子点からの相対位置) を使った1階微分の有限差分係数を、
    Fornberg法 (未定係数法) で厳密に求める。

    sum_k c_k * f(x+offsets_k*h) = h*f'(x) + O(h^len(offsets)) となる c を返す
    (m=len(offsets)点のステンシルで、次数 m-1 精度)。
    """
    offsets = np.asarray(offsets, dtype=float)
    m = len(offsets)
    M = np.vstack([offsets ** p for p in range(m)])
    rhs = np.zeros(m)
    rhs[1] = 1.0
    return np.linalg.solve(M, rhs)


def build_D1_central(n, h, order=2):
    """1階微分の疎行列を返す。

    order=2: 内部2次精度中心差分(3点)・境界2次精度片側差分
        (frg_discrete/grid_fd.py Grid2D._first_deriv_1d と同じステンシル)。
    order=4: 内部4次精度中心差分(5点)・境界寄り1点は4次精度片側寄りステンシル
        (5点、非対称)・最端点は4次精度片側差分(5点)。
        いずれもFornberg法で係数を導出するので、次数を上げても手打ちの
        係数ミスが入らない。
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
            raise ValueError("order=4 には各方向5点以上の格子が必要です")
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
    """A = -4*I + 2*rho*d/drho + 2*sigma*d/dsigma を (n_rho*n_sigma)^2 の
    疎行列として構築する。flatten順は U.ravel() (rho-major, C順)。"""
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
    リラクゼーション法: t方向にNt+1点、Crank-Nicolsonで離散化した
    ブロック双対角の大域線形系を解く。

    (I - 0.5*dt*A) U^n = (I + 0.5*dt*A) U^{n-1},  n=1..Nt
    U^0 = U0 (UV境界条件、固定)

    Aとdtがtに依らない(tree-only)ので、LHS行列は全ステップ共通。
    LU分解を1回だけ行い、ブロック消去 (= このブロック双対角の大域系を
    厳密に解くのと数学的に同値) で全ステップを解く。
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
    """特性曲線 rho(0)=rho(t_end)*exp(2*t_end) で厳密解を評価する。"""
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
    p.add_argument("--n-rho", type=int, default=41, help="rho方向の格子点数")
    p.add_argument("--n-sigma", type=int, default=41, help="sigma方向の格子点数")
    p.add_argument("--n-t", type=int, default=100, help="時間方向のステップ数")
    p.add_argument("--rho-max", type=float, default=1.75)
    p.add_argument("--sigma-max", type=float, default=1.75)
    p.add_argument("--t0", type=float, default=0.0)
    p.add_argument("--t-end", type=float, default=None)
    p.add_argument("--full-seed", action="store_true",
                    help="tree多項式のみでなく、u_seed (tree+thermal) を初期条件に使う")
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
