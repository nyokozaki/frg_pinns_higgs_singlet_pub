#!/usr/bin/env python3
"""
フル方程式 (rloop+eta込み, Higgs-only 1次元rho版) を Newton-Krylov による
リラクゼーション法で解く。

singletが存在しないため、質量固有値の混合 (M11,M22,M12,disc,sqrt_disc) は
構造的に発生しない (flow_equation_higgs_only.py 参照)。frg_discrete4_main
(singlet版) の README.md に記録されている「Higgs-singlet質量固有値のほぼ
縮退点での sqrt(disc) 分岐点非平滑性による収束の頭打ち」がこの縮約でも
再現するかどうかを直接検証するのが目的 (再現しなければ、あの非平滑性が
本当に主因だったことの傍証になる)。

離散化: 空間は中心差分 (grid_fd_higgs_only.Grid1D, scheme="central")、時間は
Crank-Nicolson。t=0..t_endの全時刻ステップ (U^1..U^Nt、U^0はUV境界条件で
固定) をまとめて1つの非線形連立方程式とみなし、scipy.optimize.newton_krylov
で解く。

使い方:
    python relax_full.py --n-rho 21 --n-t 40
"""

import argparse
import os
import time

import numpy as np
from scipy.optimize import newton_krylov

import _pathsetup  # noqa: F401
from perturbation.config_params import t_range, tau_uv, k_IR, t as t_uv_offset
import flow_equation_higgs_only as _flow_equation_mod
import cw_thermal
from grid_fd_higgs_only import Grid1D
from seed_potential_higgs_only import u_seed, u_tree_exact
from flow_equation_higgs_only import flow_rhs
from coupling_prior import coupling_prior_residual


# my_networks.py の PINN が非学習の固定バイアス項として使っている粗い
# Debye熱質量近似 (係数 kh=0.2 も同一)。tau(t)=tau_uv*exp(-t) と各時刻で
# 正しく再評価される点が、analytic_tree_solution の "tau_UV固定のまま
# 特性曲線輸送" による近似より物理的に妥当。
_KH_THERMAL = 0.2


def thermal_seed_ansatz(t_phys, rho_phys):
    tau_t = tau_uv * np.exp(-t_phys)
    return _KH_THERMAL * rho_phys * tau_t ** 2


def parse_args():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--n-rho", type=int, default=21, help="rho方向の格子点数")
    p.add_argument("--n-t", type=int, default=40, help="時間方向のステップ数")
    p.add_argument("--rho-max", type=float, default=1.75)
    p.add_argument("--rho-min", type=float, default=0.0,
                   help="rho方向の下限。singlet除去後はdisc縮退そのものが存在しないため、"
                        "singlet版と違い分岐点回避の意味は持たない(rho=0軸自体を避けたい"
                        "場合のみ使う)。")
    p.add_argument("--t0", type=float, default=0.0)
    p.add_argument("--t-end", type=float, default=None)
    p.add_argument("--f-tol", type=float, default=1e-8)
    p.add_argument("--maxiter", type=int, default=100)
    p.add_argument("--inner-maxiter", type=int, default=30)
    p.add_argument("--method", type=str, default="lgmres")
    p.add_argument("--scheme", type=str, default="cn", choices=["cn", "be"],
                   help="cn: Crank-Nicolson (2次精度), be: backward Euler (1次精度, より頑健)")
    p.add_argument("--spatial-scheme", type=str, default="central", choices=["central", "central4"],
                   help="central: 2次精度中心差分。central4: 4次精度中心差分 (5点以上必要)。")
    p.add_argument("--T-raw", type=float, default=None,
                   help="有限温度パラメータ T_RAW [GeV] (perturbation/config_params.py の"
                        "デフォルトは100.0)。指定するとtau_uv = (T_raw/k_IR)*exp(t) を"
                        "再計算し、flow_equation_higgs_only.tau_uv とこのスクリプト自身の"
                        "tau_uv (thermal_seed_ansatz用) の両方を上書きする。")
    p.add_argument("--mass-floor", type=float, default=0.0,
                   help="Higgs/Goldstoneの規格化質量二乗 1+m^2 (mG2,mH2) に対するハード"
                        "クリップのfloor値。0 (デフォルト) は従来通りEPS floor。")
    p.add_argument("--warm-start", type=str, default=None,
                   help="前回の relax_full.py 出力npz (U_all_raw フィールド) を初期推定値として使う"
                        "(継続法/continuation用。同じ格子・n_tである必要がある)。")
    p.add_argument("--coupling-prior-sign-weight", type=float, default=0.0,
                   help="coupling_prior.coupling_prior_residual の符号制約forcingの重み (0=無効)。")
    p.add_argument("--coupling-prior-mag-weight", type=float, default=0.0,
                   help="大きさ帯制約を模した forcing 項の重み (0=無効)。")
    p.add_argument("--coupling-prior-c-lower", type=float, default=0.1,
                   help="coupling_prior_residual の質量項magnitude帯の下限係数 "
                        "(F_H_NN >= c_lower*|F_H_target| を要求)。PINN側hyperparams.c_mag_lower"
                        "に相当。coupling-prior-mag-weight=0のときは無効。")
    p.add_argument("--coupling-prior-c-upper", type=float, default=3.0,
                   help="coupling_prior_residual の質量項magnitude帯の上限係数。"
                        "PINN側hyperparams.c_mag_upperに相当。")
    p.add_argument("--coupling-prior-rho-cw-cut", type=float, default=0.6,
                   help="coupling_prior_residual の低rho/高rho分割点。rho<この値では"
                        "質量項(F_H)のみ、rho>この値ではquartic(D_H)のRGE-runターゲット"
                        "帯のみを制約する。PINN側hyperparams.rho_cw_cut(=0.6)に相当。")
    p.add_argument("--disable-sign-f-h", action="store_true",
                   help="coupling prior の sign_pen から rho方向のmass-term (F_H) の"
                        "sign hinge penaltyだけを外す。")
    p.add_argument("--coupling-prior-anneal-steps", type=int, default=0,
                   help="0: 指定した重みのまま1回だけ解く。N>0: N段階に分けて重みを幾何級数的に"
                        "0まで下げながら逐次warm-start再解 (継続法)。")
    p.add_argument("--out", type=str, default="results/relax_full.npz")
    return p.parse_args()


def main():
    args = parse_args()
    t_end = args.t_end if args.t_end is not None else t_range

    global tau_uv
    if args.T_raw is not None:
        tau_uv = (args.T_raw / k_IR) * np.exp(t_uv_offset)
        _flow_equation_mod.tau_uv = tau_uv
        # coupling_prior.py's del_aH thermal target reads cw_thermal.u_thermal_finiteT,
        # which caches its own copy of tau_uv (see cw_thermal.set_T_raw's docstring) —
        # must be kept in sync or the coupling-prior forcing silently uses T_RAW=100
        # regardless of --T-raw.
        cw_thermal.set_T_raw(args.T_raw)
        print(f"T_raw override: T_RAW={args.T_raw} -> tau_uv={tau_uv:.6e} "
              f"(default T_RAW=100.0 -> tau_uv={( (100.0/k_IR)*np.exp(t_uv_offset) ):.6e})")

    grid = Grid1D(rho_max=args.rho_max, n_rho=args.n_rho, rho_min=args.rho_min)
    n_rho = grid.n_rho

    t_grid = np.linspace(args.t0, t_end, args.n_t + 1)
    dt = t_grid[1] - t_grid[0]

    U0 = u_seed(args.t0, grid.RHO)
    print(f"Grid: {n_rho} spatial (rho), {args.n_t} time steps (t: {args.t0} -> {t_end}), "
          f"mass_floor={args.mass_floor}, rho_min={args.rho_min}")

    # --- warm start: tree部分は厳密解 (u_tree_exact)、thermal部分はPINNの
    #     非学習固定バイアス項と同じ粗いDebye近似 (thermal_seed_ansatz)。
    U_tree_all = np.stack([
        u_tree_exact(tn, grid.RHO) + thermal_seed_ansatz(tn, grid.RHO)
        for tn in t_grid
    ])

    if args.warm_start is not None:
        print(f"Loading warm start from {args.warm_start} (U_all_raw)...")
        prev = np.load(args.warm_start)
        U_init = prev["U_all_raw"].reshape(args.n_t, n_rho).copy()
    else:
        print("Building warm start from tree_exact + thermal_seed_ansatz (closed form)...")
        U_init = U_tree_all[1:].copy()

    def make_rhs_at(weight_sign, weight_mag):
        def rhs_at(n, U):
            rhs = flow_rhs(t_grid[n], U, grid, scheme=args.spatial_scheme, mass_floor=args.mass_floor)
            if weight_sign or weight_mag:
                rhs = rhs + coupling_prior_residual(
                    t_grid[n], U, grid,
                    weight_sign=weight_sign, weight_mag=weight_mag,
                    c_lower=args.coupling_prior_c_lower, c_upper=args.coupling_prior_c_upper,
                    rho_cw_cut=args.coupling_prior_rho_cw_cut,
                    disable_sign_F_H=args.disable_sign_f_h,
                )
            return rhs
        return rhs_at

    def make_residual(weight_sign, weight_mag):
        rhs_at = make_rhs_at(weight_sign, weight_mag)

        def residual(U_free):
            U_all = U_free.reshape(args.n_t, n_rho)
            R = np.empty_like(U_all)
            U_prev = U0
            rhs_prev = rhs_at(0, U_prev)
            for n in range(args.n_t):
                U_curr = U_all[n]
                rhs_curr = rhs_at(n + 1, U_curr)
                if args.scheme == "cn":
                    R[n] = (U_curr - U_prev) - 0.5 * dt * (rhs_prev + rhs_curr)
                else:  # backward Euler
                    R[n] = (U_curr - U_prev) - dt * rhs_curr
                U_prev = U_curr
                rhs_prev = rhs_curr
            return R.ravel()
        return residual

    def run_newton(U_init_flat, weight_sign, weight_mag):
        residual = make_residual(weight_sign, weight_mag)
        r0 = residual(U_init_flat)
        print(f"  [prior weights sign={weight_sign:.4g} mag={weight_mag:.4g}] "
              f"initial residual norm: {np.linalg.norm(r0):.6e} (max abs {np.max(np.abs(r0)):.6e})")
        start = time.time()
        try:
            U_sol_flat = newton_krylov(
                residual, U_init_flat,
                method=args.method,
                f_tol=args.f_tol,
                maxiter=args.maxiter,
                inner_maxiter=args.inner_maxiter,
                verbose=True,
            )
            success = True
            message = "converged"
        except Exception as e:
            print(f"[WARNING] newton_krylov did not converge cleanly: {e}")
            if e.args and hasattr(e.args[0], "reshape"):
                U_sol_flat = e.args[0]
            else:
                U_sol_flat = U_init_flat
            success = False
            message = str(e)
        elapsed = time.time() - start
        r_final = residual(U_sol_flat)
        print(f"  [prior weights sign={weight_sign:.4g} mag={weight_mag:.4g}] "
              f"final residual norm: {np.linalg.norm(r_final):.6e} "
              f"(max abs {np.max(np.abs(r_final)):.6e}), elapsed {elapsed:.1f}s")
        return U_sol_flat, success, message, elapsed, r_final

    sign_w0 = args.coupling_prior_sign_weight
    mag_w0 = args.coupling_prior_mag_weight
    if args.coupling_prior_anneal_steps <= 0:
        schedule = [(sign_w0, mag_w0)]
    else:
        n_steps = args.coupling_prior_anneal_steps
        decay = 0.3
        schedule = [
            (0.0, 0.0) if i == n_steps - 1 else (sign_w0 * decay ** i, mag_w0 * decay ** i)
            for i in range(n_steps)
        ]
        print(f"Coupling-prior annealing schedule (sign, mag): {schedule}")

    U_curr_flat = U_init.ravel()
    for step_i, (ws, wm) in enumerate(schedule):
        if len(schedule) > 1:
            print(f"--- continuation step {step_i + 1}/{len(schedule)} ---")
        U_curr_flat, success, message, elapsed, r_final = run_newton(U_curr_flat, ws, wm)
    U_sol_flat = U_curr_flat

    U_all = U_sol_flat.reshape(args.n_t, n_rho)
    U_final = U_all[-1]
    U_final_tree_only = U_tree_all[-1]

    origin = U_final[0]
    U_final_norm = U_final - origin
    print(f"U(t_end) [Newton relaxation, origin-normalized] range: "
          f"[{U_final_norm.min():.6e}, {U_final_norm.max():.6e}]")
    print(f"U(rho_max, t_end) = {U_final_norm[-1]:.6f}")
    print(f"(for reference, tree_exact+thermal_seed_ansatz warm start gave "
          f"{(U_final_tree_only - U_final_tree_only[0])[-1]:.6f} at that point)")

    out_dir = os.path.dirname(args.out)
    if out_dir:
        os.makedirs(out_dir, exist_ok=True)
    np.savez(
        args.out,
        rho=grid.rho, t=t_grid,
        U_final=U_final_norm, U_all=U_all - origin,
        U_all_raw=U_all,  # 継続法 (--warm-start) 用。原点シフトしていない生の解。
        mass_floor=args.mass_floor, rho_min=args.rho_min,
        T_raw=(args.T_raw if args.T_raw is not None else 100.0),
        success=success, message=message,
        final_residual_norm=np.linalg.norm(r_final), elapsed=elapsed,
    )
    print(f"Saved to {args.out}")


if __name__ == "__main__":
    main()
