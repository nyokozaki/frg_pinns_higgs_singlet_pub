#!/usr/bin/env python3
"""
フル方程式 (rloop+eta込み) を Newton-Krylov によるリラクゼーション法で解く。

relax_tree.py はtree-only (線形) だったので、"まとめて解く"ことはCN
マーチングと数学的に同値だった。rloopが入ると非線形になるので、ここで
初めてリラクゼーション法固有の御利益 (全軌道を1つの自己無撞着解として
Newtonで同時に解くことで、forward shooting特有の"誤差が時間方向に
一方向に伝播・増幅していく"構造そのものを回避できるかもしれないこと)
を試せる。

離散化: 空間は中心差分 (grid_fd.Grid2D, scheme="central")、時間は
Crank-Nicolson。t=0..t_endの全時刻ステップ (U^1..U^Nt、U^0はUV境界条件で
固定) をまとめて1つの非線形連立方程式とみなし、

    R^n(U^1,...,U^Nt) = (U^n - U^{n-1}) - dt/2*(RHS(t^{n-1},U^{n-1}) + RHS(t^n,U^n)) = 0
    (n=1..Nt, U^0 = u_seed)

を scipy.optimize.newton_krylov (行列を陽に組まないJacobian-free
Newton-Krylov法。GMRES的な反復でNewton方向を近似する) で解く。

初期推定値には、
  - tree部分: u_tree_exact (mass parameterの a_UV -> a_UV*exp(-2t) スケーリング則
    による厳密解。線形PDEなので数値ソルブ不要)
  - thermal部分: my_networks.py のPINNが非学習の固定項として使っている粗い
    Debyeモス近似 u_th = 0.2*rho*tau(t)^2 + 0.1*sigma*tau(t)^2
    (tau(t) = tau_uv*exp(-t)、各時刻で正しく再評価)
の和を使う (t=0以外のNewton自由変数の出発点。u_th自体はPINN側でも学習対象
外の固定バイアス項として使われている粗い近似なので、正解を仕込んでいる
ことにはならない)。

使い方:
    python relax_full.py --n-rho 21 --n-sigma 21 --n-t 40
"""

import argparse
import os
import time

import numpy as np
from scipy.optimize import newton_krylov, root

import _pathsetup  # noqa: F401
from perturbation.config_params import t_range, tau_uv, k_IR, t as t_uv_offset
import flow_equation as _flow_equation_mod
from grid_fd import Grid2D
from seed_potential import u_seed, u_tree_exact
from flow_equation import flow_rhs
from coupling_prior import coupling_prior_residual


# my_networks.py の PINN が非学習の固定バイアス項として使っている粗い
# Debye熱質量近似 (係数 kh=0.2, ks=0.1 も同一)。tau(t)=tau_uv*exp(-t) と
# 各時刻で正しく再評価される点が、analytic_tree_solution の
# "tau_UV固定のまま特性曲線輸送" による近似より物理的に妥当。
_KH_THERMAL, _KS_THERMAL = 0.2, 0.1


def thermal_seed_ansatz(t_phys, rho_phys, sigma_phys):
    tau_t = tau_uv * np.exp(-t_phys)
    return _KH_THERMAL * rho_phys * tau_t ** 2 + _KS_THERMAL * sigma_phys * tau_t ** 2


def parse_args():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--n-rho", type=int, default=21, help="rho方向の格子点数")
    p.add_argument("--n-sigma", type=int, default=21, help="sigma方向の格子点数")
    p.add_argument("--n-t", type=int, default=40, help="時間方向のステップ数")
    p.add_argument("--rho-max", type=float, default=1.75)
    p.add_argument("--sigma-max", type=float, default=1.75)
    p.add_argument("--rho-min", type=float, default=0.0,
                   help="rho方向の下限。0より大きくすると sigma=0/rho=0 の各軸を避け、"
                        "M12=2*sqrt(rho*sigma)*u_rhosigma が恒等的に0になる軸上の"
                        "質量固有値縮退 (disc=(M11-M22)^2=0) を回避できる。")
    p.add_argument("--sigma-min", type=float, default=0.0,
                   help="sigma方向の下限。--rho-min 参照。")
    p.add_argument("--t0", type=float, default=0.0)
    p.add_argument("--t-end", type=float, default=None)
    p.add_argument("--f-tol", type=float, default=1e-8)
    p.add_argument("--maxiter", type=int, default=100)
    p.add_argument("--inner-maxiter", type=int, default=30)
    p.add_argument("--method", type=str, default="lgmres")
    p.add_argument("--scheme", type=str, default="cn", choices=["cn", "be"],
                   help="cn: Crank-Nicolson (2次精度), be: backward Euler (1次精度, より頑健)")
    p.add_argument("--spatial-scheme", type=str, default="central", choices=["central", "central4"],
                   help="central: 2次精度中心差分 (grid_fd.py既定)。"
                        "central4: 4次精度中心差分 (5点ステンシル, Fornberg法。各方向5点以上必要)。")
    p.add_argument("--disc-delta", type=float, default=0.0,
                   help="質量固有値判別式 disc=(M11-M22)^2+4*M12^2 の正則化パラメータ (滑らか版)。"
                        "disc_delta>0 で sqrt(disc+disc_delta**2) を使い、"
                        "disc=0 (Higgs/singlet質量の縮退点) の円錐特異点を丸める。")
    p.add_argument("--disc-cut", type=float, default=0.0,
                   help="disc_delta の代わりのハードクリップ正則化。disc_cut>0 で "
                        "sqrt(max(disc,disc_cut**2)) を使う (disc<disc_cut**2 の領域は"
                        "sqrt_discが定数になり、その領域内でのUに対する微分が厳密に0になる)。"
                        "disc_delta と排他 (disc_cut>0 のときはdisc_deltaは無視される)。")
    p.add_argument("--T-raw", type=float, default=None,
                   help="有限温度パラメータ T_RAW [GeV] (perturbation/config_params.py の"
                        "デフォルトは100.0)。指定するとtau_uv = (T_raw/k_IR)*exp(t) を"
                        "再計算し、flow_equation.tau_uv とこのスクリプト自身のtau_uv "
                        "(thermal_seed_ansatz用) の両方を上書きする。省略時はconfig_params.py"
                        "のデフォルト値のまま。")
    p.add_argument("--mass-floor", type=float, default=0.0,
                   help="Higgs/singlet/Goldstoneの規格化質量二乗 1+m^2 (mG2,m1_sq,m2_sq) に対する"
                        "ハードクリップのfloor値。物理的には1+m^2がtachyonicになるのは非現実的な"
                        "ので、Newton中間解が暴れて一時的にtachyonicへ迷い込んだ場合に "
                        "1/sqrt(1+m^2) が急峻に発散するのを防ぐ安全弁 (flow_equation.py "
                        "_rloop_from_derivs 参照)。0 (デフォルト) は従来通りEPS floor。")
    p.add_argument("--warm-start", type=str, default=None,
                   help="前回の relax_full.py 出力npz (U_all_raw フィールド) を初期推定値として使う"
                        "(継続法/continuation用。同じ格子・n_tである必要がある)。省略時は"
                        "u_tree_exact + thermal_seed_ansatz (閉形式) を初期推定値にする。")
    p.add_argument("--coupling-prior-sign-weight", type=float, default=0.0,
                   help="loss_extensions.py の符号制約を模した forcing 項の重み (0=無効)。"
                        "coupling_prior.coupling_prior_residual 参照。")
    p.add_argument("--coupling-prior-mag-weight", type=float, default=0.0,
                   help="loss_extensions.py の大きさ帯制約を模した forcing 項の重み (0=無効)。")
    p.add_argument("--disable-sign-f-h", action="store_true",
                   help="coupling prior の sign_pen から rho方向のmass-term (F_H) の"
                        "sign hinge penaltyだけを外す (magnitude band側のF_H制約は残す)。")
    p.add_argument("--coupling-prior-anneal-steps", type=int, default=0,
                   help="0: 指定した重みのまま1回だけ解く (収束解はpriorに恒久的にバイアスされる)。"
                        "N>0: N段階に分けて重みを幾何級数的に0まで下げながら逐次warm-start再解"
                        "(継続法)。最終ステップは重み0=純粋なフロー方程式になるので、最終解は"
                        "priorの影響を受けない (priorはNewtonを良い分岐に導くためだけに使われる)。")
    p.add_argument("--out", type=str, default="results/relax_full.npz")
    return p.parse_args()


def main():
    args = parse_args()
    t_end = args.t_end if args.t_end is not None else t_range

    global tau_uv
    if args.T_raw is not None:
        tau_uv = (args.T_raw / k_IR) * np.exp(t_uv_offset)
        _flow_equation_mod.tau_uv = tau_uv
        print(f"T_raw override: T_RAW={args.T_raw} -> tau_uv={tau_uv:.6e} "
              f"(default T_RAW=100.0 -> tau_uv={( (100.0/k_IR)*np.exp(t_uv_offset) ):.6e})")

    grid = Grid2D(rho_max=args.rho_max, sigma_max=args.sigma_max,
                  n_rho=args.n_rho, n_sigma=args.n_sigma,
                  rho_min=args.rho_min, sigma_min=args.sigma_min)
    n_rho, n_sigma = grid.n_rho, grid.n_sigma
    N = n_rho * n_sigma

    t_grid = np.linspace(args.t0, t_end, args.n_t + 1)
    dt = t_grid[1] - t_grid[0]

    U0 = u_seed(args.t0, grid.RHO, grid.SIGMA)
    print(f"Grid: {n_rho}x{n_sigma} spatial, {args.n_t} time steps (t: {args.t0} -> {t_end}), "
          f"disc_delta={args.disc_delta}, disc_cut={args.disc_cut}, mass_floor={args.mass_floor}, "
          f"rho_min={args.rho_min}, sigma_min={args.sigma_min}")

    # --- warm start: tree部分は厳密解 (u_tree_exact)、thermal部分はPINNの
    #     非学習固定バイアス項と同じ粗いDebye近似 (thermal_seed_ansatz)。
    #     どちらも数値線形ソルブ不要、閉形式で全時刻を即座に評価できる。
    U_tree_all = np.stack([
        u_tree_exact(tn, grid.RHO, grid.SIGMA) + thermal_seed_ansatz(tn, grid.RHO, grid.SIGMA)
        for tn in t_grid
    ])

    if args.warm_start is not None:
        print(f"Loading warm start from {args.warm_start} (U_all_raw)...")
        prev = np.load(args.warm_start)
        U_init = prev["U_all_raw"].reshape(args.n_t, n_rho, n_sigma).copy()
    else:
        print("Building warm start from tree_exact + thermal_seed_ansatz (closed form)...")
        U_init = U_tree_all[1:].copy()

    def make_rhs_at(weight_sign, weight_mag):
        def rhs_at(n, U):
            # n: time index into t_grid. U^0 は固定 (キャッシュ不要、呼ばれるのは常に同じ値)。
            rhs = flow_rhs(t_grid[n], U, grid, scheme=args.spatial_scheme,
                           disc_delta=args.disc_delta, disc_cut=args.disc_cut,
                           mass_floor=args.mass_floor)
            if weight_sign or weight_mag:
                rhs = rhs + coupling_prior_residual(
                    t_grid[n], U, grid,
                    weight_sign=weight_sign, weight_mag=weight_mag,
                    disable_sign_F_H=args.disable_sign_f_h,
                )
            return rhs
        return rhs_at

    def make_residual(weight_sign, weight_mag):
        rhs_at = make_rhs_at(weight_sign, weight_mag)

        def residual(U_free):
            U_all = U_free.reshape(args.n_t, n_rho, n_sigma)
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

    # --- coupling-prior weight schedule ---------------------------------
    # anneal_steps<=0: 指定した重みのまま1回だけ解く (収束解はpriorに恒久的に
    #   バイアスされる。診断/実験用)。
    # anneal_steps=N>0: N段階で重みを幾何級数的に0まで落としながら逐次
    #   warm-start再解 (継続法)。最終段は重み0=純粋なフロー方程式なので、
    #   最終的に保存される解はpriorの影響を受けない。
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

    U_all = U_sol_flat.reshape(args.n_t, n_rho, n_sigma)
    U_final = U_all[-1]
    U_final_tree_only = U_tree_all[-1]

    origin = U_final[0, 0]
    U_final_norm = U_final - origin
    print(f"U(t_end) [Newton relaxation, origin-normalized] range: "
          f"[{U_final_norm.min():.6e}, {U_final_norm.max():.6e}]")
    print(f"U(rho_max, sigma=0, t_end) = {U_final_norm[-1, 0]:.6f}")
    print(f"(for reference, tree_exact+thermal_seed_ansatz warm start gave "
          f"{(U_final_tree_only - U_final_tree_only[0, 0])[-1, 0]:.6f} at that point)")

    out_dir = os.path.dirname(args.out)
    if out_dir:
        os.makedirs(out_dir, exist_ok=True)
    np.savez(
        args.out,
        rho=grid.rho, sigma=grid.sigma, t=t_grid,
        U_final=U_final_norm, U_all=U_all - origin,
        U_all_raw=U_all,  # 継続法 (--warm-start) 用。原点シフトしていない生の解。
        disc_delta=args.disc_delta, disc_cut=args.disc_cut, mass_floor=args.mass_floor,
        rho_min=args.rho_min, sigma_min=args.sigma_min,
        T_raw=(args.T_raw if args.T_raw is not None else 100.0),
        success=success, message=message,
        final_residual_norm=np.linalg.norm(r_final), elapsed=elapsed,
    )
    print(f"Saved to {args.out}")


if __name__ == "__main__":
    main()
