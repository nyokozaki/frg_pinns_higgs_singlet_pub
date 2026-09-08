#!/usr/bin/env python3
"""
Solve the full equation (with rloop+eta, Higgs-only 1D rho version) with a
Newton-Krylov relaxation method.

Since there is no singlet, mass-eigenvalue mixing (M11,M22,M12,disc,sqrt_disc)
does not arise structurally (see flow_equation_higgs_only.py). The purpose is to
test directly whether the "convergence plateau due to the non-smoothness of the
sqrt(disc) branch point near the near-degenerate Higgs-singlet mass eigenvalues",
recorded in frg_discrete4_main (singlet version) README.md, is reproduced in this
reduction too (if it is not, that is circumstantial evidence that that
non-smoothness really was the main cause).

Discretization: space uses a central difference (grid_fd_higgs_only.Grid1D,
scheme="central"), time uses Crank-Nicolson. All time steps t=0..t_end
(U^1..U^Nt; U^0 fixed by the UV boundary condition) are treated together as one
nonlinear system and solved with scipy.optimize.newton_krylov.

Usage:
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


# The crude Debye thermal-mass approximation that the my_networks.py PINN uses
# as a non-trainable fixed bias term (the coefficient kh=0.2 is identical too).
# Re-evaluating tau(t)=tau_uv*exp(-t) correctly at each time is more physically
# sound than analytic_tree_solution's "characteristic transport with tau_UV held
# fixed" approximation.
_KH_THERMAL = 0.2


def thermal_seed_ansatz(t_phys, rho_phys):
    tau_t = tau_uv * np.exp(-t_phys)
    return _KH_THERMAL * rho_phys * tau_t ** 2


def parse_args():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--n-rho", type=int, default=21, help="number of grid points in rho")
    p.add_argument("--n-t", type=int, default=40, help="number of time steps")
    p.add_argument("--rho-max", type=float, default=1.75)
    p.add_argument("--rho-min", type=float, default=0.0,
                   help="lower bound in rho. After removing the singlet there is no "
                        "disc degeneracy at all, so unlike the singlet version this "
                        "has no branch-point-avoidance meaning (use only if you want "
                        "to avoid the rho=0 axis itself).")
    p.add_argument("--t0", type=float, default=0.0)
    p.add_argument("--t-end", type=float, default=None)
    p.add_argument("--f-tol", type=float, default=1e-8)
    p.add_argument("--maxiter", type=int, default=100)
    p.add_argument("--inner-maxiter", type=int, default=30)
    p.add_argument("--method", type=str, default="lgmres")
    p.add_argument("--scheme", type=str, default="cn", choices=["cn", "be"],
                   help="cn: Crank-Nicolson (2nd order), be: backward Euler (1st order, more robust)")
    p.add_argument("--spatial-scheme", type=str, default="central", choices=["central", "central4"],
                   help="central: 2nd-order central difference. central4: 4th-order central difference (needs >= 5 points).")
    p.add_argument("--T-raw", type=float, default=None,
                   help="finite-temperature parameter T_RAW [GeV] (the default in "
                        "perturbation/config_params.py is 100.0). When given, tau_uv = "
                        "(T_raw/k_IR)*exp(t) is recomputed and both "
                        "flow_equation_higgs_only.tau_uv and this script's own tau_uv "
                        "(for thermal_seed_ansatz) are overwritten.")
    p.add_argument("--mass-floor", type=float, default=0.0,
                   help="floor value for the hard clip on the normalized mass-squared "
                        "1+m^2 (mG2,mH2) of Higgs/Goldstone. 0 (default) keeps the "
                        "previous EPS floor.")
    p.add_argument("--warm-start", type=str, default=None,
                   help="use a previous relax_full.py output npz (U_all_raw field) as "
                        "the initial guess (for continuation; must be the same grid and "
                        "n_t).")
    p.add_argument("--coupling-prior-sign-weight", type=float, default=0.0,
                   help="weight of the sign-constraint forcing of coupling_prior.coupling_prior_residual (0=disabled).")
    p.add_argument("--coupling-prior-mag-weight", type=float, default=0.0,
                   help="weight of the forcing term emulating the magnitude-band constraint (0=disabled).")
    p.add_argument("--coupling-prior-c-lower", type=float, default=0.1,
                   help="lower coefficient of the mass-term magnitude band of "
                        "coupling_prior_residual (requires F_H_NN >= c_lower*|F_H_target|). "
                        "Corresponds to hyperparams.c_mag_lower on the PINN side. "
                        "Disabled when coupling-prior-mag-weight=0.")
    p.add_argument("--coupling-prior-c-upper", type=float, default=3.0,
                   help="upper coefficient of the mass-term magnitude band of "
                        "coupling_prior_residual. Corresponds to hyperparams.c_mag_upper "
                        "on the PINN side.")
    p.add_argument("--coupling-prior-rho-cw-cut", type=float, default=0.6,
                   help="low-rho/high-rho split point of coupling_prior_residual. For "
                        "rho below this value only the mass term (F_H) is constrained, "
                        "for rho above it only the RGE-run target band of the quartic "
                        "(D_H). Corresponds to hyperparams.rho_cw_cut(=0.6) on the PINN "
                        "side.")
    p.add_argument("--disable-sign-f-h", action="store_true",
                   help="remove only the sign hinge penalty of the rho-direction "
                        "mass term (F_H) from the coupling prior's sign_pen.")
    p.add_argument("--coupling-prior-anneal-steps", type=int, default=0,
                   help="0: solve once with the given weight. N>0: re-solve with "
                        "warm start in N stages, decaying the weight geometrically to 0 "
                        "(continuation method).")
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

    # --- warm start: the tree part is the exact solution (u_tree_exact), the
    #     thermal part is the same crude Debye approximation as the PINN's
    #     non-trainable fixed bias term (thermal_seed_ansatz).
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
        U_all_raw=U_all,  # for continuation (--warm-start). The raw solution, not origin-shifted.
        mass_floor=args.mass_floor, rho_min=args.rho_min,
        T_raw=(args.T_raw if args.T_raw is not None else 100.0),
        success=success, message=message,
        final_residual_norm=np.linalg.norm(r_final), elapsed=elapsed,
    )
    print(f"Saved to {args.out}")


if __name__ == "__main__":
    main()
