#!/usr/bin/env python3
"""
Solve the full equation (with rloop+eta) by a Newton-Krylov relaxation method.

relax_tree.py was tree-only (linear), so "solving all at once" was
mathematically equivalent to CN marching. Once rloop is included the system
becomes nonlinear, and only here can the benefit specific to a relaxation
method be tested: solving the whole trajectory simultaneously as one
self-consistent solution with Newton may avoid the structure of forward
shooting in which "error propagates and amplifies one-directionally in time".

Discretization: central difference in space (grid_fd.Grid2D, scheme="central"),
Crank-Nicolson in time. All time steps from t=0..t_end (U^1..U^Nt; U^0 is fixed
by the UV boundary condition) are treated as one nonlinear system,

    R^n(U^1,...,U^Nt) = (U^n - U^{n-1}) - dt/2*(RHS(t^{n-1},U^{n-1}) + RHS(t^n,U^n)) = 0
    (n=1..Nt, U^0 = u_seed)

and solved with scipy.optimize.newton_krylov (a Jacobian-free Newton-Krylov
method that does not form the matrix explicitly and approximates the Newton
direction with GMRES-like iterations).

The initial guess uses the sum of
  - tree part: u_tree_exact (exact solution from the mass-parameter scaling law
    a_UV -> a_UV*exp(-2t); a linear PDE, so no numerical solve needed)
  - thermal part: the crude Debye-mass approximation that the PINN in
    my_networks.py uses as a fixed, non-trainable term,
    u_th = 0.2*rho*tau(t)^2 + 0.1*sigma*tau(t)^2
    (tau(t) = tau_uv*exp(-t), correctly re-evaluated at each time)
as the starting point for the Newton free variables away from t=0. Since u_th
itself is only a crude approximation used as a fixed bias term (not a training
target) on the PINN side too, this does not amount to seeding in the answer.

Usage:
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


# The crude Debye thermal-mass approximation that the PINN in my_networks.py
# uses as a fixed, non-trainable bias term (the coefficients kh=0.2, ks=0.1 are
# the same). Re-evaluating tau(t)=tau_uv*exp(-t) correctly at each time is more
# physical than the "characteristic transport with tau_UV held fixed"
# approximation of analytic_tree_solution.
_KH_THERMAL, _KS_THERMAL = 0.2, 0.1


def thermal_seed_ansatz(t_phys, rho_phys, sigma_phys):
    tau_t = tau_uv * np.exp(-t_phys)
    return _KH_THERMAL * rho_phys * tau_t ** 2 + _KS_THERMAL * sigma_phys * tau_t ** 2


def parse_args():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--n-rho", type=int, default=21, help="number of grid points in rho")
    p.add_argument("--n-sigma", type=int, default=21, help="number of grid points in sigma")
    p.add_argument("--n-t", type=int, default=40, help="number of time steps")
    p.add_argument("--rho-max", type=float, default=1.75)
    p.add_argument("--sigma-max", type=float, default=1.75)
    p.add_argument("--rho-min", type=float, default=0.0,
                   help="lower bound in rho. Setting it above 0 keeps away from the sigma=0/rho=0 "
                        "axes and avoids the mass-eigenvalue degeneracy (disc=(M11-M22)^2=0) on the "
                        "axis where M12=2*sqrt(rho*sigma)*u_rhosigma vanishes identically.")
    p.add_argument("--sigma-min", type=float, default=0.0,
                   help="lower bound in sigma. See --rho-min.")
    p.add_argument("--t0", type=float, default=0.0)
    p.add_argument("--t-end", type=float, default=None)
    p.add_argument("--f-tol", type=float, default=1e-8)
    p.add_argument("--maxiter", type=int, default=100)
    p.add_argument("--inner-maxiter", type=int, default=30)
    p.add_argument("--method", type=str, default="lgmres")
    p.add_argument("--scheme", type=str, default="cn", choices=["cn", "be"],
                   help="cn: Crank-Nicolson (2nd order); be: backward Euler (1st order, more robust)")
    p.add_argument("--spatial-scheme", type=str, default="central", choices=["central", "central4"],
                   help="central: 2nd-order central difference (grid_fd.py default). "
                        "central4: 4th-order central difference (5-point stencil, Fornberg method; needs >=5 points per direction).")
    p.add_argument("--disc-delta", type=float, default=0.0,
                   help="regularization parameter for the mass-eigenvalue discriminant "
                        "disc=(M11-M22)^2+4*M12^2 (smooth version). With disc_delta>0, "
                        "sqrt(disc+disc_delta**2) is used, rounding the conical singularity at "
                        "disc=0 (the Higgs/singlet mass degeneracy point).")
    p.add_argument("--disc-cut", type=float, default=0.0,
                   help="hard-clip regularization instead of disc_delta. With disc_cut>0, "
                        "sqrt(max(disc,disc_cut**2)) is used (in the region disc<disc_cut**2, "
                        "sqrt_disc is constant, so its derivative with respect to U is exactly 0 there). "
                        "Mutually exclusive with disc_delta (when disc_cut>0, disc_delta is ignored).")
    p.add_argument("--T-raw", type=float, default=None,
                   help="finite-temperature parameter T_RAW [GeV] (the default in "
                        "perturbation/config_params.py is 100.0). If given, tau_uv = (T_raw/k_IR)*exp(t) "
                        "is recomputed and both flow_equation.tau_uv and this script's own tau_uv "
                        "(used by thermal_seed_ansatz) are overwritten. If omitted, the default value "
                        "from config_params.py is kept.")
    p.add_argument("--mass-floor", type=float, default=0.0,
                   help="hard-clip floor for the normalized mass-squared 1+m^2 (mG2,m1_sq,m2_sq) of the "
                        "Higgs/singlet/Goldstone modes. Physically 1+m^2 becoming tachyonic is unrealistic, "
                        "so this is a safety valve preventing 1/sqrt(1+m^2) from diverging steeply if a "
                        "Newton iterate blows up and wanders transiently into a tachyonic regime "
                        "(see flow_equation.py _rloop_from_derivs). 0 (default) keeps the previous EPS floor.")
    p.add_argument("--warm-start", type=str, default=None,
                   help="use a previous relax_full.py output npz (the U_all_raw field) as the initial "
                        "guess (for continuation; must have the same grid and n_t). If omitted, the "
                        "initial guess is u_tree_exact + thermal_seed_ansatz (closed form).")
    p.add_argument("--coupling-prior-sign-weight", type=float, default=0.0,
                   help="weight of the forcing term mimicking the sign constraint of loss_extensions.py "
                        "(0 = disabled). See coupling_prior.coupling_prior_residual.")
    p.add_argument("--coupling-prior-mag-weight", type=float, default=0.0,
                   help="weight of the forcing term mimicking the magnitude-band constraint of loss_extensions.py (0 = disabled).")
    p.add_argument("--disable-sign-f-h", action="store_true",
                   help="remove only the sign hinge penalty for the rho-direction mass term (F_H) from "
                        "the coupling prior's sign_pen (the magnitude-band F_H constraint is kept).")
    p.add_argument("--coupling-prior-anneal-steps", type=int, default=0,
                   help="0: solve once at the given weights (the converged solution is permanently biased "
                        "by the prior). N>0: re-solve sequentially with warm start while decaying the "
                        "weights geometrically to 0 in N stages (continuation). The last step has "
                        "weight 0 = the pure flow equation, so the final solution is not affected by "
                        "the prior (the prior is used only to guide Newton to a good branch).")
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

    # --- warm start: the tree part is the exact solution (u_tree_exact); the
    #     thermal part is the same crude Debye approximation as the PINN's
    #     non-trainable fixed bias term (thermal_seed_ansatz). Neither needs a
    #     numerical linear solve; both are evaluated in closed form at all times.
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
            # n: time index into t_grid. U^0 is fixed (no cache needed; always the same value).
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
    # anneal_steps<=0: solve once at the given weights (the converged solution is
    #   permanently biased by the prior; for diagnostics/experiments).
    # anneal_steps=N>0: re-solve sequentially with warm start while decaying the
    #   weights geometrically to 0 in N stages (continuation). The last stage has
    #   weight 0 = the pure flow equation, so the finally saved solution is not
    #   affected by the prior.
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
        U_all_raw=U_all,  # for continuation (--warm-start); the raw solution, not origin-shifted.
        disc_delta=args.disc_delta, disc_cut=args.disc_cut, mass_floor=args.mass_floor,
        rho_min=args.rho_min, sigma_min=args.sigma_min,
        T_raw=(args.T_raw if args.T_raw is not None else 100.0),
        success=success, message=message,
        final_residual_norm=np.linalg.norm(r_final), elapsed=elapsed,
    )
    print(f"Saved to {args.out}")


if __name__ == "__main__":
    main()
