#!/usr/bin/env python
"""
draft Fig. `pinn_vacuum_line`: effective potential along the *path-deformed O(3)
bounce trajectory* connecting the two axis vacua, instead of the straight (h,s)
line used previously.

The deformed paths (PINN and finite-T reference) and the stitched NN+ring 2D grid
are read from ``plots_pinn/bounce_TXX.npz`` produced by ``bounce_action.py`` -- this
script does no tunnelling itself and needs neither cosmoTransitions nor the trained
2D models.  Re-run ``bounce_action.py --T <T>`` first whenever the 2D data changes,
then re-run this.

Usage:
    python plot_vacuum_line_bounce.py --T 100
    python plot_vacuum_line_bounce.py --npz plots_pinn/bounce_T100.npz --out plots_pinn/pinn_vacuum_line_T100.png
"""
import argparse
import os
import numpy as np


def _patch_config(T_raw):
    import perturbation.config_params as cp
    cp.T_RAW = float(T_raw)
    cp.tau_uv = (cp.T_RAW / cp.k_IR) * np.exp(cp.t)
    return cp


def _arclen_frac(path):
    d = np.sqrt(np.sum(np.diff(path, axis=0) ** 2, axis=1))
    s = np.concatenate([[0.0], np.cumsum(d)])
    return s / s[-1]


def _resample_arclen(path, n_out=600):
    """Uniform arc-length resampling.  CosmoTransitions clusters points at the
    bubble wall and returns almost none near either vacuum, so the quadratic
    flattening of V at a minimum is otherwise not sampled and the plotted curve
    looks like it still has a slope at the endpoint."""
    d = np.sqrt(np.sum(np.diff(path, axis=0) ** 2, axis=1))
    s = np.concatenate([[0.0], np.cumsum(d)])
    sn = np.linspace(0.0, s[-1], n_out)
    return np.column_stack([np.interp(sn, s, path[:, 0]),
                            np.interp(sn, s, path[:, 1])]), sn / s[-1]


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--T", type=float, default=100.0)
    ap.add_argument("--t-val", type=float, default=-2.0)
    ap.add_argument("--npz", default=None)
    ap.add_argument("--out", default=None)
    args = ap.parse_args()

    npz = args.npz or f"plots_pinn/bounce_T{args.T:.0f}.npz"
    out = args.out or f"plots_pinn/pinn_vacuum_line_T{args.T:.0f}.png"

    cp = _patch_config(args.T)
    k_IR = float(cp.k_IR)

    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    from scipy.interpolate import RectBivariateSpline
    import thermal_np as thermal

    d = np.load(npz, allow_pickle=True)
    rho_ax, sig_ax = d["rho_ax"], d["sig_ax"]
    U_FULL = d["U_FULL"]                       # stitched NN + gauge/top ring, u(rho,sigma)
    spl = RectBivariateSpline(rho_ax, sig_ax, U_FULL, kx=3, ky=3, s=0)


    def _rs(hs):
        h, s = hs[:, 0], hs[:, 1]
        return (np.clip(h * h / (2.0 * k_IR ** 2), rho_ax[0], rho_ax[-1]),
                np.clip(s * s / (2.0 * k_IR ** 2), sig_ax[0], sig_ax[-1]))

    def u_nn(hs):
        r, s = _rs(hs)
        return spl.ev(r, s)

    def u_ref(hs):
        r, s = _rs(hs)
        return thermal.u_seed_finiteT(np.full_like(r, args.t_val), r, s)

    from scipy.optimize import minimize_scalar

    def _refine_axis_vac(Vf, vac):
        """Sharpen an argmin-located axis vacuum by a 1D minimization along its
        own axis, so dV/dfield is genuinely ~0 there and the plotted curve is
        flat at the endpoint (u is even in h and in s, so the minima sit on the
        axes)."""
        h0, s0 = vac
        if h0 >= s0:                                   # EW-type: minimize along h
            r = minimize_scalar(lambda h: float(Vf(np.array([[h, 0.0]]))[0]),
                                bounds=(0.3 * h0, 1.6 * h0 + 1.0), method="bounded")
            return np.array([r.x, 0.0])
        r = minimize_scalar(lambda s: float(Vf(np.array([[0.0, s]]))[0]),
                            bounds=(0.3 * s0, 1.6 * s0 + 1.0), method="bounded")
        return np.array([0.0, r.x])

    # Each deformed bounce path is closed onto *its own* method's axis vacua
    # (NN path -> NN minima, reference path -> reference minima) and oriented
    # xi = 0 : singlet vacuum (0, s_vs)  ->  xi = 1 : EW vacuum (h_v, 0).
    def _prep(path_key, vac_true, vac_false, Vf):
        vac_true = _refine_axis_vac(Vf, vac_true)
        vac_false = _refine_axis_vac(Vf, vac_false)
        p = d[path_key].astype(float)
        # order so p[0] is near EW (true), p[-1] near singlet (false)
        if np.linalg.norm(p[0] - vac_true) > np.linalg.norm(p[-1] - vac_true):
            p = p[::-1]
        p = np.vstack([vac_true, p, vac_false])         # close onto the exact minima
        p = p[::-1]                                     # xi=0 -> singlet, xi=1 -> EW
        return _resample_arclen(p)

    p_nn, xi_nn = _prep("PINN_path",
                        d["PINN_true_pt"].astype(float),
                        d["PINN_false_pt"].astype(float), u_nn)
    p_ref, xi_ref = _prep("reference_path",
                          d["reference_true_pt"].astype(float),
                          d["reference_false_pt"].astype(float), u_ref)

    fig, ax = plt.subplots(figsize=(7, 5))
    # each curve is drawn along its OWN potential's path-deformed bounce trajectory
    ax.plot(xi_nn, u_nn(p_nn), "r--", lw=1.7, label="NN+ring")
    ax.plot(xi_ref, u_ref(p_ref), color="tab:green", ls=":", lw=1.7,
            label="finite T")

    ax.set_xlabel(r"bounce-path fraction $\xi$   "
                  r"[$\xi{=}0:(0,s_{vs})\ \to\ \xi{=}1:(h_v,0)$]")
    ax.set_ylabel("u")
    ax.set_title(f"Potential along the path-deformed bounce trajectory, "
                 f"t={args.t_val:.2f}")
    ax.grid(alpha=0.3)
    ax.legend()
    fig.tight_layout()
    os.makedirs(os.path.dirname(out), exist_ok=True)
    fig.savefig(out, dpi=150, bbox_inches="tight")
    print("saved", out)

    # quick barrier-height report
    for nm, xi, uu in (("NN+ring", xi_nn, u_nn(p_nn)),
                       ("finite T", xi_ref, u_ref(p_ref))):
        print(f"  {nm:9s} barrier: u_max - u(false) = {uu.max() - uu[0]:.4f}  "
              f"(u_false={uu[0]:.4f}, u_true={uu[-1]:.4f}, u_max={uu.max():.4f})")


if __name__ == "__main__":
    main()
