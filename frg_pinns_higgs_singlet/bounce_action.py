"""
O(3) bounce action S_3(T)/T at a fixed temperature from the stitched 2D PINN
effective potential (``data_UV_2D/``).

This does *not* train anything and does *not* scan temperature: it reads an
already-trained 5x5x2 grid of 2D blocks, stitches them into u(rho,sigma) at the
IR end of the flow (t = t_range), adds the external gauge/top ring correction,
maps to the dimensionful potential V(h,s) = k_IR^4 u with canonically normalised
fields (Z_H = Z_S = 1, LPA'), locates the two axis vacua, and computes

    S_3 = 4*pi * int_0^inf r^2 [ (1/2)(dphi/dr)^2 + DeltaV ] dr

both along the straight (h,s) line between the vacua (single-field instanton, an
upper bound on S_3) and along the fully path-deformed trajectory, via
CosmoTransitions.  The same computation on the perturbative Arnold--Espinosa
finite-T reference potential is reported alongside as a sanity cross-check.

The temperature is injected by monkey-patching ``perturbation.config_params``
*before* the physics modules are imported, so nothing on disk is modified and any
concurrently running training process (which imported the module at start-up) is
unaffected.  Pass ``--T`` to match the temperature the 2D grid was trained at
(the ``data2d_T100.zip`` snapshot was trained with the paper benchmark and
T_RAW = 100).

Usage
-----
    python bounce_action.py --T 100
    python bounce_action.py --T 100 --data-dir data_UV_2D --outdir plots_pinn

Caveats (see draft Sec. 6.3): the 2D network is preliminary and not converged to
the accuracy of the 1D slices, the flow is stopped at k_IR = 150 GeV rather than
k -> 0, and the kinetic terms are taken canonical.  Treat the number as an
order-of-magnitude, end-to-end demonstration, not a converged physics result.
"""
import argparse
import os
import numpy as np


def _patch_config(T_raw, device):
    """Set the temperature / device before any physics module is imported."""
    import perturbation.config_params as cp
    cp.T_RAW = float(T_raw)
    cp.tau_uv = (cp.T_RAW / cp.k_IR) * np.exp(cp.t)
    import hyperparams as hp
    hp.device = device
    return cp


def _build_potentials(cp, t_val, n_points, data_dir):
    """Return (Vnn, dVnn, Vref, dVref, meta) with V in GeV^4, fields in GeV."""
    import torch
    from scipy.interpolate import RectBivariateSpline
    import show_UV2D as S
    from thermal_functions import get_u_ring
    import thermal_np as thermal

    k_IR = cp.k_IR
    RHO, SIGMA, _u_tree, U_PRED, _shifts = S.stitched_grid_matched_2d(
        t_val, n_points=n_points
    )
    rho_ax, sig_ax = RHO[:, 0], SIGMA[0, :]

    # external gauge/top ring correction (rho-only; sigma=0 baseline subtracted)
    t_t = S.my_to(np.full_like(rho_ax, t_val))
    with torch.no_grad():
        ring = (
            get_u_ring(t_t, S.my_to(rho_ax))
            - get_u_ring(t_t, S.my_to(np.zeros_like(rho_ax)))
        ).cpu().numpy().flatten()
    U_FULL = U_PRED + ring[:, None]

    spl = RectBivariateSpline(rho_ax, sig_ax, U_FULL, kx=3, ky=3, s=0)

    def _rs(h, s):
        return (
            np.clip(h * h / (2.0 * k_IR**2), rho_ax[0], rho_ax[-1]),
            np.clip(s * s / (2.0 * k_IR**2), sig_ax[0], sig_ax[-1]),
        )

    def Vnn(X):
        X = np.atleast_2d(np.asarray(X, float))
        rho, sig = _rs(X[:, 0], X[:, 1])
        return (k_IR**4 * spl.ev(rho, sig)).reshape(np.shape(X)[:-1])

    def dVnn(X):
        X = np.atleast_2d(np.asarray(X, float))
        h, s = X[:, 0], X[:, 1]
        rho, sig = _rs(h, s)
        dr = k_IR**4 * spl.ev(rho, sig, dx=1, dy=0)
        ds = k_IR**4 * spl.ev(rho, sig, dx=0, dy=1)
        return np.stack([dr * (h / k_IR**2), ds * (s / k_IR**2)],
                        axis=-1).reshape(np.shape(X))

    def Vref(X):
        X = np.atleast_2d(np.asarray(X, float))
        rho, sig = _rs(X[:, 0], X[:, 1])
        u = thermal.u_seed_finiteT(np.full_like(rho, t_val), rho, sig)
        return (k_IR**4 * u).reshape(np.shape(X)[:-1])

    def dVref(X, eps=0.5):
        X = np.atleast_2d(np.asarray(X, float))
        g = np.zeros_like(X)
        for i in range(2):
            d = np.zeros(2)
            d[i] = eps
            g[:, i] = (Vref(X + d) - Vref(X - d)) / (2.0 * eps)
        return g.reshape(np.shape(X))

    meta = dict(rho_ax=rho_ax, sig_ax=sig_ax, U_FULL=U_FULL, k_IR=k_IR)
    return Vnn, dVnn, Vref, dVref, meta


def _axis_vacua(Vf, rho_ax, sig_ax, k_IR, n=400):
    hh = np.linspace(1.0, k_IR * np.sqrt(2.0 * rho_ax[-1]), n)
    ss = np.linspace(1.0, k_IR * np.sqrt(2.0 * sig_ax[-1]), n)
    h_v = hh[np.argmin(Vf(np.stack([hh, np.zeros_like(hh)], -1)))]
    s_v = ss[np.argmin(Vf(np.stack([np.zeros_like(ss), ss], -1)))]
    ew, sing = np.array([h_v, 0.0]), np.array([0.0, s_v])
    if Vf(ew)[0] < Vf(sing)[0]:
        return ew, sing, (h_v, s_v)   # true, false
    return sing, ew, (h_v, s_v)


def _bounce(Vf, dVf, true_pt, false_pt, T):
    """Return dict with straight-line and path-deformed S_3 (and S_3/T)."""
    from scipy.interpolate import CubicSpline
    from cosmoTransitions.tunneling1D import SingleFieldInstanton
    from cosmoTransitions import pathDeformation

    L = float(np.linalg.norm(true_pt - false_pt))
    xi = np.linspace(0.0, 1.0, 400)
    line = false_pt + np.outer(xi, true_pt - false_pt)
    Vl = Vf(line)
    Vl = Vl - Vl[0]
    out = dict(xi=xi, Vl=Vl, L=L, S3_line=None, S3_def=None, path=None)
    if Vl[-1] >= 0.0 or Vl.max() <= 1e-9 * abs(Vl[-1]):
        return out

    cs = CubicSpline(xi * L, Vl)
    try:
        inst = SingleFieldInstanton(
            L, 0.0,
            V=lambda p: cs(np.clip(p, 0.0, L)),
            dV=lambda p: cs(np.clip(p, 0.0, L), 1),
        )
        out["S3_line"] = float(inst.findAction(inst.findProfile()))
    except Exception as exc:               # noqa: BLE001
        print(f"  [warn] single-field instanton failed: {exc}")
    try:
        res = pathDeformation.fullTunneling(
            np.array([true_pt, false_pt]), Vf, dVf, verbose=False, maxiter=30
        )
        out["S3_def"] = float(res.action)
        out["path"] = np.asarray(res.Phi)
    except Exception as exc:               # noqa: BLE001
        print(f"  [warn] path deformation failed: {exc}")
    for k in ("S3_line", "S3_def"):
        out[k + "_over_T"] = None if out[k] is None else out[k] / T
    return out


def main():
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--T", type=float, default=100.0,
                    help="temperature in GeV the 2D grid was trained at (default 100)")
    ap.add_argument("--t-val", type=float, default=-2.0,
                    help="RG-time slice (default -2, the IR end of the flow)")
    ap.add_argument("--data-dir", default="data_UV_2D",
                    help="directory holding model_*_rho{i}_sig{j}_block{b}.pt")
    ap.add_argument("--n-points", type=int, default=80,
                    help="stitched-grid resolution per block")
    ap.add_argument("--outdir", default="plots_pinn")
    ap.add_argument("--device", default="cpu", choices=["cpu", "cuda"])
    args = ap.parse_args()

    if args.data_dir != "data_UV_2D":
        raise SystemExit(
            "show_UV2D hard-codes ./data_UV_2D/; symlink or copy your models there "
            "and rerun with the default --data-dir."
        )

    cp = _patch_config(args.T, args.device)
    print(f"[config] T={cp.T_RAW}  k_IR={cp.k_IR}  lamS={cp.lamS}  "
          f"lamHS={cp.lamHS}  mssq={cp.mssq}  t_val={args.t_val}")

    Vnn, dVnn, Vref, dVref, meta = _build_potentials(
        cp, args.t_val, args.n_points, args.data_dir
    )
    k_IR = meta["k_IR"]

    results = {}
    for tag, (Vf, dVf) in (("PINN", (Vnn, dVnn)),
                           ("reference", (Vref, dVref))):
        true_pt, false_pt, (h_v, s_v) = _axis_vacua(
            Vf, meta["rho_ax"], meta["sig_ax"], k_IR
        )
        which = "EW" if true_pt[0] > true_pt[1] else "singlet"
        print(f"\n[{tag}] vacua: h_v={h_v:.2f} GeV  s_v={s_v:.2f} GeV   "
              f"true = {which}")
        b = _bounce(Vf, dVf, true_pt, false_pt, args.T)
        b["true_pt"], b["false_pt"] = true_pt, false_pt
        results[tag] = b
        sl = "n/a" if b["S3_line_over_T"] is None else f"{b['S3_line_over_T']:.1f}"
        sd = "n/a" if b["S3_def_over_T"] is None else f"{b['S3_def_over_T']:.1f}"
        print(f"[{tag}] S_3/T  straight-line = {sl:>7}   path-deformed = {sd:>7}")

    os.makedirs(args.outdir, exist_ok=True)
    _plot(results, args, k_IR)
    _save_npz(results, args, meta)


def _plot(results, args, k_IR):
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    fig, ax = plt.subplots(figsize=(7.0, 4.3))
    styles = dict(PINN=dict(c="tab:red", ls="-", label="NN+ring (stitched 2D)"),
                  reference=dict(c="tab:green", ls=":",
                                 label="finite-$T$ reference"))
    val = None
    for tag, b in results.items():
        ax.plot(b["xi"], b["Vl"] / k_IR**4, lw=1.7, **styles[tag])
        if tag == "PINN":
            val = b["S3_def_over_T"] or b["S3_line_over_T"]
    ax.axhline(0.0, color="k", lw=0.6)
    ax.set_xlabel(r"straight-path fraction $\xi$   "
                  r"($0$: false $\to$ $1$: true vacuum)")
    ax.set_ylabel(r"$u - u_{\rm false}$")
    ttl = rf"$T={args.T:.0f}$ GeV, $t={args.t_val:g}$"
    if val is not None:
        pnn = results["PINN"]
        sl = pnn["S3_line_over_T"]
        sl_txt = "n/a" if sl is None else f"{sl:.0f}"
        ttl += (rf"    provisional $S_3/T \approx {val:.0f}$"
                rf"  (straight-line {sl_txt})")
    ax.set_title(ttl)
    ax.legend()
    ax.grid(alpha=0.3)
    fig.tight_layout()
    out = os.path.join(args.outdir, f"bounce_T{args.T:.0f}.png")
    fig.savefig(out, dpi=150)
    print(f"\nsaved {out}")


def _save_npz(results, args, meta):
    out = os.path.join(args.outdir, f"bounce_T{args.T:.0f}.npz")
    payload = dict(T=args.T, t_val=args.t_val, k_IR=meta["k_IR"],
                   rho_ax=meta["rho_ax"], sig_ax=meta["sig_ax"],
                   U_FULL=meta["U_FULL"])
    for tag, b in results.items():
        for key in ("xi", "Vl", "L", "S3_line", "S3_def",
                    "S3_line_over_T", "S3_def_over_T", "true_pt", "false_pt",
                    "path"):
            v = b.get(key)
            payload[f"{tag}_{key}"] = np.nan if v is None else v
    np.savez(out, **payload)
    print(f"saved {out}")


if __name__ == "__main__":
    main()
