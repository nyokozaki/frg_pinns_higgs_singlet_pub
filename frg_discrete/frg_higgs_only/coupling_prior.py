"""
frg_pinns_higgs_only/loss_extensions.py の
`loss_sign_and_mag_couplings` (符号・大きさのソフト制約) の Higgs-only版を、
Newton-Krylov リラクゼーション法の"残差"として使える形に翻訳したもの。

weight_sign, weight_mag > 0 の間は、Newtonが実際に解く方程式は
"純粋なWetterichフロー R=0" ではなく "Wetterichフロー + このsoft forcing"
になる (継続法として使う場合は weight を段階的に0まで下げる)。
"""

import numpy as np

import _pathsetup  # noqa: F401
from perturbation.config_params import k_IR, t_range
from running_couplings_higgs_only import get_running_quartics, get_running_masses
from seed_potential_higgs_only import aH, lamH
import cw_thermal

_EPS = 1e-12


def _del_aH_thermal(t_phys, rho=None):
    """
    F_H_target (質量項のforcingターゲット) に thermal補正を折り込むための
    "有効aH" 寄与。u_thermal_finiteT (J_B/J_Fの数値積分) をrho方向に微分して
    局所2次フィットのaH成分を取り出す真面目なやり方は、Jacobian-free
    Newton-Krylovの1残差評価あたり3回のJ_B積分が乗る上に呼び出し回数も
    多く、実測で収束が極端に(20倍以上)遅くなったため採用しない。

    代わりに large-T展開のleading order (Debye熱質量 Pi_H,
    cw_thermal.Pi_H_thermal_mass) だけを使う。Pi_Hはrho非依存 (質量項への
    一様シフトのみ、quarticには効かない) なので、この項を加えても
    F_H_target自体はこれまで通りrho非依存のスカラーのまま — mask_F_Hを
    array化する必要もない。
    """
    return cw_thermal.Pi_H_thermal_mass(t_phys)

# relu(x)=max(x,0) has a kink at x=0; Jacobian-free Newton-Krylov's finite-
# difference directional derivatives see that kink as a source of noise
# (same mechanism as the sqrt(disc) branch point diagnosed elsewhere), and
# empirically fails to converge once this prior is kept active with a fixed
# nonzero weight. _softplus is a smooth (C^infinity) stand-in: beta->inf
# recovers relu exactly; beta=10 keeps the transition within ~0.1-0.3 (natural
# scale of the normalized arguments below) while removing the corner.
_SOFTPLUS_BETA = 10.0


def _softplus(x, beta=_SOFTPLUS_BETA):
    bx = beta * x
    return (np.maximum(bx, 0.0) + np.log1p(np.exp(-np.abs(bx)))) / beta


def _sign_hinge(target, val, norm):
    s = np.sign(target)
    return _softplus(-(s * val) / norm)


def _mag_band(target, val, norm, c_lower, c_upper):
    mt = np.abs(target)
    mv = np.abs(val)
    return _softplus((c_lower * mt - mv) / norm) + _softplus((mv - c_upper * mt) / norm)


def coupling_prior_residual(
    t_phys, U, grid,
    weight_sign=0.0,
    weight_mag=0.0,
    c_lower=0.1,
    c_upper=3.0,
    quartic_ratio_cap=0.99,
    rho_cut=3.0,
    rho_cw_cut=0.6,
    mask_temp=10.0,
    f_threshold=1e-2,
    d_threshold=1e-2,
    disable_sign_F_H=False,
):
    """
    摂動論の running quartic/mass (Higgs側のみ)に対する符号・大きさの
    ソフト制約を、格子点ごとの forcing (flow_rhs に加算する項) として評価する。

    rho_cw_cut: frg_pinns_higgs_only/loss_extensions.py の finite-T branch
    (`_loss_sign_and_mag_couplings_tval_finiteT`, hp.rho_cw_cut=0.6) と同じ
    低rho/高rho の役割分担を移植したもの — 質量項(F_H)の sign+magnitude
    制約は rho < rho_cw_cut のみ、quartic(D_H)のRGE-runターゲットに対する
    sign+magnitude帯 (c_lower/c_upperバンド) は rho > rho_cw_cut のみに
    適用する (低rho側でquarticを、高rho側でmassをそれぞれ縛らない、という
    PINN側の設計をそのまま踏襲)。D_H_NN/lamHに対する定数キャップ
    (quartic_ratio_cap) は元々rho全域で効く別項なので、この分割とは無関係に
    そのまま残す。
    """
    if weight_sign == 0.0 and weight_mag == 0.0:
        return 0.0

    rho = grid.RHO
    u_rho, u_rhorho = grid.derivatives(U, scheme="central")

    nn_H = 0.5 * u_rhorho
    aH_NN = u_rho - u_rhorho * rho

    lamH_t = get_running_quartics(t_phys)
    mhsq_t = get_running_masses(t_phys)
    kt = k_IR * np.exp(-t_range + t_phys)
    aH_run = mhsq_t / kt ** 2

    aH_tree = aH * np.exp(-2.0 * t_phys)

    D_H_target = lamH_t - lamH
    D_H_NN = nn_H - lamH

    del_aH = _del_aH_thermal(t_phys)
    aH_run = aH_run + del_aH

    F_H_target = aH_run - aH_tree
    F_H_NN = aH_NN - aH_tree

    norm_lamH = abs(lamH) + _EPS
    norm_aH = np.abs(aH_tree) + _EPS

    mask_D_H = float(abs(D_H_target / norm_lamH) > d_threshold)
    mask_F_H = float(abs(F_H_target / norm_aH) > f_threshold)

    # 元のdomain-wideゲート (rho_cut=3.0はdomain max 1.75より大きいので事実上1)
    mask_domain = 1.0 / (1.0 + np.exp(-mask_temp * (rho_cut - rho)))

    # rho_cw_cutでの低rho/高rho分割
    mask_rho_lo = 1.0 / (1.0 + np.exp(-mask_temp * (rho_cw_cut - rho)))
    mask_rho_hi = 1.0 / (1.0 + np.exp(-mask_temp * (rho - rho_cw_cut)))

    soft_mask_lo = mask_domain * mask_rho_lo
    soft_mask_hi = mask_domain * mask_rho_hi

    sign_pen = (
        soft_mask_hi * mask_D_H * _sign_hinge(D_H_target, D_H_NN, norm_lamH)
        + (0.0 if disable_sign_F_H else soft_mask_lo * mask_F_H * _sign_hinge(F_H_target, F_H_NN, norm_aH))
    )

    mag_pen = (
        soft_mask_lo * mask_F_H * _mag_band(F_H_target, F_H_NN, norm_aH, c_lower, c_upper)
        + soft_mask_hi * mask_D_H * _mag_band(D_H_target, D_H_NN, norm_lamH, c_lower, c_upper)
        + mask_domain * _softplus(np.abs(D_H_NN / lamH) - quartic_ratio_cap)
    )

    return weight_sign * sign_pen + weight_mag * mag_pen
