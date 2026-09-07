"""
frg_pinns_higgs_singlet/loss_extensions.py の
`loss_sign_and_mag_couplings` (符号・大きさのソフト制約) を、PINNの
"損失" ではなく Newton-Krylov リラクゼーション法の "残差" として使える
形に翻訳したもの。

PINN側では、各点での実効quartic結合定数 (u_rhorho/2, u_sigmasigma/2,
u_rhosigma) と実効質量パラメータ (u_rho - u_rhorho*rho - u_rhosigma*sigma
等) を、摂動論的RG-improved running coupling/massの値と比較し、
  - 符号が違っていたらペナルティ (sign hinge)
  - 目標値に対する比が [c_lower, c_upper] の帯から外れたらペナルティ (magnitude band)
という "大まかな向きと大きさだけを合わせる" 緩い制約を損失に加えている
(点ごとの値を厳密に一致させるものではない)。

ここではこれを、各格子点の flow_rhs (du/dt) に加える forcing 項として
実装する。rloop (質量固有値・熱閾値関数から来る反応項) と同じ "場所"
(du/dtの中の反応項) に、もう1種類の反応項として加わる形になる。

注意 (フェアネスに関する留意点):
    weight_sign, weight_mag > 0 の間は、Newtonが実際に解く方程式は
    "純粋なWetterichフロー R=0" ではなく "Wetterichフロー + このsoft
    forcing" になる。収束解はこの摂動論的priorに引っ張られたバイアス解
    であり、真のFRG解とは一般に一致しない (PINN側もこのlossを完全に
    ゼロへ追い込んでいるわけではなく、他のloss項との重み付き妥協点で
    止まっているので、この点はPINN側と同じ性質を持つ)。継続法として
    使う場合 (--coupling-prior-anneal-iters) は、weightを段階的に0まで
    下げていくことで、最終的な収束解を純粋なフロー方程式の解に戻せる。
"""

import numpy as np

import _pathsetup  # noqa: F401
from perturbation.config_params import k_IR, t_range
from running_couplings import get_running_quartics, get_running_masses
from seed_potential import aH, aS, lamH, lamS, lamHS

_EPS = 1e-12


def _relu(x):
    return np.maximum(x, 0.0)


# relu(x)=max(x,0) has a kink at x=0; Jacobian-free Newton-Krylov's finite-
# difference directional derivatives see that kink as a source of noise (same
# mechanism as the sqrt(disc) branch point diagnosed elsewhere), and empirically
# fails to converge once this prior is kept active with a fixed nonzero weight.
# _softplus is a smooth (C^infinity) stand-in: beta->inf recovers relu exactly;
# beta=10 keeps the transition within ~0.1-0.3 (natural scale of the normalized
# arguments below) while removing the corner. Matches frg_higgs_only/coupling_prior.py.
_SOFTPLUS_BETA = 10.0


def _softplus(x, beta=_SOFTPLUS_BETA):
    bx = beta * x
    return (np.maximum(bx, 0.0) + np.log1p(np.exp(-np.abs(bx)))) / beta


def _sign_hinge(target, val, norm):
    """target と符号が一致していれば0、逆符号なら |val|/norm に比例したペナルティ。"""
    s = np.sign(target)
    return _softplus(-(s * val) / norm)


def _mag_band(target, val, norm, c_lower, c_upper):
    """|val| が |target| の [c_lower, c_upper] 倍の帯に入っていれば0。"""
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
    sigma_cut=3.0,
    mask_temp=10.0,
    f_threshold=1e-2,
    d_threshold=1e-2,
    disable_sign_F_H=False,
):
    """
    摂動論の running quartics/masses に対する符号・大きさのソフト制約を、
    格子点ごとの forcing (flow_rhs に加算する項) として評価する。

    weight_sign == weight_mag == 0.0 のときは常にゼロを返す (デフォルトの
    挙動を変えない)。

    rho_cut, sigma_cut : PINN側 hp.rho_cw_cut/sigma_cw_cut (デフォルト3.0) と
        同じ役割のソフトマスク境界。既定値3.0はこのソルバーの格子範囲
        (rho_max, sigma_max ~ 1.75) を覆うので事実上マスク無効。
    f_threshold, d_threshold : 目標値 (target) がtree値からほぼ動いていない
        (相対的に小さすぎて摂動論的に信頼できない) 領域を無視する閾値。
    disable_sign_F_H : True のとき、rho方向のmass-term (F_H, aH_NN vs
        aH_run/aH_tree) に対する sign hinge penalty だけを sign_pen から
        外す (magnitude band penalty 側の F_H 制約はそのまま残す)。
    """
    if weight_sign == 0.0 and weight_mag == 0.0:
        return 0.0

    rho, sigma = grid.RHO, grid.SIGMA
    u_rho, u_sigma, u_rhorho, u_sigmasigma, u_rhosigma = grid.derivatives(U, scheme="central")

    nn_H = 0.5 * u_rhorho
    nn_S = 0.5 * u_sigmasigma
    nn_HS = u_rhosigma
    aH_NN = u_rho - u_rhorho * rho - u_rhosigma * sigma
    aS_NN = u_sigma - u_sigmasigma * sigma - u_rhosigma * rho

    lamH_t, lamS_t, lamHS_t = get_running_quartics(t_phys)
    mhsq_t, mssq_t = get_running_masses(t_phys)
    kt = k_IR * np.exp(-t_range + t_phys)
    aH_run = mhsq_t / kt ** 2
    aS_run = mssq_t / kt ** 2

    aH_tree = aH * np.exp(-2.0 * t_phys)
    aS_tree = aS * np.exp(-2.0 * t_phys)

    D_H_target, D_S_target, D_HS_target = lamH_t - lamH, lamS_t - lamS, lamHS_t - lamHS
    D_H_NN, D_S_NN, D_HS_NN = nn_H - lamH, nn_S - lamS, nn_HS - lamHS

    F_H_target, F_S_target = aH_run - aH_tree, aS_run - aS_tree
    F_H_NN, F_S_NN = aH_NN - aH_tree, aS_NN - aS_tree

    norm_lamH = abs(lamH) + _EPS
    norm_lamS = abs(lamS) + _EPS
    norm_lamHS = abs(lamHS) + _EPS
    norm_aH = np.abs(aH_tree) + _EPS
    norm_aS = np.abs(aS_tree) + _EPS

    mask_D_H = float(abs(D_H_target / norm_lamH) > d_threshold)
    mask_D_S = float(abs(D_S_target / norm_lamS) > d_threshold)
    mask_D_HS = float(abs(D_HS_target / norm_lamHS) > d_threshold)
    mask_F_H = float(abs(F_H_target / norm_aH) > f_threshold)
    mask_F_S = float(abs(F_S_target / norm_aS) > f_threshold)

    mask_rho = 1.0 / (1.0 + np.exp(-mask_temp * (rho_cut - rho)))
    mask_sigma = 1.0 / (1.0 + np.exp(-mask_temp * (sigma_cut - sigma)))
    soft_mask = mask_rho * mask_sigma

    sign_pen = (
        mask_D_H * _sign_hinge(D_H_target, D_H_NN, norm_lamH)
        + mask_D_S * _sign_hinge(D_S_target, D_S_NN, norm_lamS)
        + mask_D_HS * _sign_hinge(D_HS_target, D_HS_NN, norm_lamHS)
        + (0.0 if disable_sign_F_H else mask_F_H * _sign_hinge(F_H_target, F_H_NN, norm_aH))
        + mask_F_S * _sign_hinge(F_S_target, F_S_NN, norm_aS)
    )

    mag_pen = (
        mask_F_H * _mag_band(F_H_target, F_H_NN, norm_aH, c_lower, c_upper)
        + mask_F_S * _mag_band(F_S_target, F_S_NN, norm_aS, c_lower, c_upper)
        # quartic側は上限のみ (真空安定性を壊す方向にNNが暴走しないようにする
        # loss_extensions.py の _loss_sign_and_mag_couplings_tval_finiteT と同じ発想)
        + _softplus(np.abs(D_H_NN / lamH) - quartic_ratio_cap)
        + _softplus(np.abs(D_S_NN / lamS) - quartic_ratio_cap)
        + _softplus(np.abs(D_HS_NN / lamHS) - quartic_ratio_cap)
    )

    return soft_mask * (weight_sign * sign_pen + weight_mag * mag_pen)
