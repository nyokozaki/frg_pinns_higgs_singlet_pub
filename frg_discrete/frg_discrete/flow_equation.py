"""
Wetterich方程式 (LPA', 有限温度) の右辺を格子上で評価する。

frg_pinns_higgs_singlet/FRG_residual.py の R_FRG_in を NumPy に移植したもの。
PINN側は残差 R = rtree - rloop を損失として R=0 を学習させているが、
ここでは同じ R=0 を u_t について解いた

    du/dt = -4u + (2+eta_rho)*rho*u_rho + (2+eta_sigma)*sigma*u_sigma + rloop

を method-of-lines の右辺 (RHS) として直接使う。

FRG time の規約は running_couplings.py と同じ:
  t_phys=0 が UV端、t_phys=t_range (負) が IR端。

--------------------------------------------------------------------
tree/残差分解 (flow_rhs_w) について
--------------------------------------------------------------------
u_tree_exact(t,rho,sigma) (質量項だけ a -> a*exp(-2t) とスケールし、
quartic類 lamH,lamS,lamHS は固定) は、eta=0・rloop=0 の"自由"フロー

    du/dt = -4u + 2*rho*u_rho + 2*sigma*u_sigma

の厳密解になっている (u_tree自体がRG canonical scalingそのものだから)。
そこで u = u_tree + w とおくと、w = u - u_tree が満たす方程式は

    dw/dt = F_tree(t,rho,sigma) - 4w + (2+eta_rho)*rho*w_rho
            + (2+eta_sigma)*sigma*w_sigma + rloop(u_tree+w由来の質量, t)

    F_tree = eta_rho(t)*rho*u_tree_rho + eta_sigma(t)*sigma*u_tree_sigma

になる (u_tree部分の "-4u_tree+2rho*u_tree_rho+2sigma*u_tree_sigma-du_tree/dt"
は恒等的に0なので、残るのは eta 由来の項だけ)。eta_rho, eta_sigma は
running couplingsから来るloop生成量なので、F_treeも本質的にはloop effect の
一部であり、"tree" ではなく w 側 (残差) に属する。

u_tree部分は解析的に厳密 (離散化誤差ゼロ) なので、-4u の正準スケーリング項が
持つ指数的増幅 (積分区間 Delta t=2 で e^{4*2}=e^8 ~ 3000倍) にさらされるのは
残差 w だけになり、w は tree よりずっと小さい (thermal補正程度) ため、
数値誤差の増幅による影響を大幅に抑えられる。
"""

import numpy as np

from perturbation.config_params import tau_uv, finite_T
from running_couplings import get_running_couplings
from seed_potential import aH, aS, lamH, lamS, lamHS

kloop = 1.0 / (12.0 * np.pi ** 2)
NGS = 3.0
LPAp = 1.0
EPS = 1e-10


def eta_rho(t_phys):
    g1, g2, yt, _, _ = get_running_couplings(t_phys)
    gamma = (1.0 / (16.0 * np.pi ** 2)) * (
        3.0 * yt ** 2 - 3.0 * g1 ** 2 / 4.0 - 9.0 * g2 ** 2 / 4.0
    )
    return 2.0 * gamma


def eta_sigma(t_phys):
    _, _, _, lamS_t, lamHS_t = get_running_couplings(t_phys)
    # gamma_S from PyR@TE3 for this model (matches frg_pinn.tex Sec. 2.3 and the
    # PINN get_eta_sigma in frg_pinns_higgs_singlet/FRG_residual.py):
    #   gamma_S = (16 pi^2)^-2 (lambda_HS^2 + 3 lambda_S^2),  eta_S = 2 gamma_S.
    # Runs before 2026-09 used (18 lamS^2 + 2 lamHS^2) -- the 1-loop beta_lamS
    # coefficient reused as a proxy; eta_S ~ 1e-4 either way and its effect on the
    # flow is ~1e-5.
    gamma = (1.0 / (16.0 * np.pi ** 2)) ** 2 * (3.0 * lamS_t ** 2 + 1.0 * lamHS_t ** 2)
    return 2.0 * gamma


def _coth(x, eps=0.0):
    return np.cosh(x) / (np.sinh(x) + eps)


def _inv_sqrt(m2, floor=EPS):
    return 1.0 / np.sqrt(np.clip(1.0 + m2, floor, None))


def u_tree_derivs(t_phys, rho, sigma):
    """u_tree_exact の (u_rho, u_sigma, u_rhorho, u_sigmasigma, u_rhosigma) を厳密に返す。"""
    muH2_t = aH * np.exp(-2.0 * t_phys)
    muS2_t = aS * np.exp(-2.0 * t_phys)
    u_rho = muH2_t + 2.0 * lamH * rho + lamHS * sigma
    u_sigma = muS2_t + 2.0 * lamS * sigma + lamHS * rho
    u_rhorho = 2.0 * lamH
    u_sigmasigma = 2.0 * lamS
    u_rhosigma = lamHS
    return u_rho, u_sigma, u_rhorho, u_sigmasigma, u_rhosigma


def _rloop_from_derivs(t_phys, rho, sigma, u_rho, u_sigma, u_rhorho, u_sigmasigma, u_rhosigma,
                        warn_on_tachyon=False, disc_delta=0.0, disc_cut=0.0, mass_floor=0.0):
    """
    質量固有値と熱閾値関数から rloop = kloop*loop_sum (clip済み) を計算する。
    flow_rhs (フル u) / flow_rhs_w (残差 w、u_tree+w由来の合成微分を渡す) の共通部分。

    disc_delta : Higgs/singlet質量固有値の判別式 disc=(M11-M22)^2+4*M12^2 に
                 対する正則化パラメータ。disc_delta>0 のとき
                 sqrt_disc = sqrt(disc + disc_delta**2) を使う (円錐型の
                 特異点 disc=0 を半径~disc_deltaで滑らかに丸める。ただし
                 disc=0近傍の微分は~1/disc_deltaのオーダーで残る)。
    disc_cut : disc_delta と排他的な代替正則化。disc_cut>0 のとき
               sqrt_disc = sqrt(max(disc, disc_cut**2)) というハードクリップを使う。
               disc < disc_cut**2 の領域では sqrt_disc が定数になり、その領域内での
               U に対する微分が厳密にゼロになる (JFNKの有限差分ヤコビアンが
               この項からノイズを拾わなくなる、という狙い。ただし disc=disc_cut**2
               の境界に微分不連続の折れ目が残る)。
    デフォルト (disc_delta=disc_cut=0.0) は従来通り sqrt(clip(disc, EPS, None))
    を使い、挙動を変えない (frg_discrete3等の既存呼び出し元との後方互換性のため)。
    mass_floor : Higgs/singlet質量固有値 (mG2, m1_sq, m2_sq) の規格化質量二乗
                 1+m^2 に対するハードクリップのfloor値。mass_floor>0 のとき
                 _inv_sqrt と有限温度項のEの両方で clip(1+m^2, EPS, None) の
                 代わりに clip(1+m^2, mass_floor, None) を使う。gauge/topは
                 1+m^2>=1 で floor に触れないため無関係。デフォルト0.0は
                 従来通りEPS floor。
    """
    g1, g2, yt, _, _ = get_running_couplings(t_phys)

    mG2 = u_rho
    M11 = u_rho + 2.0 * rho * u_rhorho
    M22 = u_sigma + 2.0 * sigma * u_sigmasigma
    M12 = 2.0 * np.sqrt(np.clip(rho * sigma, EPS, None)) * u_rhosigma

    disc = (M11 - M22) ** 2 + 4.0 * M12 ** 2
    if disc_cut > 0.0:
        sqrt_disc = np.sqrt(np.clip(disc, disc_cut ** 2, None))
    elif disc_delta > 0.0:
        sqrt_disc = np.sqrt(disc + disc_delta ** 2)
    else:
        sqrt_disc = np.sqrt(np.clip(disc, EPS, None))

    m1_sq = 0.5 * (M11 + M22 - sqrt_disc)
    m2_sq = 0.5 * (M11 + M22 + sqrt_disc)

    if warn_on_tachyon:
        if np.any(1.0 + mG2 < 0.0) or np.any(1.0 + m1_sq < 0.0):
            import warnings
            warnings.warn(
                f"t={t_phys:.4f}: 1+mG2 or 1+m1_sq went negative "
                f"(min 1+mG2={np.min(1.0+mG2):.3e}, min 1+m1_sq={np.min(1.0+m1_sq):.3e}); "
                "clamped at eps, regulator is being pushed into an unphysical regime."
            )

    mW_T2 = 0.5 * g2 ** 2 * rho
    mZ_T2 = 0.5 * (g2 ** 2 + g1 ** 2) * rho
    mt2 = yt ** 2 * rho
    mW_L2 = mW_T2
    mZ_L2 = mZ_T2

    mass_eps = mass_floor if mass_floor > 0.0 else EPS
    termG_0 = _inv_sqrt(mG2, floor=mass_eps)
    term1_0 = _inv_sqrt(m1_sq, floor=mass_eps)
    term2_0 = _inv_sqrt(m2_sq, floor=mass_eps)
    termWT0 = _inv_sqrt(mW_T2)
    termWL0 = _inv_sqrt(mW_L2)
    termZT0 = _inv_sqrt(mZ_T2)
    termZL0 = _inv_sqrt(mZ_L2)
    termT0 = _inv_sqrt(mt2)

    if finite_T:
        tau_t = tau_uv * np.exp(-t_phys)

        EG = np.sqrt(np.clip(1.0 + mG2, mass_eps, None))
        E1 = np.sqrt(np.clip(1.0 + m1_sq, mass_eps, None))
        E2 = np.sqrt(np.clip(1.0 + m2_sq, mass_eps, None))
        E_WT = np.sqrt(np.clip(1.0 + mW_T2, EPS, None))
        E_WL = np.sqrt(np.clip(1.0 + mW_L2, EPS, None))
        E_ZT = np.sqrt(np.clip(1.0 + mZ_T2, EPS, None))
        E_ZL = np.sqrt(np.clip(1.0 + mZ_L2, EPS, None))
        E_T = np.sqrt(np.clip(1.0 + mt2, EPS, None))

        factorG = _coth(EG / (2.0 * tau_t))
        factor1 = _coth(E1 / (2.0 * tau_t))
        factor2 = _coth(E2 / (2.0 * tau_t))
        factor_WT = _coth(E_WT / (2.0 * tau_t))
        factor_WL = _coth(E_WL / (2.0 * tau_t))
        factor_ZT = _coth(E_ZT / (2.0 * tau_t))
        factor_ZL = _coth(E_ZL / (2.0 * tau_t))
        factor_T = np.tanh(E_T / (2.0 * tau_t))

        termG = termG_0 * factorG
        term1 = term1_0 * factor1
        term2 = term2_0 * factor2
        termWT = termWT0 * factor_WT
        termWL = termWL0 * factor_WL
        termZT = termZT0 * factor_ZT
        termZL = termZL0 * factor_ZL
        termT = termT0 * factor_T
    else:
        termG, term1, term2 = termG_0, term1_0, term2_0
        termWT, termWL = termWT0, termWL0
        termZT, termZL = termZT0, termZL0
        termT = termT0

    loop_sum = (
        3.0 * termG
        + term1
        + term2
        + 4.0 * termWT
        + 2.0 * termWL
        + 2.0 * termZT
        + 1.0 * termZL
        - 12.0 * termT
    )

    rloop = kloop * loop_sum
    rloop = np.clip(rloop, -2.0, 2.0)
    return rloop


def flow_rhs(t_phys, U, grid, warn_on_tachyon=False, scheme="upwind", disc_delta=0.0, disc_cut=0.0,
             mass_floor=0.0):
    """
    du/dt(t_phys, rho, sigma) を格子全体 (grid.shape の2次元配列) で返す (フル u を直接積分)。

    U : ndarray, shape grid.shape
        現在のRG time t_phys における無次元ポテンシャル u(rho, sigma)。
    scheme : "upwind" (デフォルト、explicit時間積分に対して安定) か
             "central" (高精度だが高解像度でexplicit積分が不安定になりうる)。
             grid_fd.Grid2D.derivatives 参照。
    disc_delta, disc_cut, mass_floor : _rloop_from_derivs 参照
        (質量固有値判別式の正則化、互いに排他。mass_floorは独立に併用可)。
    """
    u_rho, u_sigma, u_rhorho, u_sigmasigma, u_rhosigma = grid.derivatives(U, scheme=scheme)

    rho = grid.RHO
    sigma = grid.SIGMA

    eta_r = eta_rho(t_phys)
    eta_s = eta_sigma(t_phys)

    rloop = _rloop_from_derivs(
        t_phys, rho, sigma, u_rho, u_sigma, u_rhorho, u_sigmasigma, u_rhosigma,
        warn_on_tachyon=warn_on_tachyon, disc_delta=disc_delta, disc_cut=disc_cut,
        mass_floor=mass_floor,
    )

    du_dt = (
        -4.0 * U
        + (2.0 + eta_r * LPAp) * rho * u_rho
        + (2.0 + eta_s * LPAp) * sigma * u_sigma
        + rloop
    )

    return du_dt


def flow_rhs_w(t_phys, W, grid, warn_on_tachyon=False, scheme="upwind"):
    """
    u = u_tree_exact(t,rho,sigma) + w とおいたときの dw/dt を返す。

    tree部分の "-4u_tree+2rho*u_tree_rho+2sigma*u_tree_sigma-du_tree/dt" は
    恒等的に0なので (u_tree自身がeta=0・rloop=0の自由フローの厳密解のため)、
    残る強制項は eta 由来の F_tree だけになる。u_tree部分は解析的に厳密
    (離散化誤差なし) に評価され、格子上の有限差分は w にしか使わない。
    """
    w_rho, w_sigma, w_rhorho, w_sigmasigma, w_rhosigma = grid.derivatives(W, scheme=scheme)

    rho = grid.RHO
    sigma = grid.SIGMA

    ut_rho, ut_sigma, ut_rhorho, ut_sigmasigma, ut_rhosigma = u_tree_derivs(t_phys, rho, sigma)

    u_rho = ut_rho + w_rho
    u_sigma = ut_sigma + w_sigma
    u_rhorho = ut_rhorho + w_rhorho
    u_sigmasigma = ut_sigmasigma + w_sigmasigma
    u_rhosigma = ut_rhosigma + w_rhosigma

    eta_r = eta_rho(t_phys)
    eta_s = eta_sigma(t_phys)

    rloop = _rloop_from_derivs(
        t_phys, rho, sigma, u_rho, u_sigma, u_rhorho, u_sigmasigma, u_rhosigma,
        warn_on_tachyon=warn_on_tachyon,
    )

    F_tree = eta_r * rho * ut_rho + eta_s * sigma * ut_sigma

    dw_dt = (
        F_tree
        - 4.0 * W
        + (2.0 + eta_r * LPAp) * rho * w_rho
        + (2.0 + eta_s * LPAp) * sigma * w_sigma
        + rloop
    )

    return dw_dt
