"""
frg_pinns_higgs_singlet/running_couplings.py の NumPy 版。

摂動論的2-loop RGE (perturbation.rges) をUVスケールまで走らせ、
FRGフローの "UV端" (t_phys=0, k = k_IR*exp(-t_range)) におけるマッチング値
(tree_params = [aH, aS, lamH, lamS, lamHS]) を用意する。

また、FRGフロー中の任意の t_phys (物理的なFRG time) を摂動論のRGEスケール
mu にマッピングして、その場での running couplings/quartics/masses を返す
get_running_* 関数を提供する。

FRG time の規約 (frg_pinns_higgs_singlet と共通):
    t_phys = 0        : FRGフローのUV端 (k = k_IR * exp(-t_range) = k_IR * exp(2))
    t_phys = t_range  : FRGフローのIR端 (k = k_IR, config_params.py 参照)
    perturbative RGE のスケール: mu(t_phys) = k_IR * exp(t_phys - t_range)
"""

import numpy as np

from perturbation.config_params import k_IR, t_range, finite_T
from perturbation.rges import make_running_couplings

rc = make_running_couplings(mu0=k_IR, mu_end=2000.0)

_tuv = 0.0
_trge = np.log(k_IR) + (-t_range) + _tuv

lamS = float(rc["lamS"](_trge))
lamHS = float(rc["lamHS"](_trge))
lamH = float(rc["lam"](_trge))
_mssq = float(rc["mssq"](_trge))
_mhsq = float(rc["mhsq"](_trge))

aS = (_mssq / k_IR ** 2) * np.exp(2.0 * t_range)
aH = (_mhsq / k_IR ** 2) * np.exp(2.0 * t_range)

print('lamS,lamHS,lamH, mssq, mhsq at UV scale', lamS, lamHS, lamH, _mssq, _mhsq)
print('finite_T=', finite_T)

tree_params = [aH, aS, lamH, lamS, lamHS]


def _rge_t_from_phys_t(t_phys):
    t_phys_arr = np.asarray(t_phys, dtype=float)
    t_rge = np.log(k_IR) + (-t_range) + t_phys_arr
    t_min = rc["t_min"]
    t_rge = np.maximum(t_rge, t_min + 1e-8)
    return t_rge


def get_running_couplings(t_phys):
    """Return (g1, g2, yt, lamS, lamHS) at the given FRG time(s) t_phys."""
    t_rge = _rge_t_from_phys_t(t_phys)
    g1_ = rc["g1"](t_rge)
    g2_ = rc["g2"](t_rge)
    yt_ = rc["yt"](t_rge)
    lamS_ = rc["lamS"](t_rge)
    lamHS_ = rc["lamHS"](t_rge)
    return g1_, g2_, yt_, lamS_, lamHS_


def get_running_quartics(t_phys):
    """Return (lamH, lamS, lamHS) at the given FRG time(s) t_phys."""
    t_rge = _rge_t_from_phys_t(t_phys)
    lamH_ = rc["lam"](t_rge)
    lamS_ = rc["lamS"](t_rge)
    lamHS_ = rc["lamHS"](t_rge)
    return lamH_, lamS_, lamHS_


def get_running_masses(t_phys):
    """Return (mhsq, mssq) at the given FRG time(s) t_phys."""
    t_rge = _rge_t_from_phys_t(t_phys)
    mhsq_ = rc["mhsq"](t_rge)
    mssq_ = rc["mssq"](t_rge)
    return mhsq_, mssq_
