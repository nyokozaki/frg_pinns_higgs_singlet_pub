"""
NumPy version of frg_pinns_higgs_singlet/running_couplings.py.

Runs the perturbative two-loop RGEs (perturbation.rges) up to the UV scale and
prepares the matching values at the "UV edge" of the FRG flow
(t_phys=0, k = k_IR*exp(-t_range)): tree_params = [aH, aS, lamH, lamS, lamHS].

Also provides the get_running_* functions, which map an arbitrary t_phys (the
physical FRG time) during the flow onto the perturbative RGE scale mu and return
the running couplings / quartics / masses there.

FRG-time convention (shared with frg_pinns_higgs_singlet):
    t_phys = 0        : UV edge of the FRG flow (k = k_IR * exp(-t_range) = k_IR * exp(2))
    t_phys = t_range  : IR edge of the FRG flow (k = k_IR, see config_params.py)
    perturbative RGE scale: mu(t_phys) = k_IR * exp(t_phys - t_range)
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
