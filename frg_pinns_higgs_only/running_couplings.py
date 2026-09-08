import torch
import numpy as np
import torch.autograd as autograd

from perturbation.config_params import k_IR, t_range, finite_T

from perturbation.rges import make_running_couplings
rc = make_running_couplings(mu0=k_IR, mu_end=2000.0)

_tuv=0.
_trge=np.log(k_IR) + (-t_range) + _tuv
lamH = rc["lam"](_trge)
_mhsq = rc["mhsq"](_trge)
aH = (_mhsq/k_IR**2)*np.exp(2.0*t_range)

print('lamH, mhsq at UV scale',lamH, _mhsq)
print('finite_T=', finite_T)

tree_params=[aH,lamH]

def _torch_from_rc(arr, ref):
    with torch.no_grad():
        t = torch.as_tensor(arr, dtype=ref.dtype, device=ref.device).reshape_as(ref)
    return t

def _rge_t_from_phys_t(t_phys):
    # t_phys: torch.Tensor (batch,1)
    t_phys_np = t_phys.detach().cpu().numpy()
    # first flatten to 1D
    t_flat = t_phys_np.reshape(-1)
    t_rge_flat = np.log(k_IR) + (-t_range) + t_flat
    t_min = rc["t_min"]
    t_max = rc["t_max"]
    # clamp with a little margin on the lower side only
    t_rge_flat = np.maximum(t_rge_flat, t_min + 1e-8)

    return t_rge_flat  # returned still as 1D (N,)

def get_running_couplings(t):
    # t: torch.Tensor with shape (N,1) or (...,1)
    t_rge_flat = _rge_t_from_phys_t(t)          # shape (N,)
    g1_flat = rc["g1"](t_rge_flat)              # shape (N,)
    g2_flat = rc["g2"](t_rge_flat)
    yt_flat = rc["yt"](t_rge_flat)
    # restore to the same shape as the original t
    g1_ = _torch_from_rc(g1_flat.reshape(-1, 1), t)
    g2_ = _torch_from_rc(g2_flat.reshape(-1, 1), t)
    yt_ = _torch_from_rc(yt_flat.reshape(-1, 1), t)
    return g1_, g2_, yt_

def get_running_quartics(t):
    t_rge_flat = _rge_t_from_phys_t(t)
    lamH_flat = rc["lam"](t_rge_flat)
    lamH_ = _torch_from_rc(lamH_flat.reshape(-1, 1), t)
    return lamH_

def get_running_masses(t):
    t_rge_flat = _rge_t_from_phys_t(t)
    mhsq_flat = rc["mhsq"](t_rge_flat)
    mhsq_ = _torch_from_rc(mhsq_flat.reshape(-1, 1), t)
    return mhsq_
