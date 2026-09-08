import torch
import torch.nn as nn
#import numpy as np
#import torch.autograd as autograd

from perturbation.config_params import tau_uv, finite_T
from running_couplings import tree_params
[aH,lamH] = tree_params

# ============================================================
#  Higgs + Yukawa + gauge system (no singlet), rho = |H|^2/k^2
#
# tree-level seed
# ============================================================

def u_tree_exact(t_phys, rho_phys):
    muH2_t = aH * torch.exp(-2.0 * t_phys)

    return (
        + muH2_t * rho_phys
        + lamH * rho_phys**2
    )


# ============================================================
# utility modules
# ============================================================

def _as_list(hidden_dims):
    if hidden_dims is None:
        return []
    if isinstance(hidden_dims, int):
        return [hidden_dims]
    return list(hidden_dims)


class FlexibleMLP(nn.Module):
    """
    hidden_dims=[] means the identity map
    e.g.:
        in_dim=2, hidden_dims=[256,256,256]
        -> Linear(2,256)-SiLU-Linear(256,256)-SiLU-Linear(256,256)-SiLU
    """
    def __init__(self, in_dim, hidden_dims, activation=nn.SiLU):
        super().__init__()
        hidden_dims = _as_list(hidden_dims)

        layers = []
        prev_dim = in_dim
        for h in hidden_dims:
            layers.append(nn.Linear(prev_dim, h))
            layers.append(activation())
            prev_dim = h

        self.net = nn.Sequential(*layers)
        self.out_dim = prev_dim

    def forward(self, x):
        if len(self.net) == 0:
            return x
        return self.net(x)


class LinearHead(nn.Module):
    def __init__(self, in_dim, out_dim=1, zero_init=True):
        super().__init__()
        self.fc = nn.Linear(in_dim, out_dim)
        if zero_init:
            nn.init.zeros_(self.fc.weight)
            nn.init.zeros_(self.fc.bias)

    def forward(self, h):
        return self.fc(h)


# ============================================================
# BaseNet: shared stem + rho branch
# ============================================================

class BaseNet(nn.Module):
    """
    return:
        [N, N_r, N_rr]
    """
    def __init__(
        self,
        stem_hidden_dims=None,
        rho_hidden_dims=None,
        activation=nn.SiLU,
    ):
        super().__init__()

        in_dim = 2  # (t, rho)

        # we may choose 160, 192, 224, 256
        #basew = 256
        basew = 160

        # defaults
        if stem_hidden_dims is None:
            stem_hidden_dims = [basew, basew, basew]
        if rho_hidden_dims is None:
            rho_hidden_dims = [basew, basew]

        # shared stem
        self.stem = FlexibleMLP(
            in_dim=in_dim,
            hidden_dims=stem_hidden_dims,
            activation=activation,
        )

        stem_out_dim = self.stem.out_dim

        self.u_branch = FlexibleMLP(
            in_dim=stem_out_dim,
            hidden_dims=[basew],
            activation=activation,
        )
        self.head_N = LinearHead(self.u_branch.out_dim, 1)

        # rho branch
        self.branch_rho = FlexibleMLP(
            in_dim=stem_out_dim,
            hidden_dims=rho_hidden_dims,
            activation=activation,
        )
        rho_out_dim = self.branch_rho.out_dim

        self.head_rho = LinearHead(rho_out_dim, 1)   # N_r (for u_r)
        self.head_rr  = LinearHead(rho_out_dim, 1)   # N_rr (for u_rr)

    def forward(self, t_in, rho_in):
        x = torch.cat([t_in, rho_in], dim=-1)

        # shared stem
        h = self.stem(x)

        # u branch
        h_u = self.u_branch(h)
        N   = self.head_N(h_u)

        # rho branch
        h_rho = self.branch_rho(h)
        N_r   = self.head_rho(h_rho)
        N_rr  = self.head_rr(h_rho)

        return torch.cat([N, N_r, N_rr], dim=-1)


# ============================================================
# WrappedPotentialNet
# ============================================================

class WrappedPotentialNet(nn.Module):
    def __init__(self, base_net, T_min, T_max, Rho_min, Rho_max):
        super().__init__()
        self.base = base_net
        self.T_min = T_min
        self.T_max = T_max
        self.Rho_min = Rho_min
        self.Rho_max = Rho_max

    def scale_t(self, t_phys):
        return 2.0 * (t_phys - self.T_min) / (self.T_max - self.T_min) - 1.0

    def unscale_t(self, t_in):
        return 0.5 * (t_in + 1.0) * (self.T_max - self.T_min) + self.T_min

    @property
    def A_T(self):
        return 2.0 / (self.T_max - self.T_min)

    def scale_rho(self, rho_phys):
        return 2.0 * (rho_phys - self.Rho_min) / (self.Rho_max - self.Rho_min) - 1.0

    def unscale_rho(self, rho_in):
        return 0.5 * (rho_in + 1.0) * (self.Rho_max - self.Rho_min) + self.Rho_min

    @property
    def A_RHO(self):
        return 2.0 / (self.Rho_max - self.Rho_min)

    def forward(self, t_in, rho_in):
        t_phys = self.unscale_t(t_in)
        rho_phys = self.unscale_rho(rho_in)

        out = self.base(t_in, rho_in)
        N    = out[:, 0:1]
        N_r  = out[:, 1:2]
        N_rr = out[:, 2:3]

        muH2_t = aH * torch.exp(-2.0 * t_phys)

        if finite_T:
            kh1, kh2 = 1.0, 1.0
        else:
            kh1, kh2 = 0.8, 1.2

        u_tree = (
            muH2_t * rho_phys * kh1
            + lamH * rho_phys**2 * kh2
        )

        ur_tree  = muH2_t * kh1 + 2.0 * lamH * rho_phys * kh2
        urr_tree = 2.0 * lamH * kh2

        tau = tau_uv * torch.exp(-t_phys)
        _kh = 0.2

        if finite_T:
            u_th = _kh * rho_phys * tau**2
            ur_th = _kh * (tau**2)
        else:
            u_th = torch.zeros_like(rho_phys)
            ur_th = torch.zeros_like(rho_phys)

        u   = u_tree + u_th + N
        ur  = ur_tree + ur_th + N_r
        urr = urr_tree + N_rr

        return u, ur, urr
