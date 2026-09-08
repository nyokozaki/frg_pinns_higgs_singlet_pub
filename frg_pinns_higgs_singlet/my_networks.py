import torch
import torch.nn as nn
#import numpy as np
#import torch.autograd as autograd

from perturbation.config_params import tau_uv, finite_T
from running_couplings import tree_params
[aH,aS,lamH,lamS,lamHS] = tree_params

# ============================================================
#  Higgs-Z2 real singlet model, rho = |H|^2/k^2, sigma = 1/2 S^2/k^2
#
# tree-level seed
# ============================================================

def u_tree_exact(t_phys, rho_phys, sigma_phys):
    muH2_t = aH * torch.exp(-2.0 * t_phys)
    muS2_t = aS * torch.exp(-2.0 * t_phys)

    return (
        + muH2_t * rho_phys
        + lamH * rho_phys**2
        + muS2_t * sigma_phys
        + lamS * sigma_phys**2
        + lamHS * rho_phys * sigma_phys
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
        in_dim=3, hidden_dims=[256,256,256]
        -> Linear(3,256)-SiLU-Linear(256,256)-SiLU-Linear(256,256)-SiLU
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
# BaseNet: shared stem + rho/sigma branches + mix branch
# ============================================================

class BaseNet(nn.Module):
    """
    return:
        [N, N_r, N_s, N_rr, N_ss, N_rs]
    Compatible with the existing interface.
    """
    def __init__(
        self,
        stem_hidden_dims=None,
        rho_hidden_dims=None,
        sigma_hidden_dims=None,
        mix_hidden_dims=None,
        activation=nn.SiLU,
    ):
        super().__init__()

        in_dim = 3  # (t, rho, sigma)

        # we may choose 160, 192, 224, 256
        #basew = 256
        basew = 160
        basew2 = int(2 * basew)


        # defaults
        if stem_hidden_dims is None:
            stem_hidden_dims = [basew, basew, basew]
        if rho_hidden_dims is None:
            rho_hidden_dims = [basew, basew]
        if sigma_hidden_dims is None:
            sigma_hidden_dims = [basew, basew]
        if mix_hidden_dims is None:
            mix_hidden_dims = [basew2, basew2]

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

        # sigma branch
        self.branch_sigma = FlexibleMLP(
            in_dim=stem_out_dim,
            hidden_dims=sigma_hidden_dims,
            activation=activation,
        )
        sigma_out_dim = self.branch_sigma.out_dim

        self.head_sigma = LinearHead(sigma_out_dim, 1)  # N_s
        self.head_ss    = LinearHead(sigma_out_dim, 1)  # N_ss


        # mix branch: [h_rho, h_sigma] -> optional MLP -> N_rs
        mix_in_dim = rho_out_dim + sigma_out_dim + 1 + 1 # Adds N_r, N_s
        self.branch_mix = FlexibleMLP(
            in_dim=mix_in_dim,
            hidden_dims=mix_hidden_dims,
            activation=activation,
        )
        mix_out_dim = self.branch_mix.out_dim
        self.head_rs = LinearHead(mix_out_dim, 1)  # N_rs

    def forward(self, t_in, rho_in, sigma_in):
        x = torch.cat([t_in, rho_in, sigma_in], dim=-1)

        # shared stem
        h = self.stem(x)
        
        # u branch
        h_u = self.u_branch(h)
        N   = self.head_N(h_u)

        # rho branch
        h_rho = self.branch_rho(h)
        N_r   = self.head_rho(h_rho)
        N_rr  = self.head_rr(h_rho)

        # sigma branch
        h_sigma = self.branch_sigma(h)
        N_s     = self.head_sigma(h_sigma)
        N_ss    = self.head_ss(h_sigma)

        # mix branch
        h_mix_in = torch.cat([h_rho, h_sigma, N_r, N_s], dim=-1)
        h_mix    = self.branch_mix(h_mix_in)
        N_rs     = self.head_rs(h_mix)

        return torch.cat([N, N_r, N_s, N_rr, N_ss, N_rs], dim=-1)


# ============================================================
# WrappedPotentialNet
# ============================================================

class WrappedPotentialNet(nn.Module):
    def __init__(self, base_net, T_min, T_max, Rho_min, Rho_max, Sigma_min, Sigma_max):
        super().__init__()
        self.base = base_net
        self.T_min = T_min
        self.T_max = T_max
        self.Rho_min = Rho_min
        self.Rho_max = Rho_max
        self.Sigma_min = Sigma_min
        self.Sigma_max = Sigma_max

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

    def scale_sigma(self, sigma_phys):
        return 2.0 * (sigma_phys - self.Sigma_min) / (self.Sigma_max - self.Sigma_min) - 1.0

    def unscale_sigma(self, sigma_in):
        return 0.5 * (sigma_in + 1.0) * (self.Sigma_max - self.Sigma_min) + self.Sigma_min

    @property
    def A_SIGMA(self):
        return 2.0 / (self.Sigma_max - self.Sigma_min)

    def forward(self, t_in, rho_in, sigma_in):
        t_phys = self.unscale_t(t_in)
        rho_phys = self.unscale_rho(rho_in)
        sigma_phys = self.unscale_sigma(sigma_in)

        out = self.base(t_in, rho_in, sigma_in)
        N    = out[:, 0:1]
        N_r  = out[:, 1:2]
        N_s  = out[:, 2:3]
        N_rr = out[:, 3:4]
        N_ss = out[:, 4:5]
        N_rs = out[:, 5:6]

        muH2_t = aH * torch.exp(-2.0 * t_phys)
        muS2_t = aS * torch.exp(-2.0 * t_phys)

        if finite_T:
            kh1, kh2, ks1, ks2, khs = 1.0, 1.0, 0.8, 0.8, 0.8
        else:
            kh1, kh2, ks1, ks2, khs = 0.8, 1.2, 0.8, 0.8, 0.8

        u_tree = (
            muH2_t * rho_phys * kh1
            + lamH * rho_phys**2 * kh2
            + muS2_t * sigma_phys * ks1
            + lamS * sigma_phys**2 * ks2
            + lamHS * rho_phys * sigma_phys * khs
        )

        ur_tree  = muH2_t * kh1 + 2.0 * lamH * rho_phys * kh2 + lamHS * sigma_phys * khs
        us_tree  = muS2_t * ks1 + 2.0 * lamS * sigma_phys * ks2 + lamHS * rho_phys * khs
        urr_tree = 2.0 * lamH * kh2
        uss_tree = 2.0 * lamS * ks2
        urs_tree = lamHS * khs

        tau = tau_uv * torch.exp(-t_phys)
        _kh, _ks = (0.2, 0.1)

        if finite_T:
            u_th = _kh * rho_phys * tau**2 + _ks * sigma_phys * tau**2
            ur_th = _kh * (tau**2)
            us_th = _ks * (tau**2)
        else:
            u_th = torch.zeros_like(rho_phys)
            ur_th = torch.zeros_like(rho_phys)
            us_th = torch.zeros_like(rho_phys)

        u   = u_tree + u_th + N
        ur  = ur_tree + ur_th + N_r
        us  = us_tree + us_th + N_s
        urr = urr_tree + N_rr
        uss = uss_tree + N_ss
        urs = urs_tree + N_rs

        return u, ur, us, urr, uss, urs
    
