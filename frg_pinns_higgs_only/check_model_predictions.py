import torch
import numpy as np
import torch.autograd as autograd


def check_model_predictions(model, device, tree_params):

    [aH, lamH] = tree_params
    n_samp = int(3)

    t_in = torch.empty(n_samp, 1, device=device).uniform_(-1.0, 1.0)
    rho_in = torch.empty(n_samp, 1, device=device).uniform_(-1.0, 1.0)

    t_in.requires_grad_(True)
    rho_in.requires_grad_(True)

    # physical coordinates
    t_phys = model.unscale_t(t_in)
    rho_phys = model.unscale_rho(rho_in)

    u, ur, urr = model(t_in, rho_in)

    u_rr = urr.clone().detach().cpu().numpy().flatten()

    mHsq = ((ur - urr * rho_phys)*torch.exp(2.0*t_phys)).clone().detach().cpu().numpy().flatten()
    print(u_rr/(2.0*lamH), (mHsq-aH)/np.abs(aH))
