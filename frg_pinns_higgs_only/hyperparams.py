#w_mass_rho = 1.0 # for rho mass term in loss sign
w_mass_rho = 1.0 # for rho mass term in loss sign, try small value for finite T
w_sign_r2 = 1.0 # for rho self-coupling in loss sign
w_mass_mag_rho = 1.0 # for rho mass term in loss mag
w_mag_r2 = 1.0 # for rho self coupliing term in loss mag

rho_cw_cut = 0.6

w_pde = 1.0
w_consist = 1.0
w_bc1 = 1.0
w_bc2 = 1.0
w_bc1_diff2 = 0
w_bc2_diff2 = 0
w_weak = 0
#w_overlap = 1e-3
w_overlap = 5e-2
w_sign =1000.0
w_mag = 10.0
epoch_1to2 = 1.0 # for the first block of t
uv_cw = 0.0 # add to cw at UV or not
#eta_min = 3e-8
#eta_min = 5e-6 # learning rate is going down to this value
eta_min = 1e-5 # learning rate is going down to this value
c_mag_lower = 0.4
c_mag_upper = 3.0
hinge_beta = 100.0 # softplus sharpness for sign/mag hinges in loss_extensions.py (higher = closer to ReLU)
skip_2nd = False # if True, train only the UV t-block (skip the IR half)
do_extra=True
non_resume = False
#non_resume = True 
device = "cuda"
#device = "cpu"
