# Soft perturbative-consistency weights (see loss_extensions.py, paper Sec. 5.2 / Appendix A).
# w_sign_r2 / w_sign_s2 and w_mag_r2_hi / w_mag_s2_hi act only in the
# rho > rho_cw_cut / sigma > sigma_cw_cut regions. In the finite-T runs of the
# paper the Higgs mass term carries no sign constraint (w_mass_rho = w_sign_r2 =
# w_mass_mag_rho = 0); the sign hinge is applied only to the singlet mass term.
w_mass_rho = 0.0 # Higgs mass term, sign hinge
w_sign_r2 = 0    # Higgs quartic, sign hinge (rho > rho_cw_cut region only)
w_mass_mag_rho = 0 # Higgs mass term, magnitude band
w_mag_r2 = 0.1   # Higgs quartic, domain-wide flat cap |D_H_NN/lamH| < 0.99
w_mag_r2_hi = 0  # Higgs quartic vs RGE-target band (rho > rho_cw_cut region only)

w_mass_sigma = 1.0 # singlet mass term, sign hinge
w_sign_s2 = 0.0    # singlet quartic, sign hinge (sigma > sigma_cw_cut region only)
w_mass_mag_sigma = 2.0 # singlet mass term, magnitude band
w_mag_s2 = 1.0     # singlet quartic, domain-wide flat cap
w_mag_s2_hi = 1.0  # singlet quartic vs RGE-target band (sigma > sigma_cw_cut region only)

w_sign_mix = 1e-2 # for lamHS in loss_sign (not used in finite T)
#w_sign_mix = 0 # for lamHS in loss_sign (not used in finite T)
w_mag_mix = 1e-2 # for lamHS in loss_mag

rho_cw_cut = 3.0
sigma_cw_cut = 0.5

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
ws1 = 5.0 # weight of consistency loss for sinlget; 1st oder derivative term
ws2 = 5.0 # weight of consistency loss for sinlget; 2nd order derivative term
wrsmix = 5.0 # weight of consistency loss for singlet; mixing term
uv_cw = 0.0 # add to cw at UV or not
#eta_min = 3e-8
#eta_min = 5e-6 # learning rate is going down to this value
eta_min = 1e-5 # learning rate is going down to this value
c_mag_lower_rho = 0.1   # rho(Higgs)-side magnitude band lower bound (mass term F_H, quartic D_H)
c_mag_upper_rho = 3.0
c_mag_lower_sigma = 0.1 # sigma(singlet)-side magnitude band lower bound (mass term F_S, quartic D_S/D_HS)
c_mag_upper_sigma = 3.0
hinge_beta = 100.0 # softplus sharpness for sign/mag hinges in loss_extensions.py (higher = closer to ReLU)
skip_2nd = False # if True, train only the UV t-block (skip the IR half)
do_extra=True
non_resume = False
#non_resume = True 
device = "cuda"
#device = "cpu"
