w_mass_rho = 0.0 # for rho mass term in loss sign
# w_sign_r2/w_sign_s2 (below) and w_mag_r2_hi/w_mag_s2_hi (further below) only
# drive terms that exist in the rho_cw_cut/sigma_cw_cut "hi" region added this
# session (see loss_extensions.py) and have NO counterpart in
# previous_ReLU_logic/ (there the equivalent domain-wide D_H/D_S sign block is
# commented out, and quartic self-coupling only has the flat 0.99 cap below,
# no RGE-target band at all). Kept at 0.0 so the active finite-T branch reduces
# to previous_ReLU_logic's finite-T branch with torch.relu -> _hinge (softplus)
# as the only difference — don't set these nonzero without re-deriving what
# they'd add on top of that baseline.
w_sign_r2 = 0 # for rho self-coupling in loss sign (finite T: rho > rho_cw_cut region only; see note above)
w_mass_mag_rho = 0 # for rho mass term in loss mag
w_mag_r2 = 0.1 # for rho self coupliing term in loss mag
# finite T: w_mag_r2 is the domain-wide flat cap |D_H_NN/lamH|<0.99 only (this
# term DOES exist in previous_ReLU_logic, same formula); the rho > rho_cw_cut
# band vs the RGE target uses the separate w_mag_r2_hi below (see note above).
w_mag_r2_hi = 0 # for rho quartic vs RGE-target band, rho > rho_cw_cut region only (finite T; no zero-T or previous_ReLU_logic counterpart — see note above)

w_mass_sigma = 1.0 # for sigma mass terms in loss sign
w_sign_s2 = 0.0 # for sigma self-coupling in loss sign (finite T: sigma > sigma_cw_cut region only; see note above)
w_mass_mag_sigma = 2.0 # for sigma mass term in loss mag
w_mag_s2 = 1.0 # for sigma self coupliing term in loss mag
w_mag_s2_hi = 1.0 # for sigma quartic vs RGE-target band, sigma > sigma_cw_cut region only (finite T; no zero-T or previous_ReLU_logic counterpart — see note above)

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
