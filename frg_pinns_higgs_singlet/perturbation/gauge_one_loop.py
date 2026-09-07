import numpy as np

def gauge_couplings_1loop(mu_target,
                          mZ=91.1876,
                          sin2_thetaW_MSbar_mZ=0.23121,
                          alpha_em_mZ_approx=1.0/128.0,
                          alpha_s_mZ=0.1179):
    """
    Return (gY(mu_target), g2(mu_target), g3(mu_target))
    using analytic 1-loop running from mu = mZ.

    Conventions:
      d g_i / d ln(mu) = (b_i / 16 pi^2) g_i^3

    Approximations:
      - gY, g2 use the usual SM 1-loop coefficients
      - g3 uses nf = 5, i.e. b3 = -23/3
      - gY is NOT GUT-normalized
    """

    # initial couplings at mZ
    sin_thetaW = np.sqrt(sin2_thetaW_MSbar_mZ)
    cos_thetaW = np.sqrt(1.0 - sin2_thetaW_MSbar_mZ)
    e_mZ = np.sqrt(4.0 * np.pi * alpha_em_mZ_approx)

    gY_mZ = e_mZ / cos_thetaW
    g2_mZ = e_mZ / sin_thetaW
    g3_mZ = np.sqrt(4.0 * np.pi * alpha_s_mZ)

    # 1-loop beta coefficients
    bY = 41.0 / 6.0
    b2 = -19.0 / 6.0
    b3 = -23.0 / 3.0

    def g_running_1loop(g0, b, mu0, mu):
        log_ratio = np.log(mu / mu0)
        inv_g2_mu = 1.0 / g0**2 - (b / (8.0 * np.pi**2)) * log_ratio
        return 1.0 / np.sqrt(inv_g2_mu)

    gY_mu = g_running_1loop(gY_mZ, bY, mZ, mu_target)
    g2_mu = g_running_1loop(g2_mZ, b2, mZ, mu_target)
    g3_mu = g_running_1loop(g3_mZ, b3, mZ, mu_target)

    return gY_mu, g2_mu, g3_mu