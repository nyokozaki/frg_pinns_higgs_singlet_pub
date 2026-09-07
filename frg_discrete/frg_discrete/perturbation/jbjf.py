"""
frg_pinns_higgs_singlet/perturbation/JBJF_helper.py の J_Bnp / J_Fnp を
そのまま抜き出した、NumPyのみの有限温度熱積分。torch版 (J_B, J_F) はここでは
不要なので含めていない。

argument: y2 = (m/T)^2  (m, T ともに無次元)
"""

import numpy as np


def J_Bnp(y2, deg=100, x_max=20.0, epsilon=1e-6, alpha=1.0):
    y2 = np.asarray(y2)
    shape_orig = y2.shape
    y2_flat = y2.ravel()

    t, w = np.polynomial.legendre.leggauss(deg)
    t = t.reshape(1, -1)
    w = w.reshape(1, -1)

    result = np.zeros_like(y2_flat, dtype=float)

    mask_pos = y2_flat >= 0
    mask_neg = ~mask_pos

    # Case 1: y2 >= 0
    if np.any(mask_pos):
        y2_p = y2_flat[mask_pos].reshape(-1, 1)
        u_min, u_max = 0.0, np.sqrt(x_max)

        u = 0.5 * (u_max - u_min) * t + 0.5 * (u_max + u_min)
        weights_u = 0.5 * (u_max - u_min) * w

        x = u ** 2
        weights = weights_u * (2.0 * u)

        E2 = np.maximum(x ** 2 + y2_p, 1e-12)
        E = np.sqrt(E2)

        integrand = x ** 2 * np.log1p(-np.exp(-E))
        result[mask_pos] = np.sum(integrand * weights, axis=1)

    # Case 2: y2 < 0
    if np.any(mask_neg):
        y2_n = y2_flat[mask_neg].reshape(-1, 1)
        xc = np.sqrt(-y2_n)  # E=0 となる特異点

        x_upper = np.maximum(0.0, xc - epsilon)
        u_min1, u_max1 = 0.0, np.sqrt(x_upper)

        u1 = 0.5 * (u_max1 - u_min1) * t + 0.5 * (u_max1 + u_min1)
        weights_u1 = 0.5 * (u_max1 - u_min1) * w
        x1 = u1 ** 2
        weights1 = weights_u1 * (2.0 * u1)

        E2_1 = x1 ** 2 + y2_n
        abs_E = np.sqrt(np.maximum(-E2_1, 1e-12))

        val_B = np.maximum(2.0 * (1.0 - np.cos(abs_E)), 1e-12)
        integrand1 = x1 ** 2 * 0.5 * np.log(val_B)
        int1 = np.sum(integrand1 * weights1, axis=1)

        x_lower = np.minimum(xc + alpha * epsilon, x_max)
        u_min2, u_max2 = np.sqrt(x_lower), np.sqrt(x_max)

        u2 = 0.5 * (u_max2 - u_min2) * t + 0.5 * (u_max2 + u_min2)
        weights_u2 = 0.5 * (u_max2 - u_min2) * w
        x2 = u2 ** 2
        weights2 = weights_u2 * (2.0 * u2)

        E2_2 = np.maximum(x2 ** 2 + y2_n, 1e-12)
        E2_real = np.sqrt(E2_2)

        integrand2 = x2 ** 2 * np.log1p(-np.exp(-E2_real))
        int2 = np.sum(integrand2 * weights2, axis=1)

        result[mask_neg] = int1 + int2

    return result.reshape(shape_orig)


def J_Fnp(y2, deg=100, x_max=20.0, epsilon=1e-6, alpha=1.0):
    y2 = np.asarray(y2)
    shape_orig = y2.shape
    y2_flat = y2.ravel()

    t, w = np.polynomial.legendre.leggauss(deg)
    t = t.reshape(1, -1)
    w = w.reshape(1, -1)

    result = np.zeros_like(y2_flat, dtype=float)

    mask_pos = y2_flat >= 0
    mask_neg = ~mask_pos

    # Case 1: y2 >= 0
    if np.any(mask_pos):
        y2_p = y2_flat[mask_pos].reshape(-1, 1)
        u_min, u_max = 0.0, np.sqrt(x_max)

        u = 0.5 * (u_max - u_min) * t + 0.5 * (u_max + u_min)
        weights_u = 0.5 * (u_max - u_min) * w
        x = u ** 2
        weights = weights_u * (2.0 * u)

        E2 = np.maximum(x ** 2 + y2_p, 1e-12)
        E = np.sqrt(E2)

        integrand = x ** 2 * np.log1p(np.exp(-E))
        result[mask_pos] = np.sum(integrand * weights, axis=1)

    # Case 2: y2 < 0
    if np.any(mask_neg):
        y2_n = y2_flat[mask_neg].reshape(-1, 1)
        xc = np.sqrt(-y2_n)

        x_upper = np.maximum(0.0, xc - epsilon)
        u_min1, u_max1 = 0.0, np.sqrt(x_upper)

        u1 = 0.5 * (u_max1 - u_min1) * t + 0.5 * (u_max1 + u_min1)
        weights_u1 = 0.5 * (u_max1 - u_min1) * w
        x1 = u1 ** 2
        weights1 = weights_u1 * (2.0 * u1)

        E2_1 = x1 ** 2 + y2_n
        abs_E = np.sqrt(np.maximum(-E2_1, 1e-12))

        val_F = np.maximum(2.0 * (1.0 + np.cos(abs_E)), 1e-12)
        integrand1 = x1 ** 2 * 0.5 * np.log(val_F)
        int1 = np.sum(integrand1 * weights1, axis=1)

        x_lower = np.minimum(xc + alpha * epsilon, x_max)
        u_min2, u_max2 = np.sqrt(x_lower), np.sqrt(x_max)

        u2 = 0.5 * (u_max2 - u_min2) * t + 0.5 * (u_max2 + u_min2)
        weights_u2 = 0.5 * (u_max2 - u_min2) * w
        x2 = u2 ** 2
        weights2 = weights_u2 * (2.0 * u2)

        E2_2 = np.maximum(x2 ** 2 + y2_n, 1e-12)
        E2_real = np.sqrt(E2_2)

        integrand2 = x2 ** 2 * np.log1p(np.exp(-E2_real))
        int2 = np.sum(integrand2 * weights2, axis=1)

        result[mask_neg] = int1 + int2

    return result.reshape(shape_orig)
