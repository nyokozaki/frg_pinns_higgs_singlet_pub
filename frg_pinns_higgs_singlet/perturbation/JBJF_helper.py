import numpy as np
import torch


# ============================================================
# 有限温度ヘルパー: 高精度化された Thermal integrals J_B, J_F (特異点回避版)
# ============================================================
def J_Bnp(y2, deg=100, x_max=20.0, epsilon=1e-6, alpha=1.0):
    y2 = np.asarray(y2)
    shape_orig = y2.shape
    y2_flat = y2.ravel()

    # ガウス・ルジャンドル求積法の分点と重み
    t, w = np.polynomial.legendre.leggauss(deg)
    t = t.reshape(1, -1)
    w = w.reshape(1, -1)

    result = np.zeros_like(y2_flat, dtype=float)

    # 正負のマスクを作成
    mask_pos = y2_flat >= 0
    mask_neg = ~mask_pos

    # ==========================================
    # Case 1: y2 >= 0 (従来の積分範囲 0 to x_max)
    # ==========================================
    if np.any(mask_pos):
        y2_p = y2_flat[mask_pos].reshape(-1, 1)
        u_min, u_max = 0.0, np.sqrt(x_max)
        
        u = 0.5 * (u_max - u_min) * t + 0.5 * (u_max + u_min)
        weights_u = 0.5 * (u_max - u_min) * w
        
        x = u**2
        weights = weights_u * (2.0 * u)

        E2 = np.maximum(x**2 + y2_p, 1e-12)
        E = np.sqrt(E2)
        
        integrand = x**2 * np.log1p(-np.exp(-E))
        result[mask_pos] = np.sum(integrand * weights, axis=1)

    # ==========================================
    # Case 2: y2 < 0 (積分範囲を2つに分割)
    # ==========================================
    if np.any(mask_neg):
        y2_n = y2_flat[mask_neg].reshape(-1, 1)
        xc = np.sqrt(-y2_n) # E=0 となる特異点

        # --- 範囲1: 0 から xc - epsilon ---
        x_upper = np.maximum(0.0, xc - epsilon)
        u_min1, u_max1 = 0.0, np.sqrt(x_upper)
        
        u1 = 0.5 * (u_max1 - u_min1) * t + 0.5 * (u_max1 + u_min1)
        weights_u1 = 0.5 * (u_max1 - u_min1) * w
        x1 = u1**2
        weights1 = weights_u1 * (2.0 * u1)

        E2_1 = x1**2 + y2_n  # 負の値
        abs_E = np.sqrt(np.maximum(-E2_1, 1e-12))
        
        # ボソンの厳密な実部計算: 0.5 * ln(2(1 - cos|E|))
        val_B = np.maximum(2.0 * (1.0 - np.cos(abs_E)), 1e-12)
        integrand1 = x1**2 * 0.5 * np.log(val_B)
        int1 = np.sum(integrand1 * weights1, axis=1)

        # --- 範囲2: xc + alpha*epsilon から x_max ---
        x_lower = np.minimum(xc + alpha * epsilon, x_max)
        u_min2, u_max2 = np.sqrt(x_lower), np.sqrt(x_max)
        
        u2 = 0.5 * (u_max2 - u_min2) * t + 0.5 * (u_max2 + u_min2)
        weights_u2 = 0.5 * (u_max2 - u_min2) * w
        x2 = u2**2
        weights2 = weights_u2 * (2.0 * u2)

        E2_2 = np.maximum(x2**2 + y2_n, 1e-12)
        E2_real = np.sqrt(E2_2)
        
        integrand2 = x2**2 * np.log1p(-np.exp(-E2_real))
        int2 = np.sum(integrand2 * weights2, axis=1)

        # 範囲1と範囲2を合算
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

    # ==========================================
    # Case 1: y2 >= 0
    # ==========================================
    if np.any(mask_pos):
        y2_p = y2_flat[mask_pos].reshape(-1, 1)
        u_min, u_max = 0.0, np.sqrt(x_max)
        
        u = 0.5 * (u_max - u_min) * t + 0.5 * (u_max + u_min)
        weights_u = 0.5 * (u_max - u_min) * w
        x = u**2
        weights = weights_u * (2.0 * u)

        E2 = np.maximum(x**2 + y2_p, 1e-12)
        E = np.sqrt(E2)
        
        integrand = x**2 * np.log1p(np.exp(-E))
        result[mask_pos] = np.sum(integrand * weights, axis=1)

    # ==========================================
    # Case 2: y2 < 0
    # ==========================================
    if np.any(mask_neg):
        y2_n = y2_flat[mask_neg].reshape(-1, 1)
        xc = np.sqrt(-y2_n)

        # --- 範囲1: 0 から xc - epsilon ---
        x_upper = np.maximum(0.0, xc - epsilon)
        u_min1, u_max1 = 0.0, np.sqrt(x_upper)
        
        u1 = 0.5 * (u_max1 - u_min1) * t + 0.5 * (u_max1 + u_min1)
        weights_u1 = 0.5 * (u_max1 - u_min1) * w
        x1 = u1**2
        weights1 = weights_u1 * (2.0 * u1)

        E2_1 = x1**2 + y2_n
        abs_E = np.sqrt(np.maximum(-E2_1, 1e-12))
        
        # フェルミオンの厳密な実部計算: 0.5 * ln(2(1 + cos|E|))
        val_F = np.maximum(2.0 * (1.0 + np.cos(abs_E)), 1e-12)
        integrand1 = x1**2 * 0.5 * np.log(val_F)
        int1 = np.sum(integrand1 * weights1, axis=1)

        # --- 範囲2: xc + alpha*epsilon から x_max ---
        x_lower = np.minimum(xc + alpha * epsilon, x_max)
        u_min2, u_max2 = np.sqrt(x_lower), np.sqrt(x_max)
        
        u2 = 0.5 * (u_max2 - u_min2) * t + 0.5 * (u_max2 + u_min2)
        weights_u2 = 0.5 * (u_max2 - u_min2) * w
        x2 = u2**2
        weights2 = weights_u2 * (2.0 * u2)

        E2_2 = np.maximum(x2**2 + y2_n, 1e-12)
        E2_real = np.sqrt(E2_2)
        
        integrand2 = x2**2 * np.log1p(np.exp(-E2_real))
        int2 = np.sum(integrand2 * weights2, axis=1)

        result[mask_neg] = int1 + int2

    return result.reshape(shape_orig)



# ============================================================
# Gauss-Legendre constants
# ============================================================

def make_leggauss_constants(deg, device="cpu", dtype=torch.float64, copy=True):
    """
    Create Gauss-Legendre nodes and weights on [-1, 1] only once.

    Parameters
    ----------
    deg : int
        Degree / number of quadrature points.
    device : str or torch.device
        Where to place the returned tensors initially.
    dtype : torch.dtype
        Tensor dtype.
    copy : bool
        If True, use torch.tensor(...) to make an explicit copy.
        If False, use torch.from_numpy(...) (shared memory with numpy array).

    Returns
    -------
    t : torch.Tensor, shape (1, deg)
        Gauss-Legendre nodes in [-1, 1].
    w : torch.Tensor, shape (1, deg)
        Gauss-Legendre weights.
    """
    t_np, w_np = np.polynomial.legendre.leggauss(deg)

    if copy:
        t = torch.tensor(t_np, device=device, dtype=dtype).view(1, -1)
        w = torch.tensor(w_np, device=device, dtype=dtype).view(1, -1)
    else:
        t = torch.from_numpy(t_np).to(device=device, dtype=dtype).view(1, -1)
        w = torch.from_numpy(w_np).to(device=device, dtype=dtype).view(1, -1)

    return t, w


def get_leggauss_for(y2, t_base, w_base):
    """
    Move precomputed Gauss-Legendre constants to y2.device / y2.dtype.
    """
    t = t_base.to(device=y2.device, dtype=y2.dtype)
    w = w_base.to(device=y2.device, dtype=y2.dtype)
    return t, w


# ============================================================
# Finite-T helper: thermal integrals J_B, J_F
# argument: y2 = (m/T)^2
# supports both y2 >= 0 and y2 < 0
# ============================================================

def J_B(y2, t, w, x_max=20.0, epsilon=1e-6, alpha=1.0):
    """
    Bosonic thermal integral:
      J_B(y2) = ∫_0^∞ dx x^2 log(1 - exp(-sqrt(x^2 + y2)))

    This implementation uses Gauss-Legendre quadrature after the change of variable
      x = u^2,
    and handles y2 < 0 by splitting the integration range around
      x_c = sqrt(-y2),
    where sqrt(x^2 + y2) = 0.

    For y2 < 0, in the region x^2 + y2 < 0, it uses the exact real part
      Re log(1 - exp(-i|E|)) = 0.5 * log( 2 * (1 - cos|E|) ).
    """
    y2 = torch.as_tensor(y2)
    shape_orig = y2.shape
    y2_flat = y2.reshape(-1)

    device = y2.device
    dtype = y2.dtype

    t = t.to(device=device, dtype=dtype)
    w = w.to(device=device, dtype=dtype)

    result = torch.zeros_like(y2_flat)

    mask_pos = (y2_flat >= 0)
    mask_neg = ~mask_pos

    x_max_t = torch.tensor(x_max, device=device, dtype=dtype)
    zero_t = torch.tensor(0.0, device=device, dtype=dtype)
    eps_E2 = torch.tensor(1e-12, device=device, dtype=dtype)

    # ==========================================
    # Case 1: y2 >= 0
    # ==========================================
    if mask_pos.any():
        y2_p = y2_flat[mask_pos].view(-1, 1)

        # map t in [-1,1] to u in [0, sqrt(x_max)]
        u_min = zero_t
        u_max = torch.sqrt(x_max_t)

        u = 0.5 * (u_max - u_min) * t + 0.5 * (u_max + u_min)
        weights_u = 0.5 * (u_max - u_min) * w

        # x = u^2, dx = 2u du
        x = u**2
        weights = weights_u * (2.0 * u)

        E2 = torch.clamp(x**2 + y2_p, min=eps_E2)
        E = torch.sqrt(E2)

        integrand = x**2 * torch.log1p(-torch.exp(-E))
        result[mask_pos] = torch.sum(integrand * weights, dim=1)

    # ==========================================
    # Case 2: y2 < 0
    # ==========================================
    if mask_neg.any():
        y2_n = y2_flat[mask_neg].view(-1, 1)
        xc = torch.sqrt(-y2_n)   # singular point: x^2 + y2 = 0

        # --- Region 1: 0 to xc - epsilon ---
        x_upper = torch.clamp(xc - epsilon, min=0.0)
        u_min1 = torch.zeros_like(x_upper)
        u_max1 = torch.sqrt(x_upper)

        u1 = 0.5 * (u_max1 - u_min1) * t + 0.5 * (u_max1 + u_min1)
        weights_u1 = 0.5 * (u_max1 - u_min1) * w

        x1 = u1**2
        weights1 = weights_u1 * (2.0 * u1)

        E2_1 = x1**2 + y2_n   # negative in region 1
        abs_E = torch.sqrt(torch.clamp(-E2_1, min=eps_E2))

        # Re log(1 - e^{-i|E|}) = 0.5 * log(2(1 - cos|E|))
        val_B = torch.clamp(2.0 * (1.0 - torch.cos(abs_E)), min=eps_E2)
        integrand1 = x1**2 * 0.5 * torch.log(val_B)
        int1 = torch.sum(integrand1 * weights1, dim=1)

        # --- Region 2: xc + alpha*epsilon to x_max ---
        x_lower = torch.clamp(xc + alpha * epsilon, max=x_max)
        u_min2 = torch.sqrt(x_lower)
        u_max2 = torch.full_like(u_min2, torch.sqrt(x_max_t))

        u2 = 0.5 * (u_max2 - u_min2) * t + 0.5 * (u_max2 + u_min2)
        weights_u2 = 0.5 * (u_max2 - u_min2) * w

        x2 = u2**2
        weights2 = weights_u2 * (2.0 * u2)

        E2_2 = torch.clamp(x2**2 + y2_n, min=eps_E2)
        E_real = torch.sqrt(E2_2)

        integrand2 = x2**2 * torch.log1p(-torch.exp(-E_real))
        int2 = torch.sum(integrand2 * weights2, dim=1)

        result[mask_neg] = int1 + int2

    return result.reshape(shape_orig)


def J_F(y2, t, w, x_max=20.0, epsilon=1e-6, alpha=1.0):
    """
    Fermionic thermal integral:
      J_F(y2) = ∫_0^∞ dx x^2 log(1 + exp(-sqrt(x^2 + y2)))

    This implementation uses Gauss-Legendre quadrature after the change of variable
      x = u^2,
    and handles y2 < 0 by splitting the integration range around
      x_c = sqrt(-y2).

    For y2 < 0, in the region x^2 + y2 < 0, it uses the exact real part
      Re log(1 + exp(-i|E|)) = 0.5 * log( 2 * (1 + cos|E|) ).
    """
    y2 = torch.as_tensor(y2)
    shape_orig = y2.shape
    y2_flat = y2.reshape(-1)

    device = y2.device
    dtype = y2.dtype

    t = t.to(device=device, dtype=dtype)
    w = w.to(device=device, dtype=dtype)

    result = torch.zeros_like(y2_flat)

    mask_pos = (y2_flat >= 0)
    mask_neg = ~mask_pos

    x_max_t = torch.tensor(x_max, device=device, dtype=dtype)
    zero_t = torch.tensor(0.0, device=device, dtype=dtype)
    eps_E2 = torch.tensor(1e-12, device=device, dtype=dtype)

    # ==========================================
    # Case 1: y2 >= 0
    # ==========================================
    if mask_pos.any():
        y2_p = y2_flat[mask_pos].view(-1, 1)

        u_min = zero_t
        u_max = torch.sqrt(x_max_t)

        u = 0.5 * (u_max - u_min) * t + 0.5 * (u_max + u_min)
        weights_u = 0.5 * (u_max - u_min) * w

        x = u**2
        weights = weights_u * (2.0 * u)

        E2 = torch.clamp(x**2 + y2_p, min=eps_E2)
        E = torch.sqrt(E2)

        integrand = x**2 * torch.log1p(torch.exp(-E))
        result[mask_pos] = torch.sum(integrand * weights, dim=1)

    # ==========================================
    # Case 2: y2 < 0
    # ==========================================
    if mask_neg.any():
        y2_n = y2_flat[mask_neg].view(-1, 1)
        xc = torch.sqrt(-y2_n)

        # --- Region 1: 0 to xc - epsilon ---
        x_upper = torch.clamp(xc - epsilon, min=0.0)
        u_min1 = torch.zeros_like(x_upper)
        u_max1 = torch.sqrt(x_upper)

        u1 = 0.5 * (u_max1 - u_min1) * t + 0.5 * (u_max1 + u_min1)
        weights_u1 = 0.5 * (u_max1 - u_min1) * w

        x1 = u1**2
        weights1 = weights_u1 * (2.0 * u1)

        E2_1 = x1**2 + y2_n
        abs_E = torch.sqrt(torch.clamp(-E2_1, min=eps_E2))

        # Re log(1 + e^{-i|E|}) = 0.5 * log(2(1 + cos|E|))
        val_F = torch.clamp(2.0 * (1.0 + torch.cos(abs_E)), min=eps_E2)
        integrand1 = x1**2 * 0.5 * torch.log(val_F)
        int1 = torch.sum(integrand1 * weights1, dim=1)

        # --- Region 2: xc + alpha*epsilon to x_max ---
        x_lower = torch.clamp(xc + alpha * epsilon, max=x_max)
        u_min2 = torch.sqrt(x_lower)
        u_max2 = torch.full_like(u_min2, torch.sqrt(x_max_t))

        u2 = 0.5 * (u_max2 - u_min2) * t + 0.5 * (u_max2 + u_min2)
        weights_u2 = 0.5 * (u_max2 - u_min2) * w

        x2 = u2**2
        weights2 = weights_u2 * (2.0 * u2)

        E2_2 = torch.clamp(x2**2 + y2_n, min=eps_E2)
        E_real = torch.sqrt(E2_2)

        integrand2 = x2**2 * torch.log1p(torch.exp(-E_real))
        int2 = torch.sum(integrand2 * weights2, dim=1)

        result[mask_neg] = int1 + int2

    return result.reshape(shape_orig)