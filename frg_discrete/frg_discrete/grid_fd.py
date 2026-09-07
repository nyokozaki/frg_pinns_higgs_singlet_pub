"""
(rho, sigma) 場空間上の一様格子と、有限差分による微分演算子。

2種類の1階微分スキームを用意する:
  - central : 内部は2次精度中心差分、境界は2次精度片側差分。
              質量固有値・熱閾値関数 (反応項) の評価に使う分には問題ない。
  - upwind  : 1次精度の片側差分 (下流バイアス)。
              フロー方程式の (2+eta_rho)*rho*u_rho, (2+eta_sigma)*sigma*u_sigma
              は移流項であり、その"速度"は rho,sigma>=0 の物理領域で常に
              符号が固定されている ( d/dtau [tau=-t] で見ると advection速度
              (2+eta)*rho, (2+eta)*sigma は常に非負)。中心差分+explicit RK
              の組み合わせは移流項に対して数値的に不安定になりやすく、
              格子を細かくするほど安定に必要な時間刻みが小さくなって
              solve_ivp の適応刻み幅が破綻する。upwind (風上) 差分は
              人工粘性を持ち、この不安定性を抑える。

2階微分 (u_rhorho, u_sigmasigma, u_rhosigma) は反応項(質量固有値)にしか
現れず移流的な不安定性の原因にならないため、常に中心差分を用いる。
"""

import numpy as np


def _fd_coeffs(offsets, deriv_order):
    """整数offsets (格子点からの相対位置) を使った有限差分係数を、Fornberg法
    (未定係数法) で厳密に求める。

    sum_k c_k * f(x+offsets_k*h) = h^deriv_order * f^(deriv_order)(x) + O(h^len(offsets))
    となる c を返す (m=len(offsets)点のステンシルで、次数 m-deriv_order 精度)。
    """
    offsets = np.asarray(offsets, dtype=float)
    m = len(offsets)
    M = np.vstack([offsets ** p for p in range(m)])
    rhs = np.zeros(m)
    import math
    rhs[deriv_order] = math.factorial(deriv_order)
    return np.linalg.solve(M, rhs)


class Grid2D:
    def __init__(self, rho_max, sigma_max, n_rho, n_sigma, rho_min=0.0, sigma_min=0.0):
        if n_rho < 4 or n_sigma < 4:
            raise ValueError("n_rho, n_sigma must be >= 4 for the boundary stencils used here.")

        self.rho = np.linspace(rho_min, rho_max, n_rho)
        self.sigma = np.linspace(sigma_min, sigma_max, n_sigma)
        self.drho = self.rho[1] - self.rho[0]
        self.dsigma = self.sigma[1] - self.sigma[0]
        self.n_rho = n_rho
        self.n_sigma = n_sigma

        # indexing='ij' -> axis 0 は rho, axis 1 は sigma
        self.RHO, self.SIGMA = np.meshgrid(self.rho, self.sigma, indexing='ij')
        self.shape = self.RHO.shape

    # ------------------------------------------------------------
    # 1D スタンシル (軸に沿った1階・2階微分, 2次精度)
    # ------------------------------------------------------------
    @staticmethod
    def _first_deriv_1d(U, axis, h):
        n = U.shape[axis]
        d = np.empty_like(U)

        s = [slice(None)] * U.ndim

        def sl(i):
            s2 = list(s)
            s2[axis] = i
            return tuple(s2)

        # interior: central difference
        d[sl(slice(1, n - 1))] = (U[sl(slice(2, n))] - U[sl(slice(0, n - 2))]) / (2.0 * h)

        # left edge: 2nd-order forward difference
        d[sl(0)] = (-3.0 * U[sl(0)] + 4.0 * U[sl(1)] - U[sl(2)]) / (2.0 * h)

        # right edge: 2nd-order backward difference
        d[sl(n - 1)] = (3.0 * U[sl(n - 1)] - 4.0 * U[sl(n - 2)] + U[sl(n - 3)]) / (2.0 * h)

        return d

    @staticmethod
    def _second_deriv_1d(U, axis, h):
        n = U.shape[axis]
        d = np.empty_like(U)

        s = [slice(None)] * U.ndim

        def sl(i):
            s2 = list(s)
            s2[axis] = i
            return tuple(s2)

        # interior: central second difference
        d[sl(slice(1, n - 1))] = (
            U[sl(slice(2, n))] - 2.0 * U[sl(slice(1, n - 1))] + U[sl(slice(0, n - 2))]
        ) / h ** 2

        # left edge: 2nd-order forward second-derivative stencil
        d[sl(0)] = (2.0 * U[sl(0)] - 5.0 * U[sl(1)] + 4.0 * U[sl(2)] - U[sl(3)]) / h ** 2

        # right edge: 2nd-order backward second-derivative stencil
        d[sl(n - 1)] = (
            2.0 * U[sl(n - 1)] - 5.0 * U[sl(n - 2)] + 4.0 * U[sl(n - 3)] - U[sl(n - 4)]
        ) / h ** 2

        return d

    _ORDER4_OFFSETS = {
        0: [0, 1, 2, 3, 4],
        1: [-1, 0, 1, 2, 3],
        "mid": [-2, -1, 0, 1, 2],
        -2: [-3, -2, -1, 0, 1],
        -1: [-4, -3, -2, -1, 0],
    }

    @classmethod
    def _deriv_1d_order4(cls, U, axis, h, deriv_order):
        """4次精度 (5点ステンシル、Fornberg法で係数導出) の1階/2階微分。
        内部は対称5点、境界寄り2点は非対称5点ステンシルを使う。"""
        n = U.shape[axis]
        if n < 5:
            raise ValueError("scheme='central4' には各方向5点以上の格子が必要です")
        d = np.zeros_like(U)

        s = [slice(None)] * U.ndim

        def sl(i):
            s2 = list(s)
            s2[axis] = i
            return tuple(s2)

        def add(i, offsets):
            c = _fd_coeffs(offsets, deriv_order) / h ** deriv_order
            for off, cv in zip(offsets, c):
                d[sl(i)] = d[sl(i)] + cv * U[sl(i + int(off))]

        add(0, cls._ORDER4_OFFSETS[0])
        add(1, cls._ORDER4_OFFSETS[1])
        for i in range(2, n - 2):
            add(i, cls._ORDER4_OFFSETS["mid"])
        add(n - 2, cls._ORDER4_OFFSETS[-2])
        add(n - 1, cls._ORDER4_OFFSETS[-1])
        return d

    @staticmethod
    def _first_deriv_upwind_1d(U, axis, h):
        """
        1次精度、下流(backward)バイアスの片側差分。
        advection速度が常に非負 (rho,sigma>=0 の物理領域) であることを前提に、
        内部点・右境界は backward difference (i, i-1)、
        左境界 (速度=0 の停留点) だけ forward difference を使う。
        """
        n = U.shape[axis]
        d = np.empty_like(U)

        s = [slice(None)] * U.ndim

        def sl(i):
            s2 = list(s)
            s2[axis] = i
            return tuple(s2)

        # interior + right edge: backward difference
        d[sl(slice(1, n))] = (U[sl(slice(1, n))] - U[sl(slice(0, n - 1))]) / h

        # left edge (velocity = 0 の停留点): forward difference
        d[sl(0)] = (U[sl(1)] - U[sl(0)]) / h

        return d

    # ------------------------------------------------------------
    # 公開インターフェース
    # ------------------------------------------------------------
    def d_drho(self, U, scheme="central"):
        if scheme == "central":
            return self._first_deriv_1d(U, axis=0, h=self.drho)
        elif scheme == "central4":
            return self._deriv_1d_order4(U, axis=0, h=self.drho, deriv_order=1)
        elif scheme == "upwind":
            return self._first_deriv_upwind_1d(U, axis=0, h=self.drho)
        raise ValueError(f"unknown scheme: {scheme}")

    def d_dsigma(self, U, scheme="central"):
        if scheme == "central":
            return self._first_deriv_1d(U, axis=1, h=self.dsigma)
        elif scheme == "central4":
            return self._deriv_1d_order4(U, axis=1, h=self.dsigma, deriv_order=1)
        elif scheme == "upwind":
            return self._first_deriv_upwind_1d(U, axis=1, h=self.dsigma)
        raise ValueError(f"unknown scheme: {scheme}")

    def d2_drho2(self, U, scheme="central"):
        if scheme == "central4":
            return self._deriv_1d_order4(U, axis=0, h=self.drho, deriv_order=2)
        return self._second_deriv_1d(U, axis=0, h=self.drho)

    def d2_dsigma2(self, U, scheme="central"):
        if scheme == "central4":
            return self._deriv_1d_order4(U, axis=1, h=self.dsigma, deriv_order=2)
        return self._second_deriv_1d(U, axis=1, h=self.dsigma)

    def d2_drhodsigma(self, U, scheme="central"):
        # d/dsigma ( d/drho U )
        if scheme == "central4":
            return self._deriv_1d_order4(
                self.d_drho(U, scheme=scheme), axis=1, h=self.dsigma, deriv_order=1
            )
        return self._first_deriv_1d(
            self.d_drho(U, scheme=scheme), axis=1, h=self.dsigma
        ) if scheme == "central" else self._first_deriv_upwind_1d(
            self.d_drho(U, scheme=scheme), axis=1, h=self.dsigma
        )

    def derivatives(self, U, scheme="central"):
        """
        U(rho, sigma) から (u_rho, u_sigma, u_rhorho, u_sigmasigma, u_rhosigma) を返す。

        scheme="central" : 全て2次精度中心差分 (精度重視、高解像度では不安定になりうる)
        scheme="central4": 全て4次精度中心差分 (5点ステンシル, Fornberg法。
                             内部+境界寄り2点とも4次精度。各方向5点以上必要)
        scheme="upwind"   : 1階微分 (u_rho, u_sigma, u_rhosigma) を1次精度upwindに、
                             2階微分 (u_rhorho, u_sigmasigma) は中心差分のまま
                             (移流項の安定化。flow_equation.py 参照)
        """
        u_rho = self.d_drho(U, scheme=scheme)
        u_sigma = self.d_dsigma(U, scheme=scheme)
        u_rhorho = self.d2_drho2(U, scheme=scheme)
        u_sigmasigma = self.d2_dsigma2(U, scheme=scheme)
        u_rhosigma = self.d2_drhodsigma(U, scheme=scheme)
        return u_rho, u_sigma, u_rhorho, u_sigmasigma, u_rhosigma
