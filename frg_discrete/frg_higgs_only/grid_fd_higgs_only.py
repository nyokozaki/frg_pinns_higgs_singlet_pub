"""
Uniform 1D grid in rho and finite-difference derivative operators.

The version of frg_discrete/grid_fd.py's Grid2D with the sigma direction
removed (no singlet field, so the only space dimension is rho). The three
schemes central/central4/upwind and the reasons for choosing between them are
the same as in the header comment of grid_fd.py.
"""

import numpy as np


def _fd_coeffs(offsets, deriv_order):
    offsets = np.asarray(offsets, dtype=float)
    m = len(offsets)
    M = np.vstack([offsets ** p for p in range(m)])
    rhs = np.zeros(m)
    import math
    rhs[deriv_order] = math.factorial(deriv_order)
    return np.linalg.solve(M, rhs)


class Grid1D:
    def __init__(self, rho_max, n_rho, rho_min=0.0):
        if n_rho < 4:
            raise ValueError("n_rho must be >= 4 for the boundary stencils used here.")

        self.rho = np.linspace(rho_min, rho_max, n_rho)
        self.drho = self.rho[1] - self.rho[0]
        self.n_rho = n_rho

        self.RHO = self.rho.copy()
        self.shape = self.RHO.shape

    # ------------------------------------------------------------
    # 1D stencils (2nd order)
    # ------------------------------------------------------------
    @staticmethod
    def _first_deriv_1d(U, h):
        n = U.shape[0]
        d = np.empty_like(U)
        d[1:n - 1] = (U[2:n] - U[0:n - 2]) / (2.0 * h)
        d[0] = (-3.0 * U[0] + 4.0 * U[1] - U[2]) / (2.0 * h)
        d[n - 1] = (3.0 * U[n - 1] - 4.0 * U[n - 2] + U[n - 3]) / (2.0 * h)
        return d

    @staticmethod
    def _second_deriv_1d(U, h):
        n = U.shape[0]
        d = np.empty_like(U)
        d[1:n - 1] = (U[2:n] - 2.0 * U[1:n - 1] + U[0:n - 2]) / h ** 2
        d[0] = (2.0 * U[0] - 5.0 * U[1] + 4.0 * U[2] - U[3]) / h ** 2
        d[n - 1] = (2.0 * U[n - 1] - 5.0 * U[n - 2] + 4.0 * U[n - 3] - U[n - 4]) / h ** 2
        return d

    _ORDER4_OFFSETS = {
        0: [0, 1, 2, 3, 4],
        1: [-1, 0, 1, 2, 3],
        "mid": [-2, -1, 0, 1, 2],
        -2: [-3, -2, -1, 0, 1],
        -1: [-4, -3, -2, -1, 0],
    }

    @classmethod
    def _deriv_1d_order4(cls, U, h, deriv_order):
        n = U.shape[0]
        if n < 5:
            raise ValueError("scheme='central4' requires at least 5 grid points")
        d = np.zeros_like(U)

        def add(i, offsets):
            c = _fd_coeffs(offsets, deriv_order) / h ** deriv_order
            for off, cv in zip(offsets, c):
                d[i] = d[i] + cv * U[i + int(off)]

        add(0, cls._ORDER4_OFFSETS[0])
        add(1, cls._ORDER4_OFFSETS[1])
        for i in range(2, n - 2):
            add(i, cls._ORDER4_OFFSETS["mid"])
        add(n - 2, cls._ORDER4_OFFSETS[-2])
        add(n - 1, cls._ORDER4_OFFSETS[-1])
        return d

    @staticmethod
    def _first_deriv_upwind_1d(U, h):
        n = U.shape[0]
        d = np.empty_like(U)
        d[1:n] = (U[1:n] - U[0:n - 1]) / h
        d[0] = (U[1] - U[0]) / h
        return d

    # ------------------------------------------------------------
    # Public interface
    # ------------------------------------------------------------
    def d_drho(self, U, scheme="central"):
        if scheme == "central":
            return self._first_deriv_1d(U, self.drho)
        elif scheme == "central4":
            return self._deriv_1d_order4(U, self.drho, deriv_order=1)
        elif scheme == "upwind":
            return self._first_deriv_upwind_1d(U, self.drho)
        raise ValueError(f"unknown scheme: {scheme}")

    def d2_drho2(self, U, scheme="central"):
        if scheme == "central4":
            return self._deriv_1d_order4(U, self.drho, deriv_order=2)
        return self._second_deriv_1d(U, self.drho)

    def derivatives(self, U, scheme="central"):
        """Return (u_rho, u_rhorho) from U(rho)."""
        u_rho = self.d_drho(U, scheme=scheme)
        u_rhorho = self.d2_drho2(U, scheme=scheme)
        return u_rho, u_rhorho
