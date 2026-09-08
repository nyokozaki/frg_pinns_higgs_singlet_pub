"""
Uniform grid on the (rho, sigma) field space and finite-difference derivative
operators.

Two first-derivative schemes are provided:
  - central : second-order central difference in the interior, second-order
              one-sided difference at the boundaries. Adequate for evaluating
              the mass eigenvalues and thermal threshold functions (the
              reaction terms).
  - upwind  : first-order one-sided difference (downstream bias). The terms
              (2+eta_rho)*rho*u_rho and (2+eta_sigma)*sigma*u_sigma of the flow
              equation are advection terms whose "velocity" has a fixed sign
              throughout the physical region rho,sigma>=0 (viewed in
              d/dtau [tau=-t], the advection velocities (2+eta)*rho and
              (2+eta)*sigma are always non-negative). Central differencing
              combined with explicit RK tends to be numerically unstable for
              advection terms, and the finer the grid the smaller the time step
              required for stability, which breaks the adaptive step control of
              solve_ivp. The upwind difference carries artificial viscosity and
              suppresses this instability.

The second derivatives (u_rhorho, u_sigmasigma, u_rhosigma) appear only in the
reaction terms (mass eigenvalues) and do not cause advective instability, so a
central difference is always used for them.
"""

import numpy as np


def _fd_coeffs(offsets, deriv_order):
    """Finite-difference coefficients on integer offsets (positions relative to
    the grid point), obtained exactly by the Fornberg method (method of
    undetermined coefficients).

    Returns c such that
      sum_k c_k * f(x+offsets_k*h) = h^deriv_order * f^(deriv_order)(x) + O(h^len(offsets))
    (an m=len(offsets)-point stencil, accurate to order m-deriv_order).
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

        # indexing='ij' -> axis 0 is rho, axis 1 is sigma
        self.RHO, self.SIGMA = np.meshgrid(self.rho, self.sigma, indexing='ij')
        self.shape = self.RHO.shape

    # ------------------------------------------------------------
    # 1D stencils (first/second derivative along an axis, 2nd order)
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
        """Fourth-order (5-point stencil, coefficients from the Fornberg method)
        first/second derivative. A symmetric 5-point stencil in the interior, and
        asymmetric 5-point stencils for the two points near each boundary."""
        n = U.shape[axis]
        if n < 5:
            raise ValueError("scheme='central4' requires at least 5 grid points per direction")
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
        First-order, downstream (backward) biased one-sided difference.
        Assuming the advection velocity is always non-negative (physical region
        rho,sigma>=0), the interior points and the right boundary use a backward
        difference (i, i-1), and only the left boundary (a stationary point with
        velocity=0) uses a forward difference.
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

        # left edge (stationary point with velocity = 0): forward difference
        d[sl(0)] = (U[sl(1)] - U[sl(0)]) / h

        return d

    # ------------------------------------------------------------
    # Public interface
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
        Return (u_rho, u_sigma, u_rhorho, u_sigmasigma, u_rhosigma) from U(rho, sigma).

        scheme="central" : all second-order central differences (accuracy first;
                             can be unstable at high resolution)
        scheme="central4": all fourth-order central differences (5-point stencil,
                             Fornberg method; fourth order in the interior and at
                             the two points near each boundary; needs >= 5 points
                             per direction)
        scheme="upwind"   : first derivatives (u_rho, u_sigma, u_rhosigma) use a
                             first-order upwind scheme, second derivatives
                             (u_rhorho, u_sigmasigma) stay central (advection-term
                             stabilization; see flow_equation.py)
        """
        u_rho = self.d_drho(U, scheme=scheme)
        u_sigma = self.d_dsigma(U, scheme=scheme)
        u_rhorho = self.d2_drho2(U, scheme=scheme)
        u_sigmasigma = self.d2_dsigma2(U, scheme=scheme)
        u_rhosigma = self.d2_drhodsigma(U, scheme=scheme)
        return u_rho, u_sigma, u_rhorho, u_sigmasigma, u_rhosigma
