"""Exact algebra of poly-exponential signals on the half line.

Every impulse-response function, temperature response and intervention profile
used in this project is (or is well approximated by) a finite sum

.. math::  f(t) = \\sum_m P_m(t)\\, e^{-\\lambda_m t},\\qquad t \\ge 0,

where each :math:`P_m` is a polynomial.  That class of functions is closed
under addition, scaling, convolution, and integration, so the whole framework
of the manuscript can be evaluated in closed form, with no quadrature, no time
grid, no discretisation error.  This module implements that algebra.

Two objects are exposed:

``ExpSum``
    A poly-exponential function supported on :math:`[0,\\infty)`.
``Signal``
    A finite sum of Dirac atoms and *time-shifted* ``ExpSum`` pieces.  Dirac
    atoms are needed because an instantaneous removal of carbon is a delta
    function, and shifts are needed because a piece may switch on at
    :math:`t=t_0>0` (delayed release, finite-duration release windows).

Conventions
-----------
* Rates ``lam`` are inverse timescales in yr^-1; ``lam = 0`` is allowed and
  represents a constant (non-decaying) mode.
* A polynomial is stored as a coefficient array ``c`` with
  ``P(t) = sum_n c[n] t**n``.
* ``definite(T)`` always means :math:`\\int_0^T f(t)\\,dt`.
"""

from __future__ import annotations

from typing import Dict, Iterable, List, Sequence, Tuple, Union

import numpy as np
from scipy.special import gammainc, gammaln

__all__ = ["ExpSum", "Signal", "RATE_TOL"]

# Two rates closer than this (absolute, yr^-1) are treated as identical.  The
# degenerate branch of the convolution formula is the exact limit as the rates
# merge, so using it slightly off-resonance costs a relative error of order
# RATE_TOL * t, i.e. below 1e-7 over a 1000-year horizon.
RATE_TOL = 1e-10

_ArrayLike = Union[float, Sequence[float], np.ndarray]


def _poly_mul(a: np.ndarray, b: np.ndarray) -> np.ndarray:
    """Multiply two polynomials given in ascending-power coefficient form."""
    return np.convolve(a, b)


def _factorial(n: int) -> float:
    return float(np.exp(gammaln(n + 1.0)))


def _trim(c: np.ndarray) -> np.ndarray:
    """Drop trailing zero coefficients, keeping at least one entry."""
    nz = np.nonzero(c)[0]
    if nz.size == 0:
        return np.zeros(1)
    return c[: nz[-1] + 1]


class ExpSum:
    """A poly-exponential function ``sum_m P_m(t) exp(-lam_m t)`` on ``t >= 0``.

    Parameters
    ----------
    modes
        Iterable of ``(lam, coeffs)`` pairs, where ``coeffs`` is the ascending
        coefficient array of the polynomial multiplying ``exp(-lam t)``.
        Repeated rates are merged.
    """

    __slots__ = ("modes",)

    def __init__(self, modes: Iterable[Tuple[float, _ArrayLike]] = ()) -> None:
        acc: List[Tuple[float, np.ndarray]] = []
        for lam, coeffs in modes:
            lam = float(lam)
            c = np.atleast_1d(np.asarray(coeffs, dtype=float)).copy()
            if not np.any(c):
                continue
            for idx, (lam_existing, c_existing) in enumerate(acc):
                if abs(lam - lam_existing) <= RATE_TOL:
                    n = max(c_existing.size, c.size)
                    merged = np.zeros(n)
                    merged[: c_existing.size] += c_existing
                    merged[: c.size] += c
                    acc[idx] = (lam_existing, merged)
                    break
            else:
                acc.append((lam, c))
        self.modes: List[Tuple[float, np.ndarray]] = [
            (lam, _trim(c)) for lam, c in acc if np.any(c)
        ]

    # ------------------------------------------------------------------ ctors
    @classmethod
    def zero(cls) -> "ExpSum":
        return cls()

    @classmethod
    def constant(cls, value: float) -> "ExpSum":
        """The constant function ``value`` on the half line."""
        return cls([(0.0, [value])])

    @classmethod
    def exponential(cls, amplitude: float, lam: float) -> "ExpSum":
        """``amplitude * exp(-lam t)``."""
        return cls([(lam, [amplitude])])

    @classmethod
    def from_exponentials(
        cls, amplitudes: Sequence[float], rates: Sequence[float]
    ) -> "ExpSum":
        return cls(zip(rates, ([a] for a in amplitudes)))

    # ------------------------------------------------------------- properties
    @property
    def is_zero(self) -> bool:
        return len(self.modes) == 0

    @property
    def degree(self) -> int:
        return max((c.size - 1 for _, c in self.modes), default=0)

    @property
    def rates(self) -> np.ndarray:
        return np.array([lam for lam, _ in self.modes])

    # ------------------------------------------------------------- arithmetic
    def __add__(self, other: "ExpSum") -> "ExpSum":
        if not isinstance(other, ExpSum):
            return NotImplemented
        return ExpSum(list(self.modes) + list(other.modes))

    def __sub__(self, other: "ExpSum") -> "ExpSum":
        return self + (-other)

    def __neg__(self) -> "ExpSum":
        return self.scaled(-1.0)

    def scaled(self, factor: float) -> "ExpSum":
        return ExpSum([(lam, c * float(factor)) for lam, c in self.modes])

    __mul__ = scaled
    __rmul__ = scaled

    def __repr__(self) -> str:  # pragma: no cover - debugging aid
        if self.is_zero:
            return "ExpSum(0)"
        parts = [
            "({}) * exp(-{:.6g} t)".format(np.array2string(c, precision=4), lam)
            for lam, c in self.modes
        ]
        return "ExpSum(" + " + ".join(parts) + ")"

    def product(self, other: "ExpSum") -> "ExpSum":
        """Pointwise product ``f(t) g(t)`` -- used for Gram/Hessian integrals."""
        modes: List[Tuple[float, np.ndarray]] = []
        for lam_a, ca in self.modes:
            for lam_b, cb in other.modes:
                modes.append((lam_a + lam_b, _poly_mul(ca, cb)))
        return ExpSum(modes)

    # ------------------------------------------------------------ evaluation
    def eval(self, t: _ArrayLike) -> np.ndarray:
        """Evaluate at ``t``; values at ``t < 0`` are zero (causal support)."""
        t_arr = np.asarray(t, dtype=float)
        out = np.zeros(t_arr.shape, dtype=float)
        active = t_arr >= 0.0
        if not np.any(active):
            return out
        ta = t_arr[active]
        acc = np.zeros(ta.shape, dtype=float)
        for lam, c in self.modes:
            poly = np.polyval(c[::-1], ta)
            acc += poly * np.exp(-lam * ta)
        out[active] = acc
        return out

    def __call__(self, t: _ArrayLike) -> np.ndarray:
        return self.eval(t)

    # ------------------------------------------------------------ integration
    def definite(self, T: _ArrayLike) -> np.ndarray:
        """:math:`\\int_0^T f(t)\\,dt`, evaluated stably.

        Uses the regularised lower incomplete gamma function so that no
        catastrophic cancellation occurs for small ``lam * T``.
        """
        T_arr = np.asarray(T, dtype=float)
        out = np.zeros(T_arr.shape, dtype=float)
        active = T_arr > 0.0
        if not np.any(active):
            return out
        Ta = T_arr[active]
        acc = np.zeros(Ta.shape, dtype=float)
        for lam, c in self.modes:
            for n, coeff in enumerate(c):
                if coeff == 0.0:
                    continue
                if abs(lam) <= RATE_TOL:
                    acc += coeff * Ta ** (n + 1) / (n + 1)
                else:
                    # int_0^T t^n e^{-lam t} dt = n! / lam^{n+1} * P(n+1, lam T)
                    acc += (
                        coeff
                        * _factorial(n)
                        / lam ** (n + 1)
                        * gammainc(n + 1, lam * Ta)
                    )
        out[active] = acc
        return out

    def cumint(self) -> "ExpSum":
        """The antiderivative ``T -> int_0^T f`` as an ``ExpSum``.

        Exact in closed form.  Prefer :meth:`definite` for numerical
        evaluation; ``cumint`` exists so that integrated responses can be fed
        back into further convolutions symbolically.
        """
        modes: List[Tuple[float, np.ndarray]] = []
        for lam, c in self.modes:
            for n, coeff in enumerate(c):
                if coeff == 0.0:
                    continue
                if abs(lam) <= RATE_TOL:
                    poly = np.zeros(n + 2)
                    poly[n + 1] = coeff / (n + 1)
                    modes.append((0.0, poly))
                else:
                    fac_n = _factorial(n)
                    modes.append((0.0, [coeff * fac_n / lam ** (n + 1)]))
                    poly = np.zeros(n + 1)
                    for k in range(n + 1):
                        poly[k] = -coeff * fac_n / _factorial(k) / lam ** (n + 1 - k)
                    modes.append((lam, poly))
        return ExpSum(modes)

    def double_definite(self, T: _ArrayLike) -> np.ndarray:
        """:math:`\\int_0^T\\!\\int_0^{t} f(s)\\,ds\\,dt`.

        Evaluated via the identity :math:`\\int_0^T (T-t) f(t)\\,dt`, which
        avoids forming the antiderivative and is therefore as well conditioned
        as :meth:`definite`.
        """
        T_arr = np.asarray(T, dtype=float)
        out = np.zeros(T_arr.shape, dtype=float)
        active = T_arr > 0.0
        if not np.any(active):
            return out
        Ta = T_arr[active]
        acc = np.zeros(Ta.shape, dtype=float)
        for lam, c in self.modes:
            for n, coeff in enumerate(c):
                if coeff == 0.0:
                    continue
                if abs(lam) <= RATE_TOL:
                    acc += coeff * Ta ** (n + 2) / ((n + 1) * (n + 2))
                else:
                    m0 = _factorial(n) / lam ** (n + 1) * gammainc(n + 1, lam * Ta)
                    m1 = (
                        _factorial(n + 1)
                        / lam ** (n + 2)
                        * gammainc(n + 2, lam * Ta)
                    )
                    acc += coeff * (Ta * m0 - m1)
        out[active] = acc
        return out

    def limit_definite(self, rel_tol: float = 1e-12) -> float:
        """:math:`\\int_0^\\infty f`, or ``+/-inf`` if a non-decaying mode survives.

        A non-decaying mode whose amplitude is negligible against the scale of
        the function is treated as absent. Such modes arise from cancellation:
        composing several kernels that individually carry a constant term can
        leave a residual coefficient many orders of magnitude below the
        function itself, and reporting a divergence on that basis would be an
        artefact of finite precision. ``rel_tol`` sets the threshold relative
        to the largest amplitude present.
        """
        scale = max(
            (float(np.max(np.abs(c))) for _, c in self.modes if np.any(c)),
            default=0.0,
        )
        total = 0.0
        divergent = 0.0
        for lam, c in self.modes:
            if abs(lam) <= RATE_TOL:
                nz = np.nonzero(np.abs(c) > rel_tol * scale)[0]
                if nz.size:
                    divergent = c[nz[-1]]
                continue
            for n, coeff in enumerate(c):
                total += coeff * _factorial(n) / lam ** (n + 1)
        if divergent != 0.0:
            return float(np.sign(divergent) * np.inf)
        return total

    # ------------------------------------------------------------ convolution
    def conv(self, other: "ExpSum") -> "ExpSum":
        """Closed-form convolution ``(f * g)(u) = int_0^u f(s) g(u-s) ds``."""
        modes: List[Tuple[float, np.ndarray]] = []
        for lam_a, ca in self.modes:
            for lam_b, cb in other.modes:
                for a, coef_a in enumerate(ca):
                    if coef_a == 0.0:
                        continue
                    for b, coef_b in enumerate(cb):
                        if coef_b == 0.0:
                            continue
                        modes.extend(
                            _conv_monomials(
                                a, lam_a, b, lam_b, coef_a * coef_b
                            )
                        )
        return ExpSum(modes)


def _conv_monomials(
    a: int, lam_a: float, b: int, lam_b: float, amp: float
) -> List[Tuple[float, np.ndarray]]:
    """Convolve ``amp * s^a e^{-lam_a s}`` with ``(u-s)^b e^{-lam_b (u-s)}``.

    Returns the result as a list of ``(rate, coeffs)`` modes.  Derivation:
    with :math:`\\nu = \\lambda_a - \\lambda_b`,

    .. math::
        \\int_0^u s^a e^{-\\lambda_a s}(u-s)^b e^{-\\lambda_b(u-s)}\\,ds
        = e^{-\\lambda_b u}\\sum_{k=0}^{b}\\binom{b}{k}(-1)^k u^{b-k}
          \\int_0^u s^{a+k}e^{-\\nu s}\\,ds ,

    and the inner integral is elementary, contributing one term at rate
    :math:`\\lambda_b` and one at rate :math:`\\lambda_a`.  When
    :math:`\\nu \\to 0` the two collapse into a single mode, handled by the
    degenerate branch (which is the exact limit).
    """
    nu = lam_a - lam_b
    if abs(nu) <= RATE_TOL:
        # int_0^u s^{a+k} ds = u^{a+k+1}/(a+k+1)
        total = 0.0
        for k in range(b + 1):
            total += _binom(b, k) * (-1.0) ** k / (a + k + 1)
        poly = np.zeros(a + b + 2)
        poly[a + b + 1] = amp * total
        return [(lam_a, poly)]

    poly_a = np.zeros(a + b + 1)  # coefficients of the e^{-lam_a u} mode
    poly_b = np.zeros(b + 1)  # coefficients of the e^{-lam_b u} mode
    for k in range(b + 1):
        weight = _binom(b, k) * (-1.0) ** k
        p = a + k
        fac_p = _factorial(p)
        # constant part of int_0^u s^p e^{-nu s} ds -> rides on e^{-lam_b u}
        poly_b[b - k] += weight * fac_p / nu ** (p + 1)
        # e^{-nu u} * poly part -> rides on e^{-lam_a u}
        for j in range(p + 1):
            poly_a[b - k + j] -= (
                weight * fac_p / _factorial(j) / nu ** (p + 1 - j)
            )
    return [(lam_a, amp * poly_a), (lam_b, amp * poly_b)]


def _binom(n: int, k: int) -> float:
    return float(np.exp(gammaln(n + 1) - gammaln(k + 1) - gammaln(n - k + 1)))


class Signal:
    """A causal signal: Dirac atoms plus time-shifted poly-exponential pieces.

    ``atoms`` is a list of ``(t0, weight)`` giving ``weight * delta(t - t0)``.
    ``pieces`` is a list of ``(t0, ExpSum)`` giving ``f(t - t0)`` for
    ``t >= t0`` and zero before.

    Intervention profiles (Definition 2 of the manuscript) are ``Signal``
    objects; convolving one with a response kernel yields another ``Signal``
    whose integrals are available in closed form.
    """

    __slots__ = ("atoms", "pieces")

    def __init__(
        self,
        atoms: Iterable[Tuple[float, float]] = (),
        pieces: Iterable[Tuple[float, ExpSum]] = (),
    ) -> None:
        self.atoms: List[Tuple[float, float]] = [
            (float(t0), float(w)) for t0, w in atoms if w != 0.0
        ]
        self.pieces: List[Tuple[float, ExpSum]] = [
            (float(t0), es) for t0, es in pieces if not es.is_zero
        ]

    # ------------------------------------------------------------------ ctors
    @classmethod
    def delta(cls, weight: float = 1.0, t0: float = 0.0) -> "Signal":
        return cls(atoms=[(t0, weight)])

    @classmethod
    def from_expsum(cls, es: ExpSum, t0: float = 0.0) -> "Signal":
        return cls(pieces=[(t0, es)])

    @classmethod
    def window(cls, amplitude: float, t0: float, t1: float, lam: float = 0.0) -> "Signal":
        """``amplitude * exp(-lam (t - t0))`` restricted to ``t0 <= t < t1``.

        Built as the difference of two switched-on exponentials, which keeps
        the object inside the closed algebra.
        """
        head = ExpSum.exponential(amplitude, lam)
        tail = ExpSum.exponential(amplitude * float(np.exp(-lam * (t1 - t0))), lam)
        return cls(pieces=[(t0, head), (t1, -tail)])

    # ------------------------------------------------------------- arithmetic
    def __add__(self, other: "Signal") -> "Signal":
        if not isinstance(other, Signal):
            return NotImplemented
        return Signal(self.atoms + other.atoms, self.pieces + other.pieces)

    def __sub__(self, other: "Signal") -> "Signal":
        return self + (-other)

    def __neg__(self) -> "Signal":
        return self.scaled(-1.0)

    def scaled(self, factor: float) -> "Signal":
        f = float(factor)
        return Signal(
            [(t0, w * f) for t0, w in self.atoms],
            [(t0, es.scaled(f)) for t0, es in self.pieces],
        )

    __mul__ = scaled
    __rmul__ = scaled

    def shifted(self, dt: float) -> "Signal":
        return Signal(
            [(t0 + dt, w) for t0, w in self.atoms],
            [(t0 + dt, es) for t0, es in self.pieces],
        )

    def __repr__(self) -> str:  # pragma: no cover - debugging aid
        return "Signal(atoms={}, pieces={})".format(
            len(self.atoms), len(self.pieces)
        )

    # ------------------------------------------------------------ integration
    def mass(self) -> float:
        """:math:`\\int_0^\\infty` of the signal.

        For a *temporary* intervention this must be zero: every unit of carbon
        removed is eventually returned (Definition 2 normalisation).

        Non-decaying (``lam = 0``) modes diverge piece by piece even when the
        signal as a whole has finite mass -- a finite release window is built
        as the difference of two switched-on constants.  Those modes are
        therefore accumulated as a polynomial in the upper limit ``T`` across
        all pieces; the coefficients of the positive powers of ``T`` must
        cancel, and the surviving constant term is the contribution to the
        mass.
        """
        total = float(sum(w for _, w in self.atoms))
        # Polynomial in T accumulated from the lam = 0 modes of every piece.
        growth = np.zeros(1)
        for t0, es in self.pieces:
            for lam, c in es.modes:
                if abs(lam) <= RATE_TOL:
                    for n, coeff in enumerate(c):
                        if coeff == 0.0:
                            continue
                        # int_0^{T - t0} u^n du = (T - t0)^{n+1} / (n + 1)
                        shifted = np.zeros(n + 2)
                        for j in range(n + 2):
                            shifted[j] = (
                                _binom(n + 1, j)
                                * (-t0) ** (n + 1 - j)
                                * coeff
                                / (n + 1)
                            )
                        if shifted.size > growth.size:
                            growth = np.append(
                                growth, np.zeros(shifted.size - growth.size)
                            )
                        growth[: shifted.size] += shifted
                else:
                    for n, coeff in enumerate(c):
                        total += coeff * _factorial(n) / lam ** (n + 1)
        if growth.size > 1:
            scale = max(float(np.max(np.abs(growth))), 1.0)
            if np.any(np.abs(growth[1:]) > 1e-9 * scale):
                return float(np.sign(growth[np.nonzero(growth)[0][-1]]) * np.inf)
        return float(total + growth[0])

    def definite(self, T: _ArrayLike) -> np.ndarray:
        """:math:`\\int_0^T` of the signal, atoms included."""
        T_arr = np.asarray(T, dtype=float)
        acc = np.zeros(T_arr.shape, dtype=float)
        for t0, w in self.atoms:
            acc = acc + w * (T_arr >= t0)
        for t0, es in self.pieces:
            acc = acc + es.definite(np.maximum(T_arr - t0, 0.0))
        return acc

    def eval(self, t: _ArrayLike) -> np.ndarray:
        """Evaluate the *continuous part* only (Dirac atoms are not functions)."""
        t_arr = np.asarray(t, dtype=float)
        acc = np.zeros(t_arr.shape, dtype=float)
        for t0, es in self.pieces:
            acc = acc + es.eval(t_arr - t0)
        return acc

    def __call__(self, t: _ArrayLike) -> np.ndarray:
        return self.eval(t)

    # ------------------------------------------------------------ convolution
    def convolve_kernel(self, kernel: ExpSum) -> "Signal":
        """Convolve with a kernel supported at the origin.

        A Dirac atom simply reproduces a shifted copy of the kernel; a
        poly-exponential piece is convolved in closed form.  This is the
        operation that turns an intervention profile into a temperature
        response (Eq. 6 of the manuscript).
        """
        pieces: List[Tuple[float, ExpSum]] = []
        for t0, w in self.atoms:
            pieces.append((t0, kernel.scaled(w)))
        for t0, es in self.pieces:
            pieces.append((t0, es.conv(kernel)))
        return Signal(pieces=pieces)

    def double_definite(self, T: _ArrayLike) -> np.ndarray:
        """:math:`\\int_0^T\\!\\int_0^t` of the signal."""
        T_arr = np.asarray(T, dtype=float)
        acc = np.zeros(T_arr.shape, dtype=float)
        for t0, w in self.atoms:
            acc = acc + w * np.maximum(T_arr - t0, 0.0)
        for t0, es in self.pieces:
            acc = acc + es.double_definite(np.maximum(T_arr - t0, 0.0))
        return acc

    def continuous_product_integral(self, other: "Signal", T: float) -> float:
        """:math:`\\int_0^T f(t) g(t)\\,dt` for the continuous parts.

        Used for Gram matrices and quadratic objectives.  Both signals must be
        Dirac free (temperature responses always are).
        """
        if self.atoms or other.atoms:
            raise ValueError("product integral undefined for signals with Dirac atoms")
        total = 0.0
        for t0_a, es_a in self.pieces:
            for t0_b, es_b in other.pieces:
                start = max(t0_a, t0_b)
                if start >= T:
                    continue
                # Re-base both pieces at `start` so the product is a plain ExpSum.
                a = _rebase(es_a, start - t0_a)
                b = _rebase(es_b, start - t0_b)
                total += float(a.product(b).definite(T - start))
        return total


def _rebase(es: ExpSum, delta: float) -> ExpSum:
    """Return ``g`` with ``g(u) = es(u + delta)`` -- a shift of the origin."""
    if delta == 0.0:
        return es
    modes: List[Tuple[float, np.ndarray]] = []
    for lam, c in es.modes:
        n = c.size - 1
        # (u + delta)^k expansion, times exp(-lam delta)
        new = np.zeros(n + 1)
        for k, coeff in enumerate(c):
            if coeff == 0.0:
                continue
            for j in range(k + 1):
                new[j] += coeff * _binom(k, j) * delta ** (k - j)
        modes.append((lam, new * float(np.exp(-lam * delta))))
    return ExpSum(modes)
