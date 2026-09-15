"""Exact algebra for terminal-value functionals of a stochastic carbon stock.

The deterministic framework evaluates cumulative cooling as an integral of a
temperature response. Under fire risk the natural object is instead a
functional of the *stored stock*, because it is the stock that fire destroys.
The identity that makes this possible is obtained by parts. Writing
:math:`A(t) = -\\int_0^t F` for the stock of a measure whose net atmospheric
flux is :math:`F`, with :math:`A(0^-) = 0`, and
:math:`\\Psi = \\mathrm{iAGTP}_{\\mathrm{CO_2}}`,

.. math::
    b(T_{\\mathrm H})
      = -\\int_0^{T_{\\mathrm H}}\\!\\big(F * \\mathrm{AGTP}_{\\mathrm{CO_2}}\\big)(t)\\,dt
      = -\\big(F * \\Psi\\big)(T_{\\mathrm H})
      = \\int_0^{T_{\\mathrm H}} A(s)\\,
         \\mathrm{AGTP}_{\\mathrm{CO_2}}(T_{\\mathrm H} - s)\\,ds ,

so cumulative cooling is a *linear functional of the stock path*, and in fact
the convolution :math:`(A * \\mathrm{AGTP}_{\\mathrm{CO_2}})(T_{\\mathrm H})`.
Two consequences follow, and they are what the fire-integrated framework rests
on.

First, the mean delivered cooling needs only the mean stock, so it is a single
forward convolution and stays inside the algebra of :mod:`cipo.expsum`.

Second, the variance needs the stock autocovariance, and reduces to an
integral over the triangle :math:`\\{0 \\le s' < s \\le T_{\\mathrm H}\\}` of a
product of the kernel evaluated at :math:`T_{\\mathrm H}-s` and at
:math:`T_{\\mathrm H}-s'`. That *time-reversed* kernel is the only place where
the deterministic algebra does not suffice, because reversal turns a decaying
mode into a growing one. The integral remains elementary, but evaluating it
naively overflows: a kernel mode of rate :math:`\\nu` contributes a factor
:math:`e^{-\\nu T_{\\mathrm H}}` to its coefficient and a factor
:math:`e^{+\\nu T_{\\mathrm H}}` to its integral, and over a multi-century
horizon each factor alone leaves the range of a double precision number while
their product is of order one. Every routine here therefore carries the
logarithm of the prefactor alongside each mode and forms the two together.
That is the single new primitive this module adds.

The representation
------------------
A fire-adjusted stock is *piecewise* poly-exponential, because a hazard that
follows a climate trajectory is held piecewise constant in time. The class
:class:`Piecewise` carries one :class:`~cipo.expsum.ExpSum` per block, based at
the block's own left edge. That is not merely a convenience. The alternative,
building the same function as a sum of switched-on windows inside a
:class:`~cipo.expsum.Signal`, represents each block as a difference of pieces
that cancel outside it, so the number of modes active on a block grows with the
block index and the cost of the variance integral grows as the fourth power of
the number of blocks. Holding one expression per block keeps the mode count
constant, and with it the cost.

Note on the research plan
-------------------------
Equation (35) of the plan writes the effective cooling with a leading minus
sign. The identity above fixes the sign: with :math:`A \\ge 0` and
:math:`\\mathrm{AGTP}_{\\mathrm{CO_2}} \\ge 0` the cooling is positive, as the
plan's own Table 1 requires, since a fixed-term store must deliver
:math:`\\Psi(T_{\\mathrm H}) - \\Psi((T_{\\mathrm H}-\\tau)_+) \\ge 0`.
"""

from __future__ import annotations

from typing import Dict, List, Optional, Sequence, Tuple

import numpy as np
from scipy.special import gammainc, gammaln

from cipo.expsum import RATE_TOL, ExpSum, Signal

__all__ = [
    "Piecewise",
    "rebase",
    "signal_product",
    "stock_signal",
    "breakpoints",
    "merge_edges",
    "scaled_definite",
    "reverse_weighted_integral",
    "triangle_integral",
    "triangle_integral_quadrature",
]

_MAX_FACT = 96
_FACT = np.exp(gammaln(np.arange(_MAX_FACT + 1) + 1.0))


def _binom(n: int, k: int) -> float:
    return float(np.exp(gammaln(n + 1) - gammaln(k + 1) - gammaln(n - k + 1)))


# --------------------------------------------------------------------- rebasing
def rebase(es: ExpSum, delta: float) -> ExpSum:
    """Return ``g`` with ``g(u) = es(u + delta)``, a shift of the origin.

    A public counterpart of the helper used internally by :mod:`cipo.expsum`.
    Needed because a piece switched on before a block must be re-expressed
    relative to that block's left edge.
    """
    if delta == 0.0:
        return es
    modes: List[Tuple[float, np.ndarray]] = []
    for lam, c in es.modes:
        new = np.zeros(c.size)
        for k, coeff in enumerate(c):
            if coeff == 0.0:
                continue
            for j in range(k + 1):
                new[j] += coeff * _binom(k, j) * delta ** (k - j)
        modes.append((lam, new * float(np.exp(-lam * delta))))
    return ExpSum(modes)


# ---------------------------------------------------------------- signal algebra
def signal_product(a: Signal, b: Signal) -> Signal:
    """Pointwise product of two Dirac-free signals, exactly.

    A signal is a sum of pieces switched on at their own start times,
    :math:`f = \\sum_k p_k(t - a_k)\\mathbf{1}_{t \\ge a_k}`. The product of two
    such sums is again of that form: the term from pieces :math:`k` and
    :math:`l` switches on at :math:`\\max(a_k, a_l)`, and its polynomial
    exponential factor is the product of the two pieces re-based at that time.
    """
    if a.atoms or b.atoms:
        raise ValueError("pointwise product undefined for signals with Dirac atoms")
    pieces: List[Tuple[float, ExpSum]] = []
    for t_a, es_a in a.pieces:
        for t_b, es_b in b.pieces:
            start = max(t_a, t_b)
            pieces.append(
                (start, rebase(es_a, start - t_a).product(rebase(es_b, start - t_b)))
            )
    return Signal(pieces=pieces)


def stock_signal(flux: Signal) -> Signal:
    """The stored stock ``A(t) = -int_0^t F`` of a flux profile, symbolically.

    :meth:`cipo.expsum.Signal.definite` evaluates that integral numerically.
    The stock is needed here as an object that can be multiplied by a survival
    factor and convolved again, so it is formed in closed form instead: a
    Dirac atom of weight ``w`` at ``t0`` integrates to a constant ``-w``
    switched on at ``t0``, and a continuous piece integrates through
    :meth:`cipo.expsum.ExpSum.cumint`.
    """
    pieces: List[Tuple[float, ExpSum]] = []
    for t0, w in flux.atoms:
        pieces.append((t0, ExpSum.constant(-w)))
    for t0, es in flux.pieces:
        pieces.append((t0, es.cumint().scaled(-1.0)))
    return Signal(pieces=pieces)


def breakpoints(*signals: Signal, horizon: float) -> np.ndarray:
    """Sorted switch-on times of the given signals inside ``[0, horizon]``."""
    pts = {0.0, float(horizon)}
    for sig in signals:
        for t0, _ in sig.pieces:
            if 0.0 < t0 < horizon:
                pts.add(float(t0))
    return merge_edges(np.array(sorted(pts), dtype=float))


def merge_edges(*edge_sets: Sequence[float]) -> np.ndarray:
    """Union of several edge vectors, sorted and de-duplicated."""
    pts = np.concatenate([np.asarray(e, dtype=float).ravel() for e in edge_sets])
    pts = np.array(sorted(set(np.round(pts, 12))), dtype=float)
    return pts[np.r_[True, np.diff(pts) > 1e-12]]


# -------------------------------------------------------- piecewise poly-exp
class Piecewise:
    """A piecewise polynomial exponential function on ``[edges[0], edges[-1]]``.

    ``blocks[k]`` is the closed form on ``[edges[k], edges[k+1])``, expressed
    as a function of the offset from ``edges[k]``. Closed under addition,
    scaling and pointwise multiplication, which is all the fire-integrated
    moments require.
    """

    __slots__ = ("edges", "blocks")

    def __init__(self, edges: Sequence[float], blocks: Sequence[ExpSum]) -> None:
        self.edges = np.asarray(edges, dtype=float)
        self.blocks = list(blocks)
        if self.edges.size != len(self.blocks) + 1:
            raise ValueError("expected one block per interval between edges")

    # ------------------------------------------------------------------ ctors
    @classmethod
    def from_signal(cls, sig: Signal, edges: Sequence[float]) -> "Piecewise":
        """Restrict a Dirac-free signal to the given blocks."""
        if sig.atoms:
            raise ValueError("Dirac atoms have no piecewise representation")
        edges = np.asarray(edges, dtype=float)
        blocks: List[ExpSum] = []
        for k in range(edges.size - 1):
            start = float(edges[k])
            acc = ExpSum.zero()
            for t0, es in sig.pieces:
                if t0 <= start + 1e-12:
                    acc = acc + rebase(es, start - t0)
            blocks.append(acc)
        return cls(edges, blocks)

    @classmethod
    def zero(cls, edges: Sequence[float]) -> "Piecewise":
        edges = np.asarray(edges, dtype=float)
        return cls(edges, [ExpSum.zero() for _ in range(edges.size - 1)])

    # ------------------------------------------------------------- arithmetic
    def _check(self, other: "Piecewise") -> None:
        if self.edges.size != other.edges.size or not np.allclose(
            self.edges, other.edges, rtol=0.0, atol=1e-9
        ):
            raise ValueError("piecewise operands must share the same blocks")

    def __add__(self, other: "Piecewise") -> "Piecewise":
        self._check(other)
        return Piecewise(
            self.edges, [a + b for a, b in zip(self.blocks, other.blocks)]
        )

    def __sub__(self, other: "Piecewise") -> "Piecewise":
        self._check(other)
        return Piecewise(
            self.edges, [a - b for a, b in zip(self.blocks, other.blocks)]
        )

    def scaled(self, factor: float) -> "Piecewise":
        return Piecewise(self.edges, [b.scaled(factor) for b in self.blocks])

    __mul__ = scaled
    __rmul__ = scaled

    def product(self, other: "Piecewise") -> "Piecewise":
        """Pointwise product, block by block."""
        self._check(other)
        return Piecewise(
            self.edges, [a.product(b) for a, b in zip(self.blocks, other.blocks)]
        )

    def refine(self, edges: Sequence[float]) -> "Piecewise":
        """Re-express on a finer set of edges covering the same span."""
        edges = np.asarray(edges, dtype=float)
        idx = np.clip(np.searchsorted(self.edges, edges[:-1], side="right") - 1, 0, len(self.blocks) - 1)
        blocks = [
            rebase(self.blocks[j], float(edges[k] - self.edges[j]))
            for k, j in enumerate(idx)
        ]
        return Piecewise(edges, blocks)

    # ------------------------------------------------------------- evaluation
    def eval(self, t) -> np.ndarray:
        t_arr = np.asarray(t, dtype=float)
        out = np.zeros(t_arr.shape, dtype=float)
        lo, hi = float(self.edges[0]), float(self.edges[-1])
        for k, es in enumerate(self.blocks):
            a = float(self.edges[k])
            b = float(self.edges[k + 1])
            sel = (t_arr >= a) & (t_arr < b) if k + 1 < len(self.blocks) else (
                (t_arr >= a) & (t_arr <= b)
            )
            if np.any(sel):
                out[sel] = es.eval(t_arr[sel] - a)
        out[(t_arr < lo) | (t_arr > hi)] = 0.0
        return out

    def __call__(self, t) -> np.ndarray:
        return self.eval(t)

    def definite(self) -> float:
        """Integral over the whole span."""
        return float(
            sum(
                float(es.definite(float(self.edges[k + 1] - self.edges[k])))
                for k, es in enumerate(self.blocks)
            )
        )

    @property
    def horizon(self) -> float:
        return float(self.edges[-1])


# ----------------------------------------------- numerically safe 1-D integrals
def scaled_definite(power, rate, width: float, log_scale) -> np.ndarray:
    """``exp(log_scale) * int_0^width t**power * exp(-rate * t) dt``, exactly.

    ``rate`` may be negative, which is what time reversal produces, and
    ``log_scale`` is the logarithm of the prefactor that reversal attaches to
    the coefficient. The two are combined before either is exponentiated, so
    neither the vanishing prefactor nor the diverging integral is formed on its
    own. Every caller below arranges ``log_scale - rate * width <= 0``, so the
    combined exponential never exceeds one and overflow is impossible.

    Three regimes are needed, selected on ``|rate| * width``.

    When that product is small the closed forms are unusable whatever the sign
    of the rate: both carry a factor ``power! / rate**(power+1)``, which for a
    rate of order :math:`10^{-10}` and a polynomial degree of a few tens
    underflows its denominator to zero and returns infinity, while the
    integral itself is perfectly well behaved and tends to
    ``width**(power+1) / (power+1)``. The series

    .. math::
        \\int_0^{w} t^{p} e^{-rt}\\,dt
          = w^{p+1}\\sum_{n \\ge 0} \\frac{(-rw)^n}{n!\\,(p+n+1)}

    is used there instead. It involves no cancellation and no large
    intermediate, converges geometrically in ``rate * width``, and contains the
    zero-rate case as its leading term. This regime is reached routinely, not
    exceptionally: it is what the deterministic limit of the framework looks
    like, and it is also where a survival rate nearly cancels a carbon-cycle
    kernel rate.

    Above the threshold a positive rate uses the regularised lower incomplete
    gamma function directly, with the factorial and the power of the rate
    combined in an exponent so that no large intermediate is formed.

    A negative rate above the threshold is reduced to the same function by the
    substitution :math:`t = w - s`, which leaves a finite binomial sum. That
    sum alternates, and its terms grow before they decay whenever
    :math:`|r|w` is below the polynomial degree, so it is accurate only for
    :math:`|r| w \\gtrsim p`. Rather than rely on that, the callers in this
    module are arranged so the case never arises: :func:`_reversed_terms`
    expands the reversed factor about the *right* edge of each block and
    reverses the slowly varying stock and survival factors instead of the
    climate kernel. Their rates are bounded by the hazard and by the inverse
    storage time, both of order :math:`10^{-2}` or below, so a reversed rate
    times a block width never leaves the series regime. The branch is retained
    for completeness and is checked over its valid range by the test suite.

    Vectorised over ``power``, ``rate`` and ``log_scale``.
    """
    p = np.atleast_1d(np.asarray(power, dtype=np.int64))
    r = np.atleast_1d(np.asarray(rate, dtype=float))
    ls = np.atleast_1d(np.asarray(log_scale, dtype=float))
    p, r, ls = np.broadcast_arrays(p, r, ls)
    w = float(width)
    out = np.zeros(p.shape, dtype=float)
    if w <= 0.0:
        return out

    z = np.abs(r) * w
    small = z <= _SERIES_THRESHOLD
    pos = (~small) & (r > 0.0)
    neg = (~small) & (r < 0.0)

    if np.any(small):
        pp = p[small].astype(float)
        zz = -r[small] * w
        acc = np.zeros(pp.shape, dtype=float)
        term = np.ones(pp.shape, dtype=float)
        n_max = _series_length(float(np.max(z[small])) if np.any(small) else 0.0, 0)
        for n in range(n_max + 1):
            acc += term / (pp + n + 1.0)
            term = term * zz / (n + 1.0)
        out[small] = np.exp(ls[small]) * w ** (pp + 1.0) * acc

    if np.any(pos):
        pp, rr = p[pos].astype(float), r[pos]
        exponent = ls[pos] + gammaln(pp + 1.0) - (pp + 1.0) * np.log(rr)
        out[pos] = np.exp(exponent) * gammainc(pp + 1.0, rr * w)

    if np.any(neg):
        pp, ll = p[neg], ls[neg]
        mu = -r[neg]
        acc = np.zeros(pp.shape, dtype=float)
        for i in range(int(pp.max()) + 1):
            live = pp >= i
            if not np.any(live):
                continue
            n_i = pp[live].astype(float)
            mu_i = mu[live]
            acc[live] += (
                np.exp(
                    gammaln(n_i + 1.0)
                    - gammaln(n_i - i + 1.0)
                    + (n_i - i) * np.log(w)
                    - (i + 1.0) * np.log(mu_i)
                )
                * (-1.0) ** i
                * gammainc(i + 1.0, mu_i * w)
            )
        # The exp(mu * w) of the substitution joins the prefactor here.
        out[neg] = np.exp(ll + mu * w) * acc
    return out


# --------------------------------------------------- reversed-kernel term lists
_Terms = Dict[Tuple[float, float], np.ndarray]


def _accumulate(terms: _Terms, key: Tuple[float, float], poly: np.ndarray) -> None:
    acc = terms.get(key)
    if acc is None:
        terms[key] = poly
        return
    if poly.size > acc.size:
        acc = np.append(acc, np.zeros(poly.size - acc.size))
    acc[: poly.size] += poly
    terms[key] = acc


def _reversed_terms(
    es: ExpSum, kernel: ExpSum, block_start: float, block_end: float, horizon: float
) -> _Terms:
    """Terms of ``es(w - v) * kernel(horizon - block_end + v)`` on ``v`` in ``[0, w]``.

    The integration variable runs *backwards* from the block's right edge,
    :math:`v = b - s` with :math:`w = b - a`. The result maps
    ``(rate, log_scale)`` to a polynomial in :math:`v`, meaning

    .. math:: \\sum_n c_n v^n e^{-\\mathrm{rate}\\, v}\\; e^{\\mathrm{log\\_scale}} .

    Reversing about the right edge rather than the left is what keeps the
    evaluation well conditioned, and the reason is an asymmetry in the rates.
    A kernel mode :math:`P_m(G+v)e^{-\\nu(G+v)}` with :math:`G = T_{\\mathrm H}
    - b \\ge 0` contributes :math:`e^{-\\nu G}` to ``log_scale`` and
    :math:`+\\nu` to the rate: it stays *decaying*, and its polynomial
    coefficients are all of one sign. A stock-and-survival mode
    :math:`Q(w-v)e^{-\\lambda(w-v)}` contributes :math:`e^{-\\lambda w}` and
    :math:`-\\lambda`, so it is the one that grows. That is the favourable
    assignment: the climate kernel carries rates up to
    :math:`0.29\\,\\mathrm{yr}^{-1}` from the fast thermal mode, whereas the
    stock and survival rates are bounded by the hazard and by the inverse
    storage time, both of order :math:`10^{-2}`. A reversed rate times a block
    width therefore stays inside the series regime of
    :func:`scaled_definite`, where there is no cancellation at all.

    Expanding about the left edge would reverse the kernel instead, giving
    rates as large as :math:`-0.29` and an alternating polynomial; over a
    decadal block that lands squarely in the regime where the binomial sum
    loses digits. The two forms are algebraically identical and differ only in
    conditioning, which is the whole reason for the choice.
    """
    gap = float(horizon - block_end)
    width = float(block_end - block_start)
    out: _Terms = {}
    for lam_f, cf in es.modes:
        for nu, ck in kernel.modes:
            poly = np.zeros(cf.size + ck.size - 1)
            for j, coef_f in enumerate(cf):
                if coef_f == 0.0:
                    continue
                for m, coef_k in enumerate(ck):
                    if coef_k == 0.0:
                        continue
                    # (w - v)^j expanded, times (G + v)^m expanded.
                    for a in range(j + 1):
                        wa = (
                            coef_f
                            * _binom(j, a)
                            * width ** (j - a)
                            * (-1.0) ** a
                        )
                        if wa == 0.0:
                            continue
                        for i in range(m + 1):
                            poly[a + i] += (
                                wa * coef_k * _binom(m, i) * gap ** (m - i)
                            )
            if np.any(poly):
                _accumulate(
                    out,
                    (round(nu - lam_f, 14), round(-nu * gap - lam_f * width, 12)),
                    poly,
                )
    return out


def _integrate_terms(terms: _Terms, width: float) -> float:
    if not terms or width <= 0.0:
        return 0.0
    coef: List[float] = []
    power: List[int] = []
    rate: List[float] = []
    lscale: List[float] = []
    for (r, ls), poly in terms.items():
        nz = np.nonzero(poly)[0]
        for n in nz:
            coef.append(float(poly[n]))
            power.append(int(n))
            rate.append(r)
            lscale.append(ls)
    if not coef:
        return 0.0
    return float(
        np.asarray(coef)
        @ scaled_definite(
            np.asarray(power, dtype=np.int64),
            np.asarray(rate),
            width,
            np.asarray(lscale),
        )
    )


#: Below this value of ``|rate| * width`` the antiderivative is taken as a
#: power series rather than as a constant minus a decaying tail. The two forms
#: are algebraically identical; the difference is numerical, and it matters.
#: See :func:`_antiderivative_terms`.
_SERIES_THRESHOLD = 2.0
_SERIES_EPS = 1e-18
_SERIES_MAX = 64


def _series_length(z: float, p: int = 0) -> int:
    """Terms of the poly-exponential series needed at ``|rate| * width = z``.

    The ``n``-th term carries :math:`z^n/n!` relative to the leading one, so
    the count is fixed by where that ratio drops below :data:`_SERIES_EPS`.
    Bounded by :data:`_SERIES_MAX`, which the threshold on ``z`` keeps well
    clear of: at ``z = 2`` fewer than twenty terms suffice.
    """
    n, term = 0, 1.0
    while n < _SERIES_MAX and term > _SERIES_EPS:
        n += 1
        term *= float(z) / n
    return max(n, 1)


def _antiderivative_terms(terms: _Terms, width: float) -> _Terms:
    """The antiderivative of a term list, vanishing at zero, as a term list.

    For one term :math:`c\\,v^p e^{-r v}` the elementary antiderivative is

    .. math::
        \\int_0^u v^p e^{-rv}\\,dv
          = \\frac{p!}{r^{p+1}}
            - e^{-ru}\\sum_{j=0}^{p}\\frac{p!}{j!\\,r^{p+1-j}}\\,u^j ,

    a constant minus a decaying tail. That form is unusable when :math:`|r|u`
    is small, and small values arise unavoidably here rather than by accident.
    The survival factor contributes rates :math:`m_1\\lambda_k` and the
    autocovariance factor :math:`(m_1-m_2)\\lambda_k`, both of order
    :math:`10^{-3}`, while the carbon dioxide kernel carries a mode at
    :math:`1/394.4 = 2.5\\times10^{-3}`; their difference passes through zero
    for realistic hazards. Both terms above then scale as
    :math:`r^{-(p+1)}` while their difference stays of order
    :math:`u^{p+1}/(p+1)`, so the subtraction loses
    :math:`\\log_{10}[p!(p+1)/(ru)^{p+1}]` digits. At :math:`ru \\sim 10^{-1}`
    and :math:`p = 3` that is six digits, which is what a naive implementation
    silently gives up.

    Below :data:`_SERIES_THRESHOLD` the series

    .. math::
        \\int_0^u v^p e^{-rv}\\,dv
          = \\sum_{n \\ge 0} \\frac{(-r)^n\\,u^{p+n+1}}{n!\\,(p+n+1)}

    is used instead. It is a single polynomial at rate zero, involves no
    cancellation, and converges geometrically in :math:`ru`, so truncating
    where the terms stop mattering is exact to rounding. It also handles
    :math:`r = 0` as its leading term rather than as a special case.

    The choice between the two forms is made on :math:`|r| \\times
    \\mathrm{width}`, not on :math:`|r|`: the series is an expansion in that
    product, and over a block a few decades wide a rate of order one would
    need hundreds of terms and would lose to rounding before it converged.
    Above the threshold the tail is well separated from the constant and the
    elementary form is the accurate one, so the two regimes are complementary.
    """
    out: _Terms = {}
    w = float(width)
    for (r, ls), poly in terms.items():
        for p, c in enumerate(poly):
            if c == 0.0:
                continue
            if abs(r) * w <= _SERIES_THRESHOLD:
                n_max = _series_length(abs(r) * w, p)
                coeffs = np.zeros(p + n_max + 2)
                term = 1.0
                for n in range(n_max + 1):
                    coeffs[p + n + 1] = c * term / (p + n + 1)
                    term *= -r / (n + 1)
                _accumulate(out, (0.0, ls), coeffs)
                continue
            fac = _FACT[p]
            const = np.zeros(1)
            const[0] = c * fac / r ** (p + 1)
            _accumulate(out, (0.0, ls), const)
            tail = np.zeros(p + 1)
            for j in range(p + 1):
                tail[j] = -c * fac / _FACT[j] / r ** (p + 1 - j)
            _accumulate(out, (round(r, 14), ls), tail)
    return out


def _sub_triangle(f_terms: _Terms, g_terms: _Terms, width: float) -> float:
    """``int_0^w f(v) * int_v^w g(v') dv' dv`` for two term lists on one block.

    The inner limits run from :math:`v` to :math:`w`, not from zero, because
    the integration variable of :func:`_reversed_terms` runs backwards from the
    block's right edge: the ordering :math:`s' < s` of the original triangle
    becomes :math:`v' > v`. Writing :math:`G` for the antiderivative of
    :math:`g` vanishing at zero,

    .. math::
        \\int_0^w f(v)\\big[G(w) - G(v)\\big]dv
          = G(w)\\int_0^w f - \\int_0^w f\\,G ,

    both terms of which the existing primitives evaluate.
    """
    if not f_terms or not g_terms or width <= 0.0:
        return 0.0
    g_total = _integrate_terms(g_terms, width)
    f_total = _integrate_terms(f_terms, width)
    gint = _antiderivative_terms(g_terms, width)
    combined: _Terms = {}
    for (rf, lf), pf in f_terms.items():
        for (rg, lg), pg in gint.items():
            _accumulate(
                combined,
                (round(rf + rg, 14), round(lf + lg, 12)),
                np.convolve(pf, pg),
            )
    return float(g_total * f_total - _integrate_terms(combined, width))


# ------------------------------------------------------------------- integrals
def _as_piecewise(obj, horizon: float, edges: Optional[Sequence[float]]) -> Piecewise:
    if isinstance(obj, Piecewise):
        return obj
    if edges is None:
        edges = breakpoints(obj, horizon=horizon)
    return Piecewise.from_signal(obj, edges)


def reverse_weighted_integral(f, kernel: ExpSum, horizon: float, edges=None) -> float:
    """``int_0^TH f(s) * kernel(TH - s) ds`` for a Dirac-free ``f``.

    Equal to the convolution :math:`(f * \\mathrm{kernel})(T_{\\mathrm H})`, so it
    is also obtainable from :meth:`cipo.expsum.Signal.convolve_kernel` when
    ``f`` is a signal. It is provided here because it shares the block
    decomposition and the safe primitive with :func:`triangle_integral`;
    agreement between the two routes is one of the validation checks.
    """
    pw = _as_piecewise(f, horizon, edges)
    total = 0.0
    for k, es in enumerate(pw.blocks):
        if es.is_zero:
            continue
        total += _integrate_terms(
            _reversed_terms(
                es, kernel, float(pw.edges[k]), float(pw.edges[k + 1]), horizon
            ),
            float(pw.edges[k + 1] - pw.edges[k]),
        )
    return float(total)


def triangle_integral(f, g, kernel: ExpSum, horizon: float, edges=None) -> float:
    """:math:`\\int_0^{T_{\\mathrm H}}\\!\\!\\int_0^{s} f(s)g(s')K(T_{\\mathrm H}-s)K(T_{\\mathrm H}-s')\\,ds'ds`.

    On a diagonal block the integral is over a sub-triangle; on an off-diagonal
    block it factorises into a product of one-dimensional integrals, and the
    inner factors are accumulated as the blocks are swept, so the cost is
    linear in the number of blocks rather than quadratic.

    This is the quantity that gives the variance of delivered cooling under
    fire risk: with :math:`f` the mean-stock weight and :math:`g` the
    autocovariance weight of the survival process, the variance is twice this
    integral.
    """
    pf = _as_piecewise(f, horizon, edges)
    pg = _as_piecewise(g, horizon, edges)
    pf._check(pg)
    total = 0.0
    g_below = 0.0
    for k in range(len(pf.blocks)):
        w = float(pf.edges[k + 1] - pf.edges[k])
        if w <= 0.0:
            continue
        a, b = float(pf.edges[k]), float(pf.edges[k + 1])
        ft = (
            {}
            if pf.blocks[k].is_zero
            else _reversed_terms(pf.blocks[k], kernel, a, b, horizon)
        )
        gt = (
            {}
            if pg.blocks[k].is_zero
            else _reversed_terms(pg.blocks[k], kernel, a, b, horizon)
        )
        if g_below != 0.0 and ft:
            total += g_below * _integrate_terms(ft, w)
        total += _sub_triangle(ft, gt, w)
        if gt:
            g_below += _integrate_terms(gt, w)
    return float(total)


# ------------------------------------------------------------ numerical control
def triangle_integral_quadrature(
    f, g, kernel: ExpSum, horizon: float, edges=None, n: int = 240
) -> float:
    """The same integral by block-wise Gauss-Legendre quadrature.

    Used only to verify :func:`triangle_integral`. Quadrature is applied block
    by block, because the integrand is smooth inside a block and kinked at the
    edges; a single rule spanning the whole horizon converges slowly and would
    not be a fair control. Off-diagonal blocks factorise into a product of
    one-dimensional rules, and each diagonal sub-triangle is mapped to the unit
    square by :math:`s' = a + (s-a)y`, which leaves a smooth integrand.
    """
    pf = _as_piecewise(f, horizon, edges)
    pg = _as_piecewise(g, horizon, edges)
    x, wx = np.polynomial.legendre.leggauss(int(n))
    x = 0.5 * (x + 1.0)
    wx = 0.5 * wx

    n_b = len(pf.blocks)
    f_int = np.zeros(n_b)
    g_int = np.zeros(n_b)
    nodes, f_val, g_val = [], [], []
    for k in range(n_b):
        a, b = float(pf.edges[k]), float(pf.edges[k + 1])
        s = a + (b - a) * x
        fv = pf.eval(s) * kernel.eval(horizon - s)
        gv = pg.eval(s) * kernel.eval(horizon - s)
        f_int[k] = (b - a) * float(np.sum(wx * fv))
        g_int[k] = (b - a) * float(np.sum(wx * gv))
        nodes.append(s)
        f_val.append(fv)
        g_val.append(gv)

    total = 0.0
    for k in range(n_b):
        a, b = float(pf.edges[k]), float(pf.edges[k + 1])
        total += f_int[k] * float(np.sum(g_int[:k]))
        # Diagonal sub-triangle, mapped to the unit square.
        s = nodes[k]
        inner = np.empty_like(s)
        for i, si in enumerate(s):
            sp = a + (si - a) * x
            inner[i] = (si - a) * float(
                np.sum(wx * pg.eval(sp) * kernel.eval(horizon - sp))
            )
        total += (b - a) * float(np.sum(wx * f_val[k] * inner))
    return float(total)
