"""Fire occurrence and severity: the hazard process and its moments.

Fire destroys a carbon store in two steps: an event occurs, and a fraction of
the stock burns. The two are modelled separately because they are measured
separately. Occurrence is calibrated from fire return intervals and from the
burned-area record; severity is calibrated from burn-severity classifications.

Occurrence
----------
Events follow a non-homogeneous Poisson process with intensity

.. math::
    \\lambda_i(t) = \\lambda_{i,0}\\,\\exp\\!\\big(
                    \\beta_T [\\Delta T(t) - \\Delta T(0)]
                    + \\beta_D [D(t) - D(0)]\\big) ,

where :math:`\\lambda_{i,0}` is the inverse of the baseline fire return
interval, :math:`\\Delta T` is the regional temperature anomaly and :math:`D` a
standardised drought index. The modifier is normalised at the start of the
horizon, which Eq. (24) of the research plan does not do. Without it
:math:`\\lambda_{i,0}` is not the calibrated present-day rate but that rate
divided by the modifier at present-day warming, a factor of about two for the
regional anomalies used here, and every return interval quoted from the
literature would enter the model at the wrong value. The intensity is held
piecewise constant on blocks,
by default decadal. This is a deliberate choice rather than an approximation of
convenience, and it is where this implementation departs from Eq. (24) of the
research plan.

The plan writes a single intensity carrying both the secular climate trend and
the year-to-year fluctuation :math:`\\xi_i(t)`. Those two have different
mathematical roles. The secular trend is a smooth, known function of the
scenario, and resolving it decadally is ample: over a decade the trend moves
the hazard by a few per cent. The annual fluctuation is not a refinement of the
trend but a random variable, and its effect on a century-scale store is
governed by its *distribution*, not by its trajectory. Folding it into the
intensity as though it were known would understate the risk, because a store
does not care which year burned, only that one did. It is therefore carried
separately, as a random scaling of :math:`\\lambda_{i,0}` integrated over in
:mod:`wrcp.stochastic`. The payoff is that the conditional problem stays in
closed form, which is what makes the cross-measure covariance floor computable
at all.

Severity
--------
When an event occurs, a fraction :math:`\\phi \\in [0,1]` of the standing stock
is lost. Only the first two moments of :math:`\\phi` enter the robust
constraint, which is what makes the Chebyshev formulation distribution free, so
the severity classes exposed here are characterised by
:math:`m_1 = \\mathbb{E}[\\phi]` and :math:`m_2 = \\mathbb{E}[\\phi^2]` and are
otherwise interchangeable. A sampler is provided for the Monte Carlo control in
the test suite.

Survival
--------
With independent severities the stock multiplier
:math:`S(t) = \\prod_{\\tau_k \\le t}(1-\\phi_k)` has moments available in closed
form. For a Poisson process with cumulative intensity :math:`\\Lambda` and
independent marks, :math:`\\mathbb{E}\\prod f(\\phi_k)
= \\exp(-\\Lambda(t)(1 - \\mathbb{E}f(\\phi)))`, so

.. math::
    \\mathbb{E}[S(t)] = e^{-m_1 \\Lambda(t)} ,
    \\qquad
    \\mathbb{E}[S(s)S(t)] = \\mathbb{E}[S(s)]\\,\\mathbb{E}[S(t)]\\,
                            e^{m_2 \\Lambda(s)} \\quad (s \\le t) .

The second identity follows because events before :math:`s` enter both factors,
contributing :math:`\\mathbb{E}(1-\\phi)^2 = 1 - 2m_1 + m_2`, while events
between :math:`s` and :math:`t` enter one. Its consequence is that the
autocovariance of the survival multiplier is
:math:`\\mathbb{E}[S(s)]\\mathbb{E}[S(t)](e^{m_2\\Lambda(s \\wedge t)} - 1)`,
which is non-negative and vanishes with the hazard, as it must.

Correlation across measures
---------------------------
Gap 5 of the research plan asks for the covariance induced by a common climate
driver. It is obtained here by a common-shock construction. Each measure's
event process is the superposition of a regional process shared by every
measure in the region, with intensity :math:`\\theta_r(t)`, and an independent
measure-specific remainder of intensity :math:`\\lambda_i - \\theta_r \\ge 0`.
The marginal intensity is unchanged, and for :math:`s \\le t`

.. math::
    \\operatorname{Cov}[S_i(s), S_j(t)]
      = \\mathbb{E}[S_i(s)]\\,\\mathbb{E}[S_j(t)]\\,
        \\big(e^{m_{\\mathrm{eff}} \\Lambda_{\\theta}(s)} - 1\\big) ,

with :math:`m_{\\mathrm{eff}} = 1 - \\mathbb{E}(1-\\phi_i)(1-\\phi_j) - m_1` and
:math:`m_{\\mathrm{eff}} \\to m_2` as the two measures' severities become
perfectly shared, which is exactly the marginal case. The shared cumulative
intensity :math:`\\Lambda_\\theta` appears only at the earlier of the two times,
so a bad fire year early in the horizon contributes more covariance than a late
one: the stores still hold most of their carbon then. That is the floor on
portfolio variance that diversification cannot reduce.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Callable, Dict, Optional, Sequence, Tuple

import numpy as np

from cipo.expsum import ExpSum

from .algebra import Piecewise

__all__ = [
    "Severity",
    "BetaSeverity",
    "TwoPointSeverity",
    "TruncatedParetoSeverity",
    "STAND_REPLACING",
    "FireHazard",
    "NO_FIRE",
    "temperature_path",
    "drought_path",
    "shared_intensity",
    "shared_growth",
    "cross_severity_exponent",
]


# ------------------------------------------------------------------- severity
class Severity:
    """Distribution of the burned fraction of standing stock.

    Only :attr:`m1` and :attr:`m2` enter the robust constraint. Subclasses must
    supply both, and may supply a sampler for the Monte Carlo control.
    """

    name: str = "severity"

    @property
    def m1(self) -> float:
        raise NotImplementedError

    @property
    def m2(self) -> float:
        raise NotImplementedError

    @property
    def variance(self) -> float:
        return float(self.m2 - self.m1 ** 2)

    def sample(self, rng: np.random.Generator, size: int) -> np.ndarray:
        raise NotImplementedError

    def __repr__(self) -> str:  # pragma: no cover - debugging aid
        return "{}(m1={:.4g}, m2={:.4g})".format(
            self.__class__.__name__, self.m1, self.m2
        )


@dataclass(frozen=True)
class BetaSeverity(Severity):
    """Beta distributed burned fraction with a given mean and concentration.

    ``concentration`` is ``a + b``: large values concentrate the distribution
    on its mean, small values push mass towards zero and one. The second
    moment is ``m1 * (a + 1) / (concentration + 1)``.
    """

    mean: float = 0.6
    concentration: float = 4.0
    name: str = "beta"

    @property
    def m1(self) -> float:
        return float(self.mean)

    @property
    def m2(self) -> float:
        a = self.mean * self.concentration
        return float(self.mean * (a + 1.0) / (self.concentration + 1.0))

    def sample(self, rng: np.random.Generator, size: int) -> np.ndarray:
        a = self.mean * self.concentration
        b = self.concentration - a
        return rng.beta(a, b, size=size)


@dataclass(frozen=True)
class TwoPointSeverity(Severity):
    """A stand-replacing fire with probability ``p``, else a partial burn.

    This is the form in which burn-severity products arrive: the Monitoring
    Trends in Burn Severity classification separates high severity, where the
    overstorey is killed, from low and moderate severity, where a part of the
    stock survives. ``p`` is the high-severity area fraction and
    ``partial`` the fraction of stock lost in the remainder.
    """

    p: float = 0.3
    partial: float = 0.35
    full: float = 1.0
    name: str = "two_point"

    @property
    def m1(self) -> float:
        return float(self.p * self.full + (1.0 - self.p) * self.partial)

    @property
    def m2(self) -> float:
        return float(
            self.p * self.full ** 2 + (1.0 - self.p) * self.partial ** 2
        )

    def sample(self, rng: np.random.Generator, size: int) -> np.ndarray:
        hit = rng.random(size) < self.p
        return np.where(hit, self.full, self.partial)


@dataclass(frozen=True)
class TruncatedParetoSeverity(Severity):
    """Pareto burned fraction truncated to ``[floor, 1]``.

    Included because fire severity is fat tailed and because the research plan
    rests its case for a distribution-free bound on exactly that. With tail
    index ``alpha`` below two the untruncated distribution has no finite
    variance; truncation at unity restores it, but leaves a second moment far
    above the Gaussian value implied by the same mean. Comparing the robust
    portfolio under this class against the Beta class of equal mean isolates
    the cost of the tail.
    """

    alpha: float = 1.5
    floor: float = 0.05
    name: str = "truncated_pareto"

    def _moment(self, k: int) -> float:
        a, x0 = float(self.alpha), float(self.floor)
        # Density a x0^a x^{-a-1} / (1 - x0^a) renormalised on [x0, 1].
        norm = 1.0 - x0 ** a
        if abs(a - k) < 1e-12:
            return float(a * x0 ** a * np.log(1.0 / x0) / norm)
        return float(
            a * x0 ** a / (a - k) * (x0 ** (k - a) - 1.0) / norm
        )

    @property
    def m1(self) -> float:
        return self._moment(1)

    @property
    def m2(self) -> float:
        return self._moment(2)

    def sample(self, rng: np.random.Generator, size: int) -> np.ndarray:
        a, x0 = float(self.alpha), float(self.floor)
        u = rng.random(size)
        return (x0 ** (-a) - u * (x0 ** (-a) - 1.0)) ** (-1.0 / a)


#: Every fire kills the whole store. The limiting case, retained because it
#: makes the survival multiplier Bernoulli and so gives a closed-form control
#: on the variance formulae.
STAND_REPLACING = TwoPointSeverity(p=1.0, partial=1.0, full=1.0)


# ---------------------------------------------------------------- climate paths
def temperature_path(
    t0: float = 1.0, t_peak: float = 3.2, tau: float = 70.0
) -> Callable[[np.ndarray], np.ndarray]:
    """A saturating regional temperature anomaly, K above pre-industrial.

    ``dT(t) = t_peak - (t_peak - t0) exp(-t / tau)``. Regional land warming
    over the western United States and the boreal zone runs well above the
    global mean, so the defaults are set from regional rather than global
    projections and are overridden per region in :mod:`wrcp.regions`.
    """

    def path(t) -> np.ndarray:
        t = np.asarray(t, dtype=float)
        return t_peak - (t_peak - t0) * np.exp(-t / tau)

    return path


def drought_path(
    d0: float = 0.0, d_end: float = 1.0, horizon: float = 100.0
) -> Callable[[np.ndarray], np.ndarray]:
    """A linearly intensifying standardised drought index."""

    def path(t) -> np.ndarray:
        t = np.asarray(t, dtype=float)
        return d0 + (d_end - d0) * np.clip(t / horizon, 0.0, 1.0)

    return path


# -------------------------------------------------------------------- hazard
@dataclass
class FireHazard:
    """Piecewise-constant fire intensity with a severity distribution.

    Parameters
    ----------
    lambda0
        Baseline event rate in yr^-1, the inverse of the fire return interval
        under present-day climate. The climate modifier is normalised to one at
        ``t_ref``, so this is the rate at the start of the horizon and not an
        abstract intercept. Without that normalisation the calibrated return
        interval and the intensity at ``t = 0`` differ by the modifier
        evaluated at present-day warming, which for the regional anomalies used
        here is a factor of about two.
    beta_T, beta_D
        Sensitivity of the log intensity to the temperature anomaly, per K, and
        to the standardised drought index.
    t_ref
        Time at which the climate modifier is unity, in years from the start of
        the horizon.
    severity
        Distribution of the burned fraction.
    temperature, drought
        Callables giving the regional anomaly and drought index at a time in
        years from the start of the horizon.
    region
        Label used to decide which measures share a regional fire-weather
        process.
    exposure
        Fraction of the measure's stock that is exposed to fire at all. Zero
        for geological storage, one for standing biomass. A partially exposed
        store, such as biochar with a labile and a recalcitrant pool, is
        handled by this factor rather than by a separate profile family.
    """

    lambda0: float = 0.004
    beta_T: float = 0.55
    beta_D: float = 0.25
    severity: Severity = field(default_factory=lambda: BetaSeverity())
    temperature: Optional[Callable] = None
    drought: Optional[Callable] = None
    region: str = "generic"
    exposure: float = 1.0
    t_ref: float = 0.0
    name: str = "hazard"

    # ------------------------------------------------------------- intensity
    def _log_modifier(self, t) -> np.ndarray:
        t = np.asarray(t, dtype=float)
        out = np.zeros(np.shape(t), dtype=float)
        if self.temperature is not None:
            out = out + self.beta_T * np.asarray(self.temperature(t), dtype=float)
        if self.drought is not None:
            out = out + self.beta_D * np.asarray(self.drought(t), dtype=float)
        return out

    def intensity(self, t) -> np.ndarray:
        """The instantaneous intensity ``lambda_i(t)``, yr^-1.

        Normalised so that ``intensity(t_ref) == lambda0 * exposure``.
        """
        ref = float(np.asarray(self._log_modifier(float(self.t_ref))).ravel()[0])
        return (
            self.lambda0
            * self.exposure
            * np.exp(self._log_modifier(t) - ref)
        )

    def intensification(self, horizon: float) -> float:
        """Ratio of the intensity at the horizon to its present-day value."""
        return float(np.asarray(self.intensity(float(horizon))).ravel()[0] / (
            self.lambda0 * self.exposure
        )) if self.lambda0 * self.exposure > 0 else 1.0

    def block_rates(self, edges: Sequence[float]) -> np.ndarray:
        """Intensity on each block, taken at the block's midpoint.

        The midpoint rule integrates the trend exactly to second order over the
        block, so a decadal block reproduces the cumulative hazard of the
        underlying smooth trend to better than a part in a thousand for the
        trends considered here. :meth:`refinement_error` measures it.
        """
        edges = np.asarray(edges, dtype=float)
        mid = 0.5 * (edges[:-1] + edges[1:])
        return np.asarray(self.intensity(mid), dtype=float)

    def cumulative(self, edges: Sequence[float]) -> np.ndarray:
        """Cumulative hazard at each edge, ``Lambda(edges)``."""
        edges = np.asarray(edges, dtype=float)
        rates = self.block_rates(edges)
        return np.concatenate([[0.0], np.cumsum(rates * np.diff(edges))])

    def refinement_error(self, horizon: float, n_block: int) -> float:
        """Relative error in ``Lambda(TH)`` from the block discretisation.

        Compared against a fine trapezoidal integral of the smooth intensity.
        Reported so the choice of block width is a stated, checked
        approximation of the trend rather than a hidden one. The severity and
        occurrence models themselves are exact.
        """
        edges = np.linspace(0.0, horizon, int(n_block) + 1)
        coarse = float(self.cumulative(edges)[-1])
        fine_t = np.linspace(0.0, horizon, 20001)
        fine = float(np.trapz(self.intensity(fine_t), fine_t))
        if fine == 0.0:
            return 0.0
        return abs(coarse - fine) / fine

    # ------------------------------------------------------- survival factors
    def survival(self, edges: Sequence[float], exponent: Optional[float] = None) -> Piecewise:
        """``exp(-m * Lambda(t))`` as a piecewise exponential.

        ``exponent`` defaults to :attr:`Severity.m1`, giving the mean survival
        multiplier. Other values are needed for the second-moment weights.
        """
        m = float(self.severity.m1 if exponent is None else exponent)
        edges = np.asarray(edges, dtype=float)
        rates = self.block_rates(edges)
        cum = self.cumulative(edges)
        blocks = [
            ExpSum.exponential(float(np.exp(-m * cum[k])), m * float(rates[k]))
            for k in range(rates.size)
        ]
        return Piecewise(edges, blocks)

    def growth(self, edges: Sequence[float], exponent: float) -> Piecewise:
        """``exp(+m * Lambda(t)) - 1`` as a piecewise exponential.

        This is the factor by which the survival autocovariance exceeds zero.
        It is built with a negative rate, which the algebra admits, and its
        magnitude is bounded by ``exp(m * Lambda(TH))``; with a cumulative
        hazard of order one over a century there is no risk of overflow.
        """
        return shared_growth(self.block_rates(edges), edges, exponent)

    # ------------------------------------------------------------- dependence
    def scaled(self, factor: float) -> "FireHazard":
        """A copy with the baseline rate multiplied by ``factor``.

        Used to integrate over the year-to-year fire-weather fluctuation and
        for the sensitivity analysis on the hazard parameters.
        """
        return self.replace(lambda0=self.lambda0 * float(factor))

    def replace(self, **kwargs) -> "FireHazard":
        base = dict(
            lambda0=self.lambda0,
            beta_T=self.beta_T,
            beta_D=self.beta_D,
            severity=self.severity,
            temperature=self.temperature,
            drought=self.drought,
            region=self.region,
            exposure=self.exposure,
            t_ref=self.t_ref,
            name=self.name,
        )
        base.update(kwargs)
        return FireHazard(**base)

    @property
    def is_inert(self) -> bool:
        """True when no fire can reach the store."""
        return self.lambda0 * self.exposure <= 0.0 or self.severity.m1 <= 0.0

    def signature(self) -> tuple:
        return (
            round(self.lambda0, 15),
            round(self.beta_T, 15),
            round(self.beta_D, 15),
            round(self.severity.m1, 15),
            round(self.severity.m2, 15),
            self.region,
            round(self.exposure, 15),
            round(self.t_ref, 15),
        )


#: A store beyond the reach of fire: geological storage, or a mineral pool.
NO_FIRE = FireHazard(lambda0=0.0, exposure=0.0, region="inert", name="none")


def shared_intensity(
    a: FireHazard,
    b: FireHazard,
    edges: Sequence[float],
    rho_within: float = 0.6,
    rho_between: float = 0.15,
) -> np.ndarray:
    """Block rates of the fire-weather process shared by two measures.

    The shared process cannot be more frequent than either measure's own
    process, so its intensity is a fraction of the smaller of the two. Measures
    in one region share a large fraction; measures in different regions share
    only what a global climate driver imposes. Setting ``rho_between`` to zero
    recovers independence across regions and is used to isolate the
    contribution of the common driver to the variance floor.
    """
    if a.is_inert or b.is_inert:
        return np.zeros(np.asarray(edges).size - 1)
    rho = rho_within if a.region == b.region else rho_between
    return float(rho) * np.minimum(a.block_rates(edges), b.block_rates(edges))


def shared_growth(
    rates: np.ndarray, edges: Sequence[float], exponent: float
) -> Piecewise:
    """``exp(+m * Lambda_theta(t)) - 1`` for a given set of block rates."""
    edges = np.asarray(edges, dtype=float)
    rates = np.asarray(rates, dtype=float)
    cum = np.concatenate([[0.0], np.cumsum(rates * np.diff(edges))])
    m = float(exponent)
    blocks = [
        ExpSum.exponential(float(np.exp(m * cum[k])), -m * float(rates[k]))
        + ExpSum.constant(-1.0)
        for k in range(rates.size)
    ]
    return Piecewise(edges, blocks)


def cross_severity_exponent(
    a: Severity, b: Severity, sharing: float = 0.5
) -> float:
    """The exponent ``m_eff`` governing covariance between two measures.

    Carrying the common-shock construction through gives, for :math:`s \\le t`,
    a ratio :math:`\\mathbb{E}[S_i(s)S_j(t)] / (\\mathbb{E}[S_i(s)]
    \\mathbb{E}[S_j(t)]) = \\exp(m_{\\mathrm{eff}}\\Lambda_\\theta(s))` with

    .. math::
        m_{\\mathrm{eff}} = J - 1 + m_1^a + m_1^b ,
        \\qquad J = \\mathbb{E}\\big[(1-\\phi_a)(1-\\phi_b)\\big] ,

    every other term cancelling against the two marginals. Two couplings bound
    :math:`J`. Independent severities give :math:`J = (1-m_1^a)(1-m_1^b)` and
    hence :math:`m_{\\mathrm{eff}} = m_1^a m_1^b`. The comonotone coupling
    attains the Cauchy-Schwarz bound :math:`\\mathbb{E}[\\phi_a\\phi_b] =
    \\sqrt{m_2^a m_2^b}` and gives :math:`m_{\\mathrm{eff}} =
    \\sqrt{m_2^a m_2^b}`. ``sharing`` interpolates between them, so

    .. math::
        m_{\\mathrm{eff}} = \\sigma \\sqrt{m_2^a m_2^b}
                            + (1-\\sigma)\\, m_1^a m_1^b .

    At ``sharing = 1`` with a single measure this is exactly :math:`m_2`, the
    marginal exponent, which is the consistency check the construction must
    pass.
    """
    s = float(np.clip(sharing, 0.0, 1.0))
    return float(s * np.sqrt(a.m2 * b.m2) + (1.0 - s) * a.m1 * b.m1)
