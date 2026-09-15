"""Fire-integrated carbon dynamics: the moments of delivered cooling.

This is the module the rest of the framework rests on. It converts a
deterministic measure, as defined in :mod:`cipo.profiles`, together with a fire
hazard, as defined in :mod:`wrcp.hazard`, into the two numbers the robust
constraint needs: the mean and the covariance of the cumulative cooling the
measure actually delivers.

The construction
----------------
The stock of measure :math:`i` under fire is the deterministic stock multiplied
by the survival process,

.. math:: A_i(t) = A_i^{\\mathrm{det}}(t)\\, S_i(t) ,
          \\qquad S_i(t) = \\prod_{\\tau_k \\le t} (1 - \\phi_k) ,

because a fire removes a fraction of whatever is standing and the deterministic
release continues to act on the remainder. Cumulative cooling is a linear
functional of the stock path (see :mod:`wrcp.algebra`), so

.. math::
    \\mu_i = \\int_0^{T_{\\mathrm H}} A_i^{\\mathrm{det}}(s)\\,
             \\mathbb{E}[S_i(s)]\\, \\mathrm{AGTP}_{\\mathrm{CO_2}}
             (T_{\\mathrm H}-s)\\, ds ,

a single forward convolution, and

.. math::
    \\Sigma_{ij} = 2 \\int_0^{T_{\\mathrm H}}\\!\\!\\int_0^{s}
        f_i(s)\\, h_j(s')\\, \\mathrm{AGTP}_{\\mathrm{CO_2}}(T_{\\mathrm H}-s)\\,
        \\mathrm{AGTP}_{\\mathrm{CO_2}}(T_{\\mathrm H}-s')\\, ds'\\, ds
    \\quad (i = j) ,

with :math:`f_i = A_i^{\\mathrm{det}} \\mathbb{E}[S_i]` and
:math:`h_i = f_i\\,(e^{m_2 \\Lambda_i} - 1)`, and the symmetrised pair of such
integrals off the diagonal. Every factor is piecewise polynomial exponential
and the integrals are exact.

Two layers of uncertainty
-------------------------
The expressions above are conditional on the hazard parameters. The year-to-year
fire-weather fluctuation of Gap 4 and the parameter uncertainty of the
limitations section are a second layer, and they are handled by the law of total
variance,

.. math::
    \\Sigma = \\mathbb{E}_\\xi\\big[\\Sigma(\\xi)\\big]
              + \\operatorname{Cov}_\\xi\\big[\\mu(\\xi)\\big] ,

with each conditional term evaluated in closed form at a quadrature node in
:math:`\\xi`. The two pieces are reported separately, because they behave
differently: the first is the *idiosyncratic* timing risk, which diversification
reduces, and the second is the *common* risk, which it does not. Their ratio is
the variance floor the research plan asks for in Eq. (18), and it is what
decides whether a portfolio of many forests is safer than one.

Note on the research plan
-------------------------
Section 3.4 of the plan states that the lost cooling and the transient spike
"cancel to zero in the mean". They do not cancel; they are one effect counted
twice. Carbon released by a fire warms the planet once, and that single warming
*is* the loss of the cooling the store was delivering. The mean effect of a fire
is therefore a loss of cooling of the full magnitude, not zero, which is what
:attr:`FireExposedStore.mean_loss_fraction` reports. What is genuinely extra
about a megafire is not a second warming term but the *concentration* of the
release in time, which matters only for a peak-constrained target; that is
treated in :mod:`wrcp.spike`.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Dict, List, Optional, Sequence, Tuple

import numpy as np

from cipo.climate import ClimateModel
from cipo.expsum import ExpSum, Signal
from cipo.profiles import Intervention

from .algebra import (
    Piecewise,
    breakpoints,
    merge_edges,
    reverse_weighted_integral,
    stock_signal,
    triangle_integral,
)
from .hazard import (
    NO_FIRE,
    FireHazard,
    cross_severity_exponent,
    shared_growth,
    shared_intensity,
)

__all__ = [
    "FireExposedStore",
    "FireMoments",
    "block_edges",
    "cooling_moments",
]


def block_edges(
    horizon: float, n_block: int = 10, extra: Sequence[float] = ()
) -> np.ndarray:
    """Uniform blocks over the horizon, merged with any required breakpoints.

    Ten blocks over a century is the default: it resolves the secular hazard
    trend to better than a part in a thousand (see
    :meth:`wrcp.hazard.FireHazard.refinement_error`) while keeping the variance
    integral cheap enough to sit inside a Monte Carlo loop.
    """
    base = np.linspace(0.0, float(horizon), int(n_block) + 1)
    if len(extra):
        return merge_edges(base, np.asarray(extra, dtype=float))
    return merge_edges(base)


@dataclass
class FireMoments:
    """Mean and covariance of delivered cooling, with its decomposition."""

    mean: np.ndarray
    covariance: np.ndarray
    idiosyncratic: np.ndarray
    common: np.ndarray
    deterministic: np.ndarray
    names: List[str] = field(default_factory=list)
    psd_adjustment: float = 0.0

    @property
    def std(self) -> np.ndarray:
        return np.sqrt(np.clip(np.diag(self.covariance), 0.0, None))

    @property
    def loss_fraction(self) -> np.ndarray:
        """Share of deterministic cooling lost to fire, in expectation."""
        det = np.where(self.deterministic > 0.0, self.deterministic, np.nan)
        return 1.0 - self.mean / det

    @property
    def coefficient_of_variation(self) -> np.ndarray:
        mean = np.where(self.mean > 0.0, self.mean, np.nan)
        return self.std / mean

    @property
    def diversifiable_share(self) -> float:
        """Fraction of the equally weighted portfolio variance that diversifies.

        The idiosyncratic part falls as measures are added; the common part does
        not. Their ratio at equal weights is the cleanest scalar summary of the
        variance floor.
        """
        n = self.mean.size
        d = np.ones(n) / n
        tot = float(d @ self.covariance @ d)
        if tot <= 0.0:
            return 0.0
        return float(d @ self.idiosyncratic @ d / tot)

    def correlation(self) -> np.ndarray:
        s = self.std
        out = np.zeros_like(self.covariance)
        ok = s > 0.0
        idx = np.ix_(ok, ok)
        out[idx] = self.covariance[idx] / np.outer(s[ok], s[ok])
        return out

    def frame(self):
        import pandas as pd

        return pd.DataFrame(
            {
                "measure": self.names or list(range(self.mean.size)),
                "cooling_deterministic": self.deterministic,
                "cooling_mean": self.mean,
                "cooling_std": self.std,
                "loss_fraction": self.loss_fraction,
                "coefficient_of_variation": self.coefficient_of_variation,
                "variance_idiosyncratic": np.diag(self.idiosyncratic),
                "variance_common": np.diag(self.common),
            }
        )


class FireExposedStore:
    """A deterministic measure coupled to a fire hazard.

    The measure supplies the flux profile and hence the deterministic stock;
    the hazard supplies the survival statistics. Nothing here modifies the
    measure, so a portfolio may hold the same profile family under different
    regional hazards, which is how the case studies represent the same
    silvicultural practice in two fire regimes.
    """

    def __init__(
        self,
        intervention: Intervention,
        hazard: FireHazard = NO_FIRE,
        label: str = "",
    ) -> None:
        self.intervention = intervention
        self.hazard = hazard
        self.label = label or intervention.display
        self._stock: Optional[Signal] = None

    # ------------------------------------------------------------- shortcuts
    @property
    def name(self) -> str:
        return self.intervention.name

    @property
    def max_scale(self) -> Optional[float]:
        return self.intervention.max_scale

    def cost(self, scale: float) -> float:
        return self.intervention.cost(scale)

    def marginal_cost(self, scale: float) -> float:
        return self.intervention.marginal_cost(scale)

    # ----------------------------------------------------------------- stock
    def stock(self) -> Signal:
        """The deterministic stored stock per unit deployment."""
        if self._stock is None:
            flux = self.intervention.flux()
            if flux is None:
                raise TypeError(
                    "measure {!r} has no flux profile, so it holds no carbon "
                    "that fire could reach; wrap it with NO_FIRE instead"
                    .format(self.intervention.name)
                )
            self._stock = stock_signal(flux)
        return self._stock

    def edges(self, horizon: float, n_block: int = 10) -> np.ndarray:
        """Blocks for this measure: uniform, plus the profile's own switches."""
        return block_edges(
            horizon, n_block, extra=breakpoints(self.stock(), horizon=horizon)
        )

    def mean_stock_weight(self, edges: Sequence[float]) -> Piecewise:
        """``A_det(t) * E[S(t)]``, the weight in the mean and in the variance."""
        a = Piecewise.from_signal(self.stock(), edges)
        if self.hazard.is_inert:
            return a
        return a.product(self.hazard.survival(edges))

    def covariance_weight(self, edges: Sequence[float]) -> Piecewise:
        """``A_det(t) E[S(t)] (exp(m2 * Lambda(t)) - 1)``, the marginal weight."""
        if self.hazard.is_inert:
            return Piecewise.zero(edges)
        return self.mean_stock_weight(edges).product(
            self.hazard.growth(edges, self.hazard.severity.m2)
        )

    # --------------------------------------------------------------- moments
    def deterministic_cooling(self, cm: ClimateModel, horizon: float) -> float:
        """``b_i(TH)`` with no fire, for reference and for the premium."""
        return float(self.intervention.cooling(cm, horizon))

    def mean_cooling(
        self, cm: ClimateModel, horizon: float, n_block: int = 10
    ) -> float:
        """``mu_i(TH)``, the expected cumulative cooling under fire risk."""
        if self.hazard.is_inert:
            return self.deterministic_cooling(cm, horizon)
        edges = self.edges(horizon, n_block)
        return reverse_weighted_integral(
            self.mean_stock_weight(edges),
            cm.agtp_kernel("CO2"),
            horizon,
            edges=edges,
        )

    def variance_cooling(
        self, cm: ClimateModel, horizon: float, n_block: int = 10
    ) -> float:
        """``sigma_i^2(TH)`` from the timing and severity of fire alone."""
        if self.hazard.is_inert:
            return 0.0
        edges = self.edges(horizon, n_block)
        return 2.0 * triangle_integral(
            self.mean_stock_weight(edges),
            self.covariance_weight(edges),
            cm.agtp_kernel("CO2"),
            horizon,
            edges=edges,
        )

    def mean_loss_fraction(self, cm: ClimateModel, horizon: float, n_block: int = 10) -> float:
        """Share of the deterministic cooling that fire removes on average.

        The quantity the research plan claims cancels. It does not: it is the
        systematic over-credit that deterministic accounting incurs.
        """
        det = self.deterministic_cooling(cm, horizon)
        if det <= 0.0:
            return float("nan")
        return 1.0 - self.mean_cooling(cm, horizon, n_block) / det

    def survival_probability(self, horizon: float, n_block: int = 10) -> float:
        """``exp(-Lambda(TH))``: the chance of no fire at all over the horizon."""
        if self.hazard.is_inert:
            return 1.0
        edges = self.edges(horizon, n_block)
        return float(np.exp(-self.hazard.cumulative(edges)[-1]))

    def __repr__(self) -> str:  # pragma: no cover - debugging aid
        return "FireExposedStore({!r}, lambda0={:.4g}, region={})".format(
            self.intervention.name, self.hazard.lambda0, self.hazard.region
        )


# --------------------------------------------------------------- cross terms
def _pair_covariance(
    a: FireExposedStore,
    b: FireExposedStore,
    cm: ClimateModel,
    horizon: float,
    edges: np.ndarray,
    rho_within: float,
    rho_between: float,
    sharing: float,
) -> float:
    """``Sigma_ij`` for two distinct measures, through the common shock."""
    if a.hazard.is_inert or b.hazard.is_inert:
        return 0.0
    theta = shared_intensity(a.hazard, b.hazard, edges, rho_within, rho_between)
    if not np.any(theta > 0.0):
        return 0.0
    m_eff = cross_severity_exponent(a.hazard.severity, b.hazard.severity, sharing)
    if m_eff <= 0.0:
        return 0.0
    grow = shared_growth(theta, edges, m_eff)
    kernel = cm.agtp_kernel("CO2")
    fa = a.mean_stock_weight(edges)
    fb = b.mean_stock_weight(edges)
    ha = fa.product(grow)
    hb = fb.product(grow)
    # The shared cumulative hazard enters at the earlier of the two times, so
    # each ordering contributes its own term and the result is symmetric.
    return float(
        triangle_integral(fa, hb, kernel, horizon, edges=edges)
        + triangle_integral(fb, ha, kernel, horizon, edges=edges)
    )


def _nearest_psd(mat: np.ndarray) -> Tuple[np.ndarray, float]:
    """Clip negative eigenvalues, reporting the relative size of the change.

    The pairwise common-shock construction is exact for every pair but is not
    guaranteed to assemble into a positive semidefinite matrix once three or
    more measures share drivers to different degrees. Clipping is the standard
    repair; reporting its magnitude keeps the approximation visible. In every
    configuration used in this study the adjustment is below a part in
    :math:`10^{6}`, so the constraint geometry is unaffected.
    """
    sym = 0.5 * (mat + mat.T)
    vals, vecs = np.linalg.eigh(sym)
    if np.all(vals >= 0.0):
        return sym, 0.0
    scale = float(np.max(np.abs(vals))) or 1.0
    fixed = (vecs * np.clip(vals, 0.0, None)) @ vecs.T
    return 0.5 * (fixed + fixed.T), float(-np.min(vals) / scale)


def cooling_moments(
    stores: Sequence[FireExposedStore],
    cm: ClimateModel,
    horizon: float,
    n_block: int = 10,
    rho_within: float = 0.6,
    rho_between: float = 0.15,
    sharing: float = 0.5,
    weather_spread: float = 0.0,
    n_weather: int = 9,
) -> FireMoments:
    """Mean and covariance of the cooling vector under fire risk.

    Parameters
    ----------
    stores
        The fire-exposed measures, in the order used by the optimisation.
    horizon
        Policy horizon in years.
    n_block
        Blocks over which the hazard trend is held constant.
    rho_within, rho_between
        Strength of the shared fire-weather process within and across regions.
    sharing
        Coupling of burn severity between two measures hit by one event.
    weather_spread
        Log-normal spread of the year-to-year fire-weather multiplier on the
        baseline hazard. Zero switches off the second layer, leaving only the
        conditional timing risk; a positive value adds the common component
        through the law of total variance. GFED5 interannual variability in
        fire carbon emissions supports a value near ``0.35``.
    n_weather
        Gauss-Hermite nodes used for the outer expectation. Nine nodes
        integrate a smooth function of a log-normal exactly to fifteenth order,
        which is far beyond what the moments require.

    Returns
    -------
    FireMoments
        Carrying the total covariance and its split into the part that
        diversification removes and the part it cannot.
    """
    stores = list(stores)
    n = len(stores)
    names = [s.label for s in stores]
    edges = merge_edges(
        block_edges(horizon, n_block),
        *[breakpoints(s.stock(), horizon=horizon) for s in stores],
    )

    det = np.array([s.deterministic_cooling(cm, horizon) for s in stores])

    kernel = cm.agtp_kernel("CO2")

    def conditional(factor: float) -> Tuple[np.ndarray, np.ndarray]:
        """Mean vector and covariance at one fire-weather multiplier.

        The mean-stock weights are formed once per measure and reused by the
        diagonal and by every pair, which is what keeps the outer quadrature
        over fire weather affordable.
        """
        hazards = [s.hazard.scaled(factor) for s in stores]
        inert = [h.is_inert for h in hazards]
        weights: List[Optional[Piecewise]] = []
        for s, h, dead in zip(stores, hazards, inert):
            a = Piecewise.from_signal(s.stock(), edges)
            weights.append(a if dead else a.product(h.survival(edges)))

        mu = np.array(
            [
                det[i]
                if inert[i]
                else reverse_weighted_integral(
                    weights[i], kernel, horizon, edges=edges
                )
                for i in range(n)
            ]
        )

        cov = np.zeros((n, n))
        for i in range(n):
            if inert[i]:
                continue
            h_i = weights[i].product(hazards[i].growth(edges, hazards[i].severity.m2))
            cov[i, i] = 2.0 * triangle_integral(
                weights[i], h_i, kernel, horizon, edges=edges
            )
        for i in range(n):
            for j in range(i + 1, n):
                if inert[i] or inert[j]:
                    continue
                theta = shared_intensity(
                    hazards[i], hazards[j], edges, rho_within, rho_between
                )
                if not np.any(theta > 0.0):
                    continue
                m_eff = cross_severity_exponent(
                    hazards[i].severity, hazards[j].severity, sharing
                )
                if m_eff <= 0.0:
                    continue
                grow = shared_growth(theta, edges, m_eff)
                # The shared cumulative hazard enters at the earlier of the two
                # times, so each ordering contributes a term of its own.
                v = triangle_integral(
                    weights[i], weights[j].product(grow), kernel, horizon, edges=edges
                ) + triangle_integral(
                    weights[j], weights[i].product(grow), kernel, horizon, edges=edges
                )
                cov[i, j] = cov[j, i] = float(v)
        return mu, cov

    if weather_spread <= 0.0:
        mu, idio = conditional(1.0)
        common = np.zeros((n, n))
    else:
        # Gauss-Hermite nodes on a unit-median log-normal, re-normalised so the
        # multiplier has mean one: a bad fire year is more than proportionally
        # bad, and shifting the mean instead of the median would smuggle an
        # extra hazard increase into the baseline calibration.
        x, w = np.polynomial.hermite_e.hermegauss(int(n_weather))
        w = w / np.sum(w)
        sigma = float(weather_spread)
        factors = np.exp(sigma * x - 0.5 * sigma * sigma)
        mus, covs = [], []
        for f in factors:
            m_f, c_f = conditional(float(f))
            mus.append(m_f)
            covs.append(c_f)
        mus = np.array(mus)
        mu = w @ mus
        idio = np.tensordot(w, np.array(covs), axes=(0, 0))
        dev = mus - mu
        common = (dev * w[:, None]).T @ dev

    total, adj = _nearest_psd(idio + common)
    return FireMoments(
        mean=mu,
        covariance=total,
        idiosyncratic=idio,
        common=common,
        deterministic=det,
        names=names,
        psd_adjustment=adj,
    )
