"""Wildfire-robust portfolios of temporary carbon dioxide removal.

Most carbon dioxide removal deployed today is temporary, and the dominant
approach to crediting it treats durability as a known number. Wildfire makes
that treatment unsafe: a store expected to hold carbon for a century may lose
it in thirty years, and the regions where forest carbon projects are most
attractive are the regions where fire risk is highest, so the error is
systematic rather than random.

This package replaces the deterministic durability parameter with a stochastic
one, and the deterministic neutrality constraint with a chance constraint. It
builds on the deterministic framework of :mod:`cipo`, which supplies the
impulse-response climate model, the intervention profile families and the
closed poly-exponential algebra, and adds four things.

:mod:`wrcp.algebra`
    The identity that cumulative cooling is a linear functional of the stored
    stock, and the one new primitive the stochastic problem needs: an exact,
    overflow-free integral over the triangle that a terminal-value variance
    produces.

:mod:`wrcp.hazard`
    Fire occurrence as a non-homogeneous Poisson process with climate-dependent
    intensity, severity as a distribution on the unit interval, and the
    common-shock construction that couples measures sharing a fire-weather
    regime.

:mod:`wrcp.stochastic`
    The mean and covariance of delivered cooling in closed form, split into the
    part diversification removes and the part it cannot.

:mod:`wrcp.robust`, :mod:`wrcp.spike`, :mod:`wrcp.schedule`
    The chance-constrained portfolio problem, the fire premium, the confidence
    ceiling, the peak-constrained variant, and the fire-aware deployment
    schedule.

Calibration lives in :mod:`wrcp.calibration` and the two case studies in
:mod:`wrcp.regions`. Every parameter is traceable to a published source, and
where a published statistic must be inverted to reach a model parameter the
inversion is a named function rather than a buried constant.
"""

from .algebra import (
    Piecewise,
    reverse_weighted_integral,
    signal_product,
    stock_signal,
    triangle_integral,
)
from .hazard import (
    NO_FIRE,
    STAND_REPLACING,
    BetaSeverity,
    FireHazard,
    Severity,
    TruncatedParetoSeverity,
    TwoPointSeverity,
    drought_path,
    temperature_path,
)
from .robust import FirePortfolio, RobustResult, confidence_from_kappa, kappa
from .schedule import FireSchedule, ScheduleResult
from .spike import PeakPortfolio, SpikeResult, spike_trajectory, worst_case_spike
from .stochastic import FireExposedStore, FireMoments, cooling_moments

__version__ = "1.0.0"

__all__ = [
    "BetaSeverity",
    "FireExposedStore",
    "FireHazard",
    "FireMoments",
    "FirePortfolio",
    "FireSchedule",
    "NO_FIRE",
    "PeakPortfolio",
    "Piecewise",
    "RobustResult",
    "STAND_REPLACING",
    "ScheduleResult",
    "Severity",
    "SpikeResult",
    "TruncatedParetoSeverity",
    "TwoPointSeverity",
    "confidence_from_kappa",
    "cooling_moments",
    "drought_path",
    "kappa",
    "reverse_weighted_integral",
    "signal_product",
    "spike_trajectory",
    "stock_signal",
    "temperature_path",
    "triangle_integral",
    "worst_case_spike",
]
