"""Climate intervention portfolio optimisation.

A framework for designing portfolios of temporary climate interventions that
offset the warming of arbitrary, time varying, multi species emission
pathways.

The central design decision is that every object in the framework, that is
every impulse response function, every intervention profile, every emission
pathway and every resulting temperature trajectory, is represented as a sum of
polynomial exponential terms. That class of functions is closed under the two
operations the physics demands, convolution and integration, so the whole
calculation is carried out in closed form. There is no time grid, no
quadrature, and no discretisation error anywhere in the results.
"""

from .climate import (
    AR5_THERMAL,
    AR6,
    AR6_CH6_THERMAL,
    AR6_REFERENCE,
    AR6_THERMAL,
    CarbonClimateFeedback,
    CarbonCycle,
    ClimateModel,
    Species,
    ThermalResponse,
)
from .expsum import ExpSum, Signal
from .pathways import (
    EmissionPathway,
    exponential_decline,
    piecewise_linear,
    pulse,
    ramp,
    step,
)
from .portfolio import Portfolio, PortfolioResult, weighting_kernel
from .profiles import (
    Afforestation,
    CostModel,
    DelayedPulse,
    ExponentialRelease,
    FluxProfile,
    Intervention,
    LinearCost,
    LinearRelease,
    PermanentRemoval,
    PowerCost,
    SpeciesRemoval,
)

__version__ = "1.0.0"

__all__ = [
    "AR5_THERMAL",
    "AR6",
    "AR6_CH6_THERMAL",
    "AR6_REFERENCE",
    "AR6_THERMAL",
    "Afforestation",
    "CarbonClimateFeedback",
    "CarbonCycle",
    "ClimateModel",
    "CostModel",
    "DelayedPulse",
    "EmissionPathway",
    "ExpSum",
    "ExponentialRelease",
    "FluxProfile",
    "Intervention",
    "LinearCost",
    "LinearRelease",
    "PermanentRemoval",
    "Portfolio",
    "PortfolioResult",
    "PowerCost",
    "Signal",
    "Species",
    "SpeciesRemoval",
    "ThermalResponse",
    "exponential_decline",
    "piecewise_linear",
    "pulse",
    "ramp",
    "step",
    "weighting_kernel",
]
