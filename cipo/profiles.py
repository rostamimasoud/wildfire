"""Intervention profile families and their exact climate responses.

An intervention is represented by the *net flux profile* it imposes on the
atmosphere,

.. math::  F_i(t;\\theta_i) = R_i(t;\\theta_i) - L_i(t;\\theta_i),

with :math:`R_i` the removal and :math:`L_i` the release, normalised so that
:math:`\\int_0^\\infty F_i\\,dt = 0` for a genuinely *temporary* measure.  The
sign convention is that of an emission: removal is negative.

The induced temperature response is the convolution

.. math::  \\mathrm{AGTP}_{F_i}(t) = (F_i * \\mathrm{AGTP}_{\\mathrm{CO_2}})(t),

and the cumulative cooling delivered over a horizon is

.. math::  b_i(TH) = -\\int_0^{TH} \\mathrm{AGTP}_{F_i}(t)\\,dt \\;\\ge\\; 0 .

Because every profile in this module lies in the poly-exponential class, both
quantities are available in closed form (see :mod:`cipo.expsum`).

Characteristic timescale
------------------------
Profiles with different shapes are compared through a single invariant, the
**mean storage time**

.. math::  \\bar\\tau = \\frac{1}{A_{\\max}}\\int_0^\\infty A(t)\\,dt
                     = \\frac{1}{A_{\\max}}\\int_0^\\infty t\\,F(t)\\,dt ,

where :math:`A(t) = -\\int_0^t F` is the stored stock.  This is the
tonne-years delivered per tonne of peak storage, and it reduces to the
familiar parameter in every standard family (:math:`\\bar\\tau = \\tau` for
exponential release and for a delayed pulse, :math:`\\tau/2` for constant-rate
release).  Using it in place of the family-specific parameter is what makes
the lifetime threshold of :mod:`cipo.threshold` comparable across families.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Dict, Optional, Sequence, Tuple

import numpy as np

from .climate import ClimateModel
from .expsum import ExpSum, Signal

__all__ = [
    "CostModel",
    "LinearCost",
    "PowerCost",
    "Intervention",
    "PermanentRemoval",
    "ExponentialRelease",
    "DelayedPulse",
    "LinearRelease",
    "Afforestation",
    "SpeciesRemoval",
    "FluxProfile",
]


# --------------------------------------------------------------------- costs
class CostModel:
    """Deployment cost as a function of scale, in USD per unit deployment."""

    def total(self, scale: float) -> float:
        raise NotImplementedError

    def marginal(self, scale: float) -> float:
        raise NotImplementedError

    @property
    def description(self) -> str:
        return self.__class__.__name__


@dataclass(frozen=True)
class LinearCost(CostModel):
    """``C(alpha) = unit_cost * alpha`` -- constant returns to scale."""

    unit_cost: float

    def total(self, scale: float) -> float:
        return float(self.unit_cost * scale)

    def marginal(self, scale: float) -> float:
        return float(self.unit_cost)

    @property
    def description(self) -> str:
        return "linear, {:.4g} per unit".format(self.unit_cost)


@dataclass(frozen=True)
class PowerCost(CostModel):
    """``C(alpha) = unit_cost * alpha**gamma / gamma`` with ``gamma >= 1``.

    Strictly convex for ``gamma > 1``, representing the rising marginal cost
    of scaling a finite resource (land, feedstock, suitable geology).  Strict
    convexity is what makes the interior multi-intervention optimum -- and
    hence the equalised-marginal-cost condition -- generic rather than
    exceptional.
    """

    unit_cost: float
    gamma: float = 1.0
    reference_scale: float = 1.0

    def total(self, scale: float) -> float:
        x = scale / self.reference_scale
        return float(
            self.unit_cost * self.reference_scale * x ** self.gamma / self.gamma
        )

    def marginal(self, scale: float) -> float:
        x = scale / self.reference_scale
        return float(self.unit_cost * x ** (self.gamma - 1.0))

    @property
    def description(self) -> str:
        return "power gamma={:.3g}, {:.4g} per unit at reference scale".format(
            self.gamma, self.unit_cost
        )


# -------------------------------------------------------------- base classes
class Intervention:
    """Base class: a deployable measure with a climate response and a cost.

    Subclasses supply either a flux profile (:meth:`flux`) expressed in
    kilograms of CO2, or a response kernel directly (used by
    :class:`SpeciesRemoval`, where the natural unit is the removed species).
    """

    name: str = "intervention"
    label: str = ""
    cost_model: CostModel = LinearCost(0.0)
    max_scale: Optional[float] = None
    side_effects: Dict[str, float] = {}

    # ------------------------------------------------------------- interface
    def flux(self) -> Optional[Signal]:
        """Net atmospheric CO2 flux per unit deployment, or ``None``."""
        return None

    def response_signal(self, cm: ClimateModel) -> Signal:
        """``AGTP_{F_i}(t)`` per unit deployment, as a Dirac-free signal."""
        flux = self.flux()
        if flux is None:
            raise NotImplementedError
        return flux.convolve_kernel(cm.agtp_kernel("CO2"))

    @property
    def parameters(self) -> Dict[str, float]:
        return {}

    @property
    def mean_storage_time(self) -> float:
        """The invariant characteristic timescale, in years."""
        raise NotImplementedError

    @property
    def display(self) -> str:
        return self.label or self.name

    # ------------------------------------------------------------- responses
    def response(self, cm: ClimateModel, t) -> np.ndarray:
        """Temperature response (negative for a cooling measure), K."""
        return self.response_signal(cm).eval(t)

    def cumulative_response(self, cm: ClimateModel, horizon) -> np.ndarray:
        """``iAGTP_{F_i}(TH)`` -- signed, negative for cooling, K yr."""
        return self.response_signal(cm).definite(horizon)

    def cooling(self, cm: ClimateModel, horizon) -> np.ndarray:
        """``b_i(TH) = -iAGTP_{F_i}(TH)`` -- cumulative cooling, K yr, >= 0."""
        return -self.cumulative_response(cm, horizon)

    def peak_response(self, cm: ClimateModel, horizon: float, n: int = 4001) -> float:
        """Largest absolute temperature response over ``[0, TH]``."""
        t = np.linspace(0.0, horizon, n)
        return float(np.max(np.abs(self.response(cm, t))))

    # ------------------------------------------------------------ book-keeping
    def is_temporary(self, tol: float = 1e-9) -> bool:
        flux = self.flux()
        if flux is None:
            return False
        return abs(flux.mass()) <= tol

    def cost(self, scale: float) -> float:
        return self.cost_model.total(scale)

    def marginal_cost(self, scale: float) -> float:
        return self.cost_model.marginal(scale)

    def __repr__(self) -> str:  # pragma: no cover - debugging aid
        pars = ", ".join("{}={:.4g}".format(k, v) for k, v in self.parameters.items())
        return "{}({})".format(self.__class__.__name__, pars)


# ------------------------------------------------------------------ families
@dataclass
class PermanentRemoval(Intervention):
    """Geological-timescale removal: ``F(t) = -delta(t)``.

    The reference against which temporary measures are judged.  Cumulative
    cooling is exactly ``iAGTP_CO2(TH)`` and grows without bound.
    """

    cost_model: CostModel = LinearCost(0.0)
    max_scale: Optional[float] = None
    side_effects: Dict[str, float] = field(default_factory=dict)
    name: str = "permanent"
    label: str = "Permanent removal"

    def flux(self) -> Signal:
        return Signal.delta(-1.0)

    @property
    def mean_storage_time(self) -> float:
        return float(np.inf)


@dataclass
class ExponentialRelease(Intervention):
    """First-order re-release: ``F(t) = -delta(t) + tau^-1 e^{-t/tau}``.

    The canonical model of temporary carbon dioxide removal (biochar, soil
    carbon, mineral pools with first-order loss).  Mean storage time is
    ``tau``.
    """

    tau: float = 50.0
    cost_model: CostModel = LinearCost(0.0)
    max_scale: Optional[float] = None
    side_effects: Dict[str, float] = field(default_factory=dict)
    name: str = "exponential"
    label: str = ""

    def flux(self) -> Signal:
        return Signal.delta(-1.0) + Signal.from_expsum(
            ExpSum.exponential(1.0 / self.tau, 1.0 / self.tau)
        )

    @property
    def parameters(self) -> Dict[str, float]:
        return {"tau": self.tau}

    @property
    def mean_storage_time(self) -> float:
        return float(self.tau)

    @property
    def display(self) -> str:
        return self.label or "Exponential release, $\\bar\\tau$ = {:g} yr".format(self.tau)


@dataclass
class DelayedPulse(Intervention):
    """Storage for a fixed term then abrupt release: ``-delta(t) + delta(t-tau)``.

    Harvested wood products and other engineered pools with a well-defined
    service life.  The cumulative cooling has the exceptionally simple exact
    form

    .. math:: b(TH;\\tau) = \\Psi(TH) - \\Psi\\big((TH-\\tau)_+\\big),

    with :math:`\\Psi = \\mathrm{iAGTP}_{\\mathrm{CO_2}}`: the measure buys
    exactly the cumulative CO2 warming accrued in the last ``tau`` years of
    the horizon.
    """

    tau: float = 50.0
    cost_model: CostModel = LinearCost(0.0)
    max_scale: Optional[float] = None
    side_effects: Dict[str, float] = field(default_factory=dict)
    name: str = "delayed_pulse"
    label: str = ""

    def flux(self) -> Signal:
        return Signal.delta(-1.0) + Signal.delta(1.0, self.tau)

    @property
    def parameters(self) -> Dict[str, float]:
        return {"tau": self.tau}

    @property
    def mean_storage_time(self) -> float:
        return float(self.tau)

    @property
    def display(self) -> str:
        return self.label or "Delayed pulse, $\\bar\\tau$ = {:g} yr".format(self.tau)


@dataclass
class LinearRelease(Intervention):
    """Constant-rate release over ``[0, T_r]``: ``-delta(t) + T_r^-1 1_{[0,T_r]}``.

    Mean storage time is ``T_r / 2``, so a linear-release pool must run twice
    as long as an exponential pool to deliver the same tonne-years.
    """

    release_time: float = 100.0
    cost_model: CostModel = LinearCost(0.0)
    max_scale: Optional[float] = None
    side_effects: Dict[str, float] = field(default_factory=dict)
    name: str = "linear"
    label: str = ""

    def flux(self) -> Signal:
        return Signal.delta(-1.0) + Signal.window(
            1.0 / self.release_time, 0.0, self.release_time
        )

    @property
    def parameters(self) -> Dict[str, float]:
        return {"release_time": self.release_time}

    @property
    def mean_storage_time(self) -> float:
        return float(self.release_time) / 2.0

    @property
    def display(self) -> str:
        return self.label or "Constant-rate release, $T_r$ = {:g} yr".format(
            self.release_time
        )


@dataclass
class Afforestation(Intervention):
    """Gradual uptake with disturbance losses.

    The stock follows

    .. math:: A(t) = N\\,(1 - e^{-t/\\tau_g})\\,e^{-t/\\tau_d},

    a saturating growth phase of timescale :math:`\\tau_g` combined with
    first-order disturbance losses of timescale :math:`\\tau_d`; ``N`` is fixed
    so that peak stock is one unit.  The flux ``F = -dA/dt`` is
    poly-exponential, integrates to zero, and begins negative (net uptake),
    so no Dirac atom is required.  Unlike the other families the removal
    itself is spread in time, which delays the onset of cooling.
    """

    tau_growth: float = 30.0
    tau_disturbance: float = 150.0
    cost_model: CostModel = LinearCost(0.0)
    max_scale: Optional[float] = None
    side_effects: Dict[str, float] = field(default_factory=dict)
    name: str = "afforestation"
    label: str = ""

    # ------------------------------------------------------------- internals
    @property
    def _peak_time(self) -> float:
        tg, td = self.tau_growth, self.tau_disturbance
        return float(tg * np.log((tg + td) / tg))

    @property
    def _norm(self) -> float:
        tg, td = self.tau_growth, self.tau_disturbance
        tp = self._peak_time
        raw = (1.0 - np.exp(-tp / tg)) * np.exp(-tp / td)
        return float(1.0 / raw)

    def stock(self, t) -> np.ndarray:
        tg, td = self.tau_growth, self.tau_disturbance
        t = np.asarray(t, dtype=float)
        return self._norm * (1.0 - np.exp(-t / tg)) * np.exp(-t / td) * (t >= 0)

    def flux(self) -> Signal:
        tg, td = self.tau_growth, self.tau_disturbance
        n = self._norm
        # F = -dA/dt = N [ (1/td) e^{-t/td} - (1/tg + 1/td) e^{-t(1/tg+1/td)} ]
        es = ExpSum(
            [
                (1.0 / td, [n / td]),
                (1.0 / tg + 1.0 / td, [-n * (1.0 / tg + 1.0 / td)]),
            ]
        )
        return Signal.from_expsum(es)

    @property
    def parameters(self) -> Dict[str, float]:
        return {"tau_growth": self.tau_growth, "tau_disturbance": self.tau_disturbance}

    @property
    def mean_storage_time(self) -> float:
        tg, td = self.tau_growth, self.tau_disturbance
        # int_0^inf A dt = N [ td - tg td / (tg + td) ] = N td^2 / (tg + td)
        return float(self._norm * td * td / (tg + td))

    @property
    def display(self) -> str:
        return self.label or "Afforestation ($\\tau_g$={:g}, $\\tau_d$={:g} yr)".format(
            self.tau_growth, self.tau_disturbance
        )


@dataclass
class SpeciesRemoval(Intervention):
    """Removal of a non-CO2 species from the atmosphere.

    Per kilogram of ``species`` removed with efficiency ``efficiency``, the
    avoided warming is ``-efficiency * AGTP_species(t)``.  Any process CO2
    released in doing so is charged at ``co2_penalty`` kilograms of CO2 per
    kilogram removed and carries the full CO2 response, including its
    permanent component.

    This family is *not* temporary in the sense of Definition 2: it removes a
    forcing agent outright rather than storing carbon, so its cumulative
    cooling saturates instead of being repaid.
    """

    species: str = "CH4"
    efficiency: float = 1.0
    co2_penalty: float = 0.0
    cost_model: CostModel = LinearCost(0.0)
    max_scale: Optional[float] = None
    side_effects: Dict[str, float] = field(default_factory=dict)
    name: str = "species_removal"
    label: str = ""

    def flux(self) -> Optional[Signal]:
        return None

    def response_signal(self, cm: ClimateModel) -> Signal:
        kern = cm.agtp_kernel(self.species).scaled(-self.efficiency)
        if self.co2_penalty:
            kern = kern + cm.agtp_kernel("CO2").scaled(self.co2_penalty)
        return Signal.from_expsum(kern)

    @property
    def parameters(self) -> Dict[str, float]:
        return {"efficiency": self.efficiency, "co2_penalty": self.co2_penalty}

    @property
    def mean_storage_time(self) -> float:
        return float(np.inf)

    def is_temporary(self, tol: float = 1e-9) -> bool:
        return False

    @property
    def display(self) -> str:
        return self.label or "{} removal".format(self.species)


@dataclass
class FluxProfile(Intervention):
    """An intervention defined by an arbitrary user-supplied flux signal."""

    signal: Signal = field(default_factory=Signal)
    characteristic_time: float = float("nan")
    cost_model: CostModel = LinearCost(0.0)
    max_scale: Optional[float] = None
    side_effects: Dict[str, float] = field(default_factory=dict)
    name: str = "custom"
    label: str = "Custom profile"

    def flux(self) -> Signal:
        return self.signal

    @property
    def mean_storage_time(self) -> float:
        return float(self.characteristic_time)


def mean_storage_time_numeric(
    intervention: Intervention, horizon: float = 20000.0, n: int = 400001
) -> float:
    """Mean storage time by quadrature -- an independent check of the analytics.

    Integrates the stored stock ``A(t) = -int_0^t F`` on a fine grid and
    divides by its peak.  Used only in the test suite.
    """
    flux = intervention.flux()
    if flux is None:
        return float("nan")
    t = np.linspace(0.0, horizon, n)
    stock = -flux.definite(t)
    peak = float(np.max(stock))
    return float(np.trapz(stock, t) / peak)
