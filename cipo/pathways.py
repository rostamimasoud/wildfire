"""Multi species emission pathways and their exact temperature response.

An emission pathway is a set of species specific emission rate histories
:math:`\\epsilon_s(t)` in kg yr^-1. The warming it causes is

.. math::
    \\Delta T_E(t) = \\sum_s \\int_0^t \\epsilon_s(t')\\,
                     \\mathrm{AGTP}_s(t - t')\\, dt' ,

and the cumulative warming over a horizon is
:math:`B_E(TH) = \\int_0^{TH} \\Delta T_E(t)\\,dt`.

Real inventory data arrive as a table of annual values. A piecewise linear
interpolant of such a table is written here as a sum of shifted ramp and step
atoms, which keeps the pathway inside the closed poly exponential algebra of
:mod:`cipo.expsum`. Both integrals above are then exact, with no time grid and
no quadrature error, for observational pathways as well as for the idealised
pulses used in the analytical results.
"""

from __future__ import annotations

from typing import Dict, Iterable, Mapping, Optional, Sequence

import numpy as np

from .climate import ClimateModel
from .expsum import ExpSum, Signal

__all__ = [
    "pulse",
    "step",
    "ramp",
    "piecewise_linear",
    "exponential_decline",
    "EmissionPathway",
]


def pulse(mass: float = 1.0, t0: float = 0.0) -> Signal:
    """An instantaneous emission of ``mass`` kg at ``t0``."""
    return Signal.delta(mass, t0)


def step(rate: float, t0: float = 0.0, t1: Optional[float] = None) -> Signal:
    """A constant emission ``rate`` in kg yr^-1 over ``[t0, t1)``."""
    if t1 is None:
        return Signal.from_expsum(ExpSum.constant(rate), t0)
    return Signal.window(rate, t0, t1)


def ramp(slope: float, t0: float = 0.0) -> Signal:
    """A rate rising linearly as ``slope * (t - t0)`` for ``t >= t0``."""
    return Signal.from_expsum(ExpSum([(0.0, [0.0, slope])]), t0)


def piecewise_linear(times: Sequence[float], rates: Sequence[float]) -> Signal:
    """Exact representation of a piecewise linear emission rate history.

    ``rates[i]`` is the emission rate in kg yr^-1 at ``times[i]``; the rate is
    interpolated linearly between the nodes and is zero outside
    ``[times[0], times[-1]]``.

    The construction sums one step atom at the first node and one ramp atom at
    every node where the slope changes, so the result is a ``Signal`` whose
    convolutions and integrals are available in closed form.
    """
    t = np.asarray(times, dtype=float)
    y = np.asarray(rates, dtype=float)
    if t.ndim != 1 or t.size < 2 or t.size != y.size:
        raise ValueError("times and rates must be matching 1-D arrays of length >= 2")
    if np.any(np.diff(t) <= 0):
        raise ValueError("times must be strictly increasing")

    sig = Signal()
    # Initial value as a step, then accumulate changes in slope.
    sig = sig + step(float(y[0]), float(t[0]))
    slopes = np.diff(y) / np.diff(t)
    previous = 0.0
    for i, s in enumerate(slopes):
        sig = sig + ramp(float(s - previous), float(t[i]))
        previous = float(s)
    # Switch the pathway off after the last node: remove the final level and
    # cancel the trailing slope.
    sig = sig + step(-float(y[-1]), float(t[-1]))
    sig = sig + ramp(-previous, float(t[-1]))
    return sig


def exponential_decline(
    initial_rate: float, decline_time: float, t0: float = 0.0
) -> Signal:
    """Emissions falling as ``initial_rate * exp(-(t - t0) / decline_time)``."""
    return Signal.from_expsum(
        ExpSum.exponential(initial_rate, 1.0 / decline_time), t0
    )


class EmissionPathway:
    """A named collection of species specific emission signals.

    Parameters
    ----------
    emissions
        Mapping from species name (a key of the climate model) to a
        :class:`~cipo.expsum.Signal` giving the emission rate in kg yr^-1, or
        a Dirac mass in kg for pulse emissions.
    """

    def __init__(self, emissions: Mapping[str, Signal], name: str = "") -> None:
        self.emissions: Dict[str, Signal] = dict(emissions)
        self.name = name
        self._response_cache: Dict[tuple, Signal] = {}

    # ------------------------------------------------------------ combination
    def __add__(self, other: "EmissionPathway") -> "EmissionPathway":
        merged = dict(self.emissions)
        for k, v in other.emissions.items():
            merged[k] = merged[k] + v if k in merged else v
        return EmissionPathway(merged, self.name or other.name)

    def scaled(self, factor: float) -> "EmissionPathway":
        return EmissionPathway(
            {k: v.scaled(factor) for k, v in self.emissions.items()}, self.name
        )

    def species(self) -> Sequence[str]:
        return tuple(self.emissions)

    # -------------------------------------------------------------- responses
    def response(self, cm: ClimateModel) -> Signal:
        """The warming signal ``Delta T_E(t)`` in K.

        Cached per climate model, keyed on the model parameters instead of on
        the object identity: an identity key is unsafe here because a model
        built inside an uncertainty loop is collected at the end of the
        iteration and its address is reused for the next draw.
        """
        key = cm.signature()
        if key not in self._response_cache:
            total = Signal()
            for name, emis in self.emissions.items():
                total = total + emis.convolve_kernel(cm.agtp_kernel(name))
            self._response_cache[key] = total
        return self._response_cache[key]

    def temperature(self, cm: ClimateModel, t) -> np.ndarray:
        return self.response(cm).eval(t)

    def cumulative_warming(self, cm: ClimateModel, horizon) -> np.ndarray:
        """``B_E(TH)``, the cumulative warming in K yr."""
        return self.response(cm).definite(horizon)

    def total_emission(self, species: str, horizon) -> np.ndarray:
        """Cumulative emission of one species over the horizon, kg."""
        return self.emissions[species].definite(horizon)

    def co2_equivalent(self, cm: ClimateModel, horizon: float, metric: str = "GWP"):
        """Cumulative emissions expressed as CO2 equivalent, kg.

        Provided for comparison with conventional inventory accounting, which
        the portfolio framework replaces. ``metric`` is ``"GWP"``, ``"GTP"`` or
        ``"iGTP"``.
        """
        fn = {"GWP": cm.gwp, "GTP": cm.gtp, "iGTP": cm.igtp}[metric]
        total = 0.0
        for name in self.emissions:
            total += float(self.total_emission(name, horizon)) * float(
                fn(name, horizon)
            )
        return total

    def __repr__(self) -> str:  # pragma: no cover - debugging aid
        return "EmissionPathway({}, species={})".format(
            self.name or "unnamed", list(self.emissions)
        )
