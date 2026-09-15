"""Effective lifetime, horizon sensitivity and optimal durability.

For an intervention :math:`i` and an emission species :math:`s`, the
**compensation ratio** is the deployment needed to cancel the cumulative
warming of one kilogram of that species over a horizon :math:`TH`,

.. math::
    \\alpha^{(i,s)}(TH)
        = \\frac{\\mathrm{iAGTP}_s(TH)}{b_i(TH)} ,
    \\qquad b_i(TH) = -\\int_0^{TH}\\mathrm{AGTP}_{F_i}(t)\\,dt .

Because the horizon is a policy convention and not a physical constant, a
compensation ratio is usable in accounting only to the extent that it is
stable against the choice of horizon. Differentiating gives

.. math::
    \\frac{\\partial \\alpha}{\\partial TH}
      = \\frac{\\mathrm{AGTP}_s(TH)\\, b_i(TH)
               + \\mathrm{iAGTP}_s(TH)\\, \\mathrm{AGTP}_{F_i}(TH)}
              {b_i(TH)^2} ,

which is Eq. (28) of the research plan with the sign convention
:math:`b_i = -\\mathrm{iAGTP}_{F_i}` made explicit, and the dimensionless
**horizon sensitivity**

.. math::
    S^{(i,s)}(TH)
      = \\left| \\frac{\\partial \\ln \\alpha}{\\partial \\ln TH} \\right| .

Two corrections to the research plan
------------------------------------
*Dimensional consistency.* Definition 8 of the plan sets a threshold by
requiring :math:`|\\partial \\alpha / \\partial TH| < \\epsilon` with
:math:`\\epsilon = 0.01`. That derivative carries units of inverse time, so
its numerical value changes with the unit in which the horizon is counted and
the same intervention passes or fails depending on whether years or decades
are used. Its scale also varies between gases, so one :math:`\\epsilon` cannot
serve several. The logarithmic sensitivity :math:`S` above is dimensionless
and comparable across gases and units: a one per cent change in the horizon
changes the required compensation by at most :math:`S` per cent.

*Existence.* The plan asserts that above a threshold durability an
intervention "behaves like a permanent offset" and the compensation ratio
becomes horizon insensitive. That is false whenever a carbon based measure
offsets a different gas, and the framework here shows why. As the horizon
grows, :math:`\\mathrm{iAGTP}_s` saturates for any gas with a finite lifetime,
while :math:`b_i` grows without bound for a durable carbon pool, because the
airborne fraction of carbon dioxide retains the permanent component
:math:`a_0 > 0`. Hence :math:`\\alpha \\sim \\mathrm{const}/TH` and
:math:`S \\to 1` in the durable limit. At the other extreme a pool that
releases its carbon almost at once delivers little cooling and :math:`S` is
again large. The sensitivity is therefore a U shaped function of the mean
storage time with an interior minimum: there exists an **optimal durability**
that makes the compensation ratio most nearly horizon free, and an
**irreducible sensitivity floor** below which no durability can go. The set of
admissible durabilities for a given tolerance is a bounded interval, and it is
empty for tolerances below the floor.

Exact horizon invariance is attainable only when the intervention removes the
same forcing agent it is meant to offset, in which case
:math:`\\alpha \\equiv 1` for every horizon. That case is included below as a
limiting check.
"""

from __future__ import annotations

from typing import Callable, Dict, Optional, Sequence, Tuple

import numpy as np

from .climate import ClimateModel
from .profiles import Intervention

__all__ = [
    "compensation_ratio",
    "compensation_sensitivity",
    "log_sensitivity",
    "max_log_sensitivity",
    "sensitivity_profile",
    "optimal_durability",
    "durability_interval",
    "durability_table",
    "breakeven_storage_time",
]


def compensation_ratio(
    cm: ClimateModel, intervention: Intervention, species: str, horizon
) -> np.ndarray:
    """The compensation ratio, in units of intervention per kg of species."""
    return cm.iagtp(species, horizon) / intervention.cooling(cm, horizon)


def compensation_sensitivity(
    cm: ClimateModel, intervention: Intervention, species: str, horizon
) -> np.ndarray:
    """``d alpha / d TH``, in units of intervention per kg per year."""
    th = np.asarray(horizon, dtype=float)
    b = intervention.cooling(cm, th)
    return (
        cm.agtp(species, th) * b + cm.iagtp(species, th) * intervention.response(cm, th)
    ) / (b * b)


def log_sensitivity(
    cm: ClimateModel, intervention: Intervention, species: str, horizon
) -> np.ndarray:
    """The dimensionless horizon sensitivity ``|d ln alpha / d ln TH|``."""
    th = np.asarray(horizon, dtype=float)
    ratio = compensation_ratio(cm, intervention, species, th)
    deriv = compensation_sensitivity(cm, intervention, species, th)
    with np.errstate(divide="ignore", invalid="ignore"):
        out = np.abs(th * deriv / ratio)
    return np.where(np.isfinite(out), out, np.inf)


def max_log_sensitivity(
    cm: ClimateModel,
    intervention: Intervention,
    species: str,
    horizon_min: float = 20.0,
    horizon_max: float = 500.0,
    n: int = 241,
) -> float:
    """Worst horizon sensitivity over a horizon window."""
    th = np.geomspace(horizon_min, horizon_max, n)
    return float(np.max(log_sensitivity(cm, intervention, species, th)))


def sensitivity_profile(
    cm: ClimateModel,
    family: Callable[[float], Intervention],
    species: str,
    storage_times: Optional[Sequence[float]] = None,
    horizon_min: float = 20.0,
    horizon_max: float = 500.0,
    n_horizon: int = 241,
) -> Dict[str, np.ndarray]:
    """Worst horizon sensitivity as a function of mean storage time.

    ``family`` maps a mean storage time in years to an intervention. Using the
    mean storage time as the argument, in place of each family's own shape
    parameter, is what makes results comparable between profile shapes.
    """
    taus = (
        np.asarray(storage_times, dtype=float)
        if storage_times is not None
        else np.geomspace(1.0, 1.0e5, 120)
    )
    s = np.array(
        [
            max_log_sensitivity(
                cm, family(float(t)), species, horizon_min, horizon_max, n_horizon
            )
            for t in taus
        ]
    )
    return {"storage_time": taus, "sensitivity": s}


def optimal_durability(
    cm: ClimateModel,
    family: Callable[[float], Intervention],
    species: str,
    horizon_min: float = 20.0,
    horizon_max: float = 500.0,
    search_range: Tuple[float, float] = (1.0, 1.0e5),
    n_scan: int = 96,
    n_refine: int = 60,
) -> Dict[str, float]:
    """Mean storage time minimising the worst horizon sensitivity.

    Located by a coarse geometric scan followed by a golden section search on
    the bracketing interval. Returns the optimal mean storage time, the
    irreducible sensitivity floor attained there, and the sensitivity in the
    durable limit for comparison.
    """
    taus = np.geomspace(search_range[0], search_range[1], n_scan)
    vals = np.array(
        [
            max_log_sensitivity(cm, family(float(t)), species, horizon_min, horizon_max)
            for t in taus
        ]
    )
    i = int(np.argmin(vals))
    lo = taus[max(i - 1, 0)]
    hi = taus[min(i + 1, n_scan - 1)]

    def f(t: float) -> float:
        return max_log_sensitivity(
            cm, family(float(t)), species, horizon_min, horizon_max
        )

    # Golden section search in log space.
    phi = 0.5 * (np.sqrt(5.0) - 1.0)
    a, b = np.log(lo), np.log(hi)
    c, d = b - phi * (b - a), a + phi * (b - a)
    fc, fd = f(np.exp(c)), f(np.exp(d))
    for _ in range(n_refine):
        if fc < fd:
            b, d, fd = d, c, fc
            c = b - phi * (b - a)
            fc = f(np.exp(c))
        else:
            a, c, fc = c, d, fd
            d = a + phi * (b - a)
            fd = f(np.exp(d))
    t_opt = float(np.exp(0.5 * (a + b)))
    return {
        "optimal_storage_time": t_opt,
        "sensitivity_floor": float(f(t_opt)),
        "sensitivity_durable_limit": float(f(search_range[1])),
        "compensation_ratio_at_optimum": float(
            compensation_ratio(cm, family(t_opt), species, horizon_max)
        ),
    }


def durability_interval(
    cm: ClimateModel,
    family: Callable[[float], Intervention],
    species: str,
    tolerance: float = 0.35,
    horizon_min: float = 20.0,
    horizon_max: float = 500.0,
    search_range: Tuple[float, float] = (1.0, 1.0e5),
    n_bisect: int = 30,
    n_horizon: int = 121,
    optimum: Optional[Dict[str, float]] = None,
) -> Dict[str, float]:
    """Admissible mean storage times for a given sensitivity tolerance.

    Returns the bounded interval :math:`[\\tau_{\\mathrm{lo}},
    \\tau_{\\mathrm{hi}}]` on which the horizon sensitivity does not exceed
    ``tolerance``, together with the optimum inside it. The interval is empty,
    reported as ``nan`` bounds, when the tolerance lies below the irreducible
    floor.

    ``optimum`` accepts the output of :func:`optimal_durability` so that a
    sweep over tolerances locates the optimum once instead of repeating it.
    """
    opt = optimum or optimal_durability(
        cm, family, species, horizon_min, horizon_max, search_range
    )
    t_opt = opt["optimal_storage_time"]
    floor = opt["sensitivity_floor"]
    out = dict(opt)
    out.update({"tolerance": tolerance, "lower": float("nan"), "upper": float("nan")})
    if floor > tolerance:
        out["feasible"] = False
        return out
    out["feasible"] = True

    def f(t: float) -> float:
        return max_log_sensitivity(
            cm, family(float(t)), species, horizon_min, horizon_max, n_horizon
        )

    def bisect(a: float, b: float) -> float:
        """Find the crossing of ``f = tolerance`` between ``a`` and ``b``."""
        fa = f(a)
        for _ in range(n_bisect):
            m = np.sqrt(a * b)
            fm = f(m)
            if (fm > tolerance) == (fa > tolerance):
                a, fa = m, fm
            else:
                b = m
        return float(np.sqrt(a * b))

    lo_end = search_range[0]
    out["lower"] = lo_end if f(lo_end) <= tolerance else bisect(lo_end, t_opt)
    hi_end = search_range[1]
    out["upper"] = hi_end if f(hi_end) <= tolerance else bisect(hi_end, t_opt)
    return out


def durability_table(
    cm: ClimateModel,
    families: Dict[str, Callable[[float], Intervention]],
    species: Sequence[str],
    tolerances: Sequence[float] = (0.35,),
    horizon_min: float = 20.0,
    horizon_max: float = 500.0,
):
    """Optimal durability and admissible interval for every combination."""
    import pandas as pd

    rows = []
    for fam_name, fam in families.items():
        for sp in species:
            base = optimal_durability(cm, fam, sp, horizon_min, horizon_max)
            for tol in tolerances:
                iv = durability_interval(
                    cm, fam, sp, tol, horizon_min, horizon_max, optimum=base
                )
                rows.append(
                    {
                        "family": fam_name,
                        "species": sp,
                        "horizon_min": horizon_min,
                        "horizon_max": horizon_max,
                        "optimal_storage_time_yr": base["optimal_storage_time"],
                        "sensitivity_floor": base["sensitivity_floor"],
                        "sensitivity_durable_limit": base[
                            "sensitivity_durable_limit"
                        ],
                        "tolerance": tol,
                        "feasible": iv["feasible"],
                        "interval_lower_yr": iv["lower"],
                        "interval_upper_yr": iv["upper"],
                    }
                )
    return pd.DataFrame(rows)


def breakeven_storage_time(
    cm: ClimateModel,
    family: Callable[[float], Intervention],
    unit_cost: Callable[[float], float],
    permanent_cost: float,
    horizon: float = 100.0,
    search_range: Tuple[float, float] = (1.0, 1.0e5),
    n_bisect: int = 80,
) -> float:
    """Mean storage time at which a temporary measure matches permanent removal.

    Cost effectiveness is measured as cost per unit cumulative cooling. The
    permanent benchmark delivers :math:`\\mathrm{iAGTP}_{\\mathrm{CO_2}}(TH)`
    per unit at cost ``permanent_cost``. Returns the mean storage time where
    the two ratios are equal, or ``inf`` when the temporary option cannot match
    the benchmark at any durability.
    """
    from .profiles import PermanentRemoval

    target = permanent_cost / float(PermanentRemoval().cooling(cm, horizon))

    def gap(tau_bar: float) -> float:
        return unit_cost(tau_bar) / float(
            family(tau_bar).cooling(cm, horizon)
        ) - target

    lo, hi = search_range
    if gap(hi) > 0:
        return float("inf")
    if gap(lo) <= 0:
        return float(lo)
    a, b = lo, hi
    for _ in range(n_bisect):
        mid = np.sqrt(a * b)
        if gap(mid) > 0:
            a = mid
        else:
            b = mid
    return float(b)
