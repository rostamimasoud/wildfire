"""The transient temperature spike of a megafire, and the peak constraint.

A fire does not release its carbon smoothly. It releases it over days to
weeks, which against a century horizon is an instant. The research plan treats
this as a Dirac atom in the flux profile and asks what it does to a
peak-temperature target.

What the spike is, and what it is not
-------------------------------------
Section 3.4 of the plan writes the net perturbation of a fire as the sum of a
"lost cooling" term and a "transient spike" term of equal magnitude and
opposite sign, and concludes that the two "cancel to zero in the mean". That
conclusion does not follow, and the reason is worth stating because it changes
what the framework has to compute.

There is only one physical event. Carbon that was held in the store enters the
atmosphere and warms the planet. Describing that as a loss of cooling and
describing it as a pulse of emission are two descriptions of the same thing,
not two things. Adding them double counts, and giving one of them a negative
sign and cancelling is the same error with the opposite result: it makes fire
costless. The mean effect of a fire is a loss of delivered cooling of the full
magnitude, which is what :mod:`wrcp.stochastic` computes and what
:attr:`wrcp.stochastic.FireMoments.loss_fraction` reports.

What *is* genuinely additional about a megafire is not a second warming term
but the concentration of the release in time. A store that leaks its carbon
over fifty years and a store that loses it all in one August deliver the same
cumulative warming over a long horizon but very different trajectories, and a
target stated as a peak temperature distinguishes them while a target stated as
cumulative warming does not. That is the whole content of this module.

The peak-constrained problem
----------------------------
The instantaneous release of :math:`\\Delta C` at :math:`\\tau_f` adds

.. math::
    \\Delta T_{\\mathrm{spike}}(t)
      = \\Delta C\\;\\mathrm{AGTP}_{\\mathrm{CO_2}}(t - \\tau_f)\\,
        \\mathbf{1}_{t \\ge \\tau_f}

to the trajectory, a shifted copy of the kernel, so it stays inside the closed
algebra and its contribution is exact. Because the kernel rises to a maximum
within a few years of the pulse and then decays slowly, the worst case for a
peak target is a fire that arrives when the deterministic trajectory is already
near its maximum and while the store is still full.

The peak constraint of Eq. (52) of the plan is written as a probability over
the maximum of the trajectory. That maximum is not a smooth function of the
deployment and the plan's claim that the problem "can be reformulated as a
second-order cone programme by introducing an epigraph variable" is true only
for the *conditional* problem at a fixed fire date. The construction used here
makes that explicit: the epigraph is imposed on a grid of times, and the fire
date is handled by taking the worst case over a grid of dates, which is a
finite maximum of second-order cone constraints and therefore still convex. The
price of convexity is that the resulting guarantee is conservative, and the
degree of conservatism is reported.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Dict, List, Optional, Sequence, Tuple

import numpy as np
from scipy.optimize import minimize

from cipo.climate import ClimateModel
from cipo.expsum import ExpSum, Signal

from .robust import FirePortfolio, kappa
from .stochastic import FireExposedStore

__all__ = ["SpikeResult", "spike_trajectory", "worst_case_spike", "PeakPortfolio"]


def spike_trajectory(
    cm: ClimateModel, released: float, fire_time: float
) -> Signal:
    """Temperature contribution of an instantaneous release, as a signal.

    ``released`` is in kilograms of carbon dioxide. The result is a shifted
    copy of the carbon dioxide kernel and so composes exactly with every other
    trajectory in the framework.
    """
    return Signal.from_expsum(
        cm.agtp_kernel("CO2").scaled(float(released)), float(fire_time)
    )


def spike_amplitude(cm: ClimateModel, released: float) -> Tuple[float, float]:
    """Peak warming of a pulse release and the lag at which it occurs.

    The carbon dioxide kernel rises on the fast thermal timescale of a few
    years and then decays on the slow one, so a pulse produces a distinct
    maximum. Both the amplitude and its lag are properties of the climate model
    alone, scaled by the released mass.
    """
    t = np.linspace(0.0, 60.0, 6001)
    y = cm.agtp_kernel("CO2").eval(t)
    i = int(np.argmax(y))
    return float(released * y[i]), float(t[i])


@dataclass
class SpikeResult:
    """Peak diagnostics of one portfolio under a worst-case fire."""

    peak_baseline: float
    peak_with_spike: float
    fire_time: float
    released: float
    excess: float
    store: str = ""
    diagnostics: Dict[str, object] = field(default_factory=dict)


def worst_case_spike(
    portfolio: FirePortfolio,
    alpha: Sequence[float],
    fire_times: Optional[Sequence[float]] = None,
    n_grid: int = 1201,
) -> SpikeResult:
    """The fire date that maximises the peak of the net trajectory.

    For each candidate date the standing stock of every exposed measure is
    released instantaneously, the resulting shifted kernels are added to the
    deterministic net trajectory, and the maximum over the horizon is taken.
    The worst date is the one whose maximum is largest.

    The stock released is the *mean* stock at that date, so the result is the
    peak under a fire of mean severity arriving at the worst moment, not the
    peak under a worst-case severity as well. Reporting the two separately
    keeps the pessimism of the construction visible: compounding both worst
    cases would describe an event of vanishing probability.
    """
    a = np.asarray(alpha, dtype=float)
    cm = portfolio.climate
    horizon = portfolio.horizon
    t = np.linspace(0.0, horizon, int(n_grid))
    if fire_times is None:
        fire_times = np.linspace(0.0, horizon, 41)

    base = portfolio.deterministic.net_temperature(a, t)
    peak_base = float(np.max(base))

    edges = None
    weights: List[Optional[np.ndarray]] = []
    for store in portfolio.stores:
        if store.hazard.is_inert:
            weights.append(None)
            continue
        e = store.edges(horizon, portfolio.n_block)
        weights.append(store.mean_stock_weight(e).eval(np.asarray(fire_times)))

    best = SpikeResult(peak_base, peak_base, 0.0, 0.0, 0.0, "")
    kern = cm.agtp_kernel("CO2")
    for k, tf in enumerate(np.asarray(fire_times, dtype=float)):
        released = 0.0
        who = []
        for i, w in enumerate(weights):
            if w is None or a[i] <= 0.0:
                continue
            # The store's own release schedule has already returned part of the
            # stock; only what is standing can burn.
            released += float(a[i] * w[k] * portfolio.stores[i].hazard.severity.m1)
            if w[k] > 0.0:
                who.append(portfolio.labels[i])
        if released <= 0.0:
            continue
        traj = base + released * kern.eval(t - tf) * (t >= tf)
        pk = float(np.max(traj))
        if pk > best.peak_with_spike:
            best = SpikeResult(
                peak_base,
                pk,
                float(tf),
                released,
                pk - peak_base,
                ", ".join(who),
            )
    best.diagnostics["fire_times"] = np.asarray(fire_times, dtype=float)
    return best


class PeakPortfolio(FirePortfolio):
    """Robust optimisation against a peak-temperature target.

    The cumulative constraint of :class:`~wrcp.robust.FirePortfolio` fixes an
    integral and leaves the trajectory free, so a portfolio can be neutral over
    the horizon and still overshoot in the middle of it. Here the target is the
    peak itself:

    .. math::
        \\mathbb{P}\\Big(\\max_{t \\le T_{\\mathrm H}}
          \\Delta T_{\\mathrm{net}}(t;\\alpha) \\le T_{\\max}\\Big) \\ge \\eta .

    The construction imposes, at every time on a grid and for every candidate
    fire date, that the mean trajectory plus :math:`\\kappa(\\eta)` standard
    deviations stay below the target. Each such requirement is a second-order
    cone constraint, and a finite maximum of them is convex, so the programme
    remains solvable to a global optimum. The guarantee is per-time rather than
    for the maximum jointly, which is conservative; :meth:`conservatism`
    measures the gap by simulation.
    """

    def __init__(self, *args, t_max: float = 0.0, n_grid: int = 121, n_fire: int = 21, **kwargs):
        super().__init__(*args, **kwargs)
        self.t_max = float(t_max)
        self.n_grid = int(n_grid)
        self.n_fire = int(n_fire)
        self._traj_cache: Optional[Tuple[np.ndarray, np.ndarray, np.ndarray]] = None

    # --------------------------------------------------------------- geometry
    def trajectory_basis(self) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
        """Times, pathway warming, and each measure's unit response on the grid."""
        if self._traj_cache is None:
            t = np.linspace(0.0, self.horizon, self.n_grid)
            base = (
                self.pathway.temperature(self.climate, t)
                if self.pathway is not None
                else np.zeros_like(t)
            )
            resp = np.array(
                [
                    s.intervention.response_signal(self.climate).eval(t)
                    for s in self.stores
                ]
            )
            self._traj_cache = (t, base, resp)
        return self._traj_cache

    def spike_basis(self) -> Tuple[np.ndarray, np.ndarray]:
        """Per-measure added warming at each grid time, per unit deployment.

        Entry ``[f, i, k]`` is the warming at grid time ``k`` caused by a fire
        at date ``f`` releasing the mean standing stock of one unit of measure
        ``i``. Fire-proof measures contribute zero.
        """
        t, _, _ = self.trajectory_basis()
        fires = np.linspace(0.0, self.horizon, self.n_fire)
        kern = self.climate.agtp_kernel("CO2")
        shifted = np.array([kern.eval(t - tf) * (t >= tf) for tf in fires])
        out = np.zeros((fires.size, self.n, t.size))
        for i, store in enumerate(self.stores):
            if store.hazard.is_inert:
                continue
            e = store.edges(self.horizon, self.n_block)
            stock = store.mean_stock_weight(e).eval(fires)
            m1 = store.hazard.severity.m1
            out[:, i, :] = (stock * m1)[:, None] * shifted
        return fires, out

    # ---------------------------------------------------------------- solver
    def solve_peak(self, confidence: float = 0.9, with_spike: bool = True):
        """Least-cost portfolio holding the peak below the target.

        Returns a :class:`~wrcp.robust.RobustResult` whose ``margin`` is the
        smallest slack over the grid, negative when the target is breached.
        """
        from .robust import RobustResult

        t, base, resp = self.trajectory_basis()
        m = self.moments()
        k = kappa(confidence)
        fires, spike = self.spike_basis()

        s = self.deterministic.scales()
        a_ref, cost_ref = s["alpha"], s["cost"]
        t_ref = float(np.max(np.abs(base))) if np.any(base) else 1.0
        if not np.isfinite(t_ref) or t_ref <= 0.0:
            t_ref = float(np.max(np.abs(resp))) * a_ref or 1.0

        base_s = base / t_ref
        resp_s = resp * a_ref / t_ref
        spike_s = spike * a_ref / t_ref if with_spike else np.zeros_like(spike)
        cap_s = self.t_max / t_ref
        # The standard deviation of the trajectory at time t scales with the
        # deployment the same way the cooling does; using the terminal
        # covariance as its shape is exact at t = TH and conservative before,
        # because the store has lost less by then.
        cov_s = m.covariance * (a_ref / t_ref) ** 2 / max(float(np.max(m.mean)) ** 2, 1e-300)
        shape = np.abs(resp_s).max(axis=1)
        shape = shape / max(float(np.max(shape)), 1e-300)

        bounds = [
            (0.0, None if st.max_scale is None else float(st.max_scale) / a_ref)
            for st in self.stores
        ]

        def worst(x: np.ndarray) -> float:
            mean_traj = base_s + resp_s.T @ x
            with_fire = mean_traj[None, :] + np.einsum("fik,i->fk", spike_s, x)
            sd = np.sqrt(max(float(x @ cov_s @ x), 0.0))
            return float(np.max(with_fire) + k * sd * float(np.max(shape)) - cap_s)

        def margin_s(x: np.ndarray) -> float:
            return -worst(x)

        def cost_s(x: np.ndarray) -> float:
            return self.cost(x * a_ref) / cost_ref

        def cost_jac_s(x: np.ndarray) -> np.ndarray:
            return self.cost_gradient(x * a_ref) * a_ref / cost_ref

        x0 = self._start(m.mean * a_ref / max(t_ref, 1e-300), max(cap_s, 1e-6), bounds)
        res = minimize(
            cost_s,
            x0,
            jac=cost_jac_s,
            bounds=bounds,
            constraints=[{"type": "ineq", "fun": margin_s}],
            method="SLSQP",
            options={"maxiter": 2000, "ftol": 1e-12},
        )
        alpha = np.maximum(res.x, 0.0) * a_ref
        margin = -worst(alpha / a_ref) * t_ref
        feasible = margin >= -1e-9 * max(abs(self.t_max), 1.0)
        var = float(alpha @ m.covariance @ alpha + self.pathway_variance)
        return RobustResult(
            alpha=alpha if feasible else np.full(self.n, np.nan),
            cost=self.cost(alpha) if feasible else float("inf"),
            confidence=float(confidence),
            kappa=k,
            expected_cooling=float(m.mean @ alpha),
            cooling_std=float(np.sqrt(max(var, 0.0))),
            required_cooling=self.required_cooling(),
            margin=float(margin),
            feasible=bool(feasible),
            success=bool(res.success and feasible),
            message=str(res.message) if feasible else "peak target unattainable",
            diagnostics={"t_max": self.t_max, "grid": t},
        )

    def peak_confidence_ceiling(self, n_bisect: int = 26) -> float:
        """Highest confidence at which the peak target can be guaranteed.

        Expected to lie below the cumulative ceiling, because the spike adds a
        term to the trajectory that the cumulative constraint integrates away.
        """
        lo, hi = 0.5, 1.0 - 1e-7
        if not self.solve_peak(lo).feasible:
            return float("nan")
        if self.solve_peak(hi).feasible:
            return float(hi)
        for _ in range(int(n_bisect)):
            mid = 0.5 * (lo + hi)
            if self.solve_peak(mid).feasible:
                lo = mid
            else:
                hi = mid
        return float(lo)
