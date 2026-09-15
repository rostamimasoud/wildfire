"""Deployment timing under fire risk.

The static problem asks how much of each measure to buy. This one asks when.
Timing matters under fire risk for a reason it does not matter in the
deterministic framework: exposure accumulates. A store established today is
exposed to the hazard for the whole horizon, and it is exposed to the early
part of the horizon, when it is still full and so has the most to lose. A store
established in forty years faces a hazard that has intensified with the climate
but for a shorter time and with less standing carbon at risk.

Which effect dominates is not obvious in advance, and it is the question this
module answers. Writing :math:`u_{i,n} \\ge 0` for the deployment of measure
:math:`i` begun at :math:`t_n`, its stock at time :math:`t` is
:math:`A_i^{\\mathrm{det}}(t - t_n)` and its survival multiplier accumulates
hazard only from :math:`t_n`,

.. math::
    \\mathbb{E}\\big[S_{i,n}(t)\\big]
      = \\exp\\!\\big(-m_1 [\\Lambda_i(t) - \\Lambda_i(t_n)]\\big) ,
    \\qquad t \\ge t_n ,

so the mean and covariance of its delivered cooling follow from exactly the
same closed forms as the static case, with the cumulative hazard measured from
the start date. The decision variables enter the chance constraint linearly and
the objective convexly, so the fire-aware scheduling problem is a second-order
cone programme, as Eq. (31) of the research plan states.

Cost
----
The problem is posed with a cost that scales each decision epoch's outlay by a
discount factor. Without discounting, delay is free and the schedule would be
determined by the physics alone; with it, the trade-off between deferring
exposure and deferring benefit is priced. Both are reported, because the
undiscounted schedule isolates the physical effect that is the point of the
exercise.

Cost of computation
-------------------
The covariance of the scheduled problem has one row per measure and epoch, so
the number of exact variance integrals grows as the square of the product. That
is affordable for the handful of epochs a policy actually chooses between, and
the defaults here are set accordingly; a finer grid is available but should be
paired with a coarser hazard block count.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Dict, List, Optional, Sequence, Tuple

import numpy as np
from scipy.optimize import minimize

from cipo.climate import ClimateModel
from cipo.expsum import ExpSum
from cipo.pathways import EmissionPathway
from cipo.portfolio import Portfolio

from .algebra import (
    Piecewise,
    breakpoints,
    merge_edges,
    reverse_weighted_integral,
    triangle_integral,
)
from .hazard import cross_severity_exponent, shared_growth, shared_intensity
from .robust import kappa
from .stochastic import FireExposedStore, block_edges

__all__ = ["ScheduleResult", "FireSchedule"]


@dataclass
class ScheduleResult:
    """A deployment schedule and its diagnostics."""

    times: np.ndarray
    deployment: np.ndarray  # (n_measures, n_times)
    cost: float
    discounted_cost: float
    confidence: float
    expected_cooling: float
    cooling_std: float
    required_cooling: float
    margin: float
    feasible: bool
    success: bool
    message: str = ""
    diagnostics: Dict[str, object] = field(default_factory=dict)

    def totals(self) -> np.ndarray:
        return self.deployment.sum(axis=1)

    def centre_of_mass(self) -> np.ndarray:
        """Deployment-weighted mean start date of each measure, in years."""
        out = np.full(self.deployment.shape[0], np.nan)
        for i, row in enumerate(self.deployment):
            tot = float(row.sum())
            if tot > 0:
                out[i] = float(self.times @ row / tot)
        return out

    def frame(self, labels: Sequence[str]):
        import pandas as pd

        return pd.DataFrame(
            self.deployment.T, index=self.times, columns=list(labels)
        ).rename_axis("start_year")


class FireSchedule:
    """Optimal timing of fire-exposed deployment over a decision horizon."""

    def __init__(
        self,
        climate: ClimateModel,
        stores: Sequence[FireExposedStore],
        pathway: Optional[EmissionPathway],
        horizon: float = 100.0,
        decision_times: Sequence[float] = (0.0, 15.0, 30.0, 45.0),
        n_block: int = 8,
        rho_within: float = 0.6,
        rho_between: float = 0.15,
        sharing: float = 0.5,
        weather_spread: float = 0.35,
        n_weather: int = 5,
        pathway_variance: float = 0.0,
    ) -> None:
        self.climate = climate
        self.stores = list(stores)
        self.pathway = pathway
        self.horizon = float(horizon)
        self.times = np.asarray(decision_times, dtype=float)
        self.n_block = int(n_block)
        self.rho_within = float(rho_within)
        self.rho_between = float(rho_between)
        self.sharing = float(sharing)
        self.weather_spread = float(weather_spread)
        self.n_weather = int(n_weather)
        self.pathway_variance = float(pathway_variance)
        self._base = Portfolio(
            climate, [s.intervention for s in self.stores], pathway, self.horizon
        )
        self._moments: Optional[Tuple[np.ndarray, np.ndarray]] = None

    # ---------------------------------------------------------------- shapes
    @property
    def n_iv(self) -> int:
        return len(self.stores)

    @property
    def n_t(self) -> int:
        return self.times.size

    @property
    def n_var(self) -> int:
        return self.n_iv * self.n_t

    @property
    def labels(self) -> List[str]:
        return [s.label for s in self.stores]

    def variable_labels(self) -> List[str]:
        return [
            "{} @ {:g} yr".format(s.label, t)
            for s in self.stores
            for t in self.times
        ]

    def _edges(self) -> np.ndarray:
        sets = [block_edges(self.horizon, self.n_block), self.times]
        for s in self.stores:
            shifted = [
                float(t0 + t)
                for t0, _ in s.stock().pieces
                for t in self.times
                if 0.0 < t0 + t < self.horizon
            ]
            if shifted:
                sets.append(np.array(shifted))
        return merge_edges(*sets)

    # --------------------------------------------------------------- weights
    def _weights(self, edges: np.ndarray, factor: float):
        """Mean-stock and autocovariance weights for every (measure, epoch).

        The survival multiplier of a deployment begun at ``t0`` accumulates
        hazard only from ``t0``, which is obtained by rescaling the hazard's
        own survival factor by its value at ``t0``. Nothing else changes, so the
        static closed forms carry over unmodified.
        """
        mean_w: List[Optional[Piecewise]] = []
        cov_w: List[Optional[Piecewise]] = []
        hazards = []
        for s in self.stores:
            h = s.hazard.scaled(factor)
            cum = h.cumulative(edges)
            at = np.interp(self.times, edges, cum)
            for k, t0 in enumerate(self.times):
                stock = Piecewise.from_signal(s.stock().shifted(float(t0)), edges)
                hazards.append(h)
                if h.is_inert:
                    mean_w.append(stock)
                    cov_w.append(None)
                    continue
                surv = h.survival(edges).scaled(
                    float(np.exp(h.severity.m1 * at[k]))
                )
                f = stock.product(surv)
                # exp(m2 [Lambda(t) - Lambda(t0)]) - 1, zero at the start date.
                grow = h.growth(edges, h.severity.m2)
                shifted_grow = Piecewise(
                    edges,
                    [
                        (b + ExpSum.constant(1.0)).scaled(
                            float(np.exp(-h.severity.m2 * at[k]))
                        )
                        + ExpSum.constant(-1.0)
                        for b in grow.blocks
                    ],
                )
                mean_w.append(f)
                cov_w.append(f.product(shifted_grow))
        return mean_w, cov_w, hazards

    def moments(self) -> Tuple[np.ndarray, np.ndarray]:
        """Mean and covariance of the scheduled cooling vector."""
        if self._moments is not None:
            return self._moments
        edges = self._edges()
        kern = self.climate.agtp_kernel("CO2")
        m_var = self.n_var

        def conditional(factor: float) -> Tuple[np.ndarray, np.ndarray]:
            mean_w, cov_w, hazards = self._weights(edges, factor)
            mu = np.array(
                [
                    reverse_weighted_integral(w, kern, self.horizon, edges=edges)
                    for w in mean_w
                ]
            )
            cov = np.zeros((m_var, m_var))
            for p in range(m_var):
                if cov_w[p] is None:
                    continue
                cov[p, p] = 2.0 * triangle_integral(
                    mean_w[p], cov_w[p], kern, self.horizon, edges=edges
                )
            for p in range(m_var):
                for q in range(p + 1, m_var):
                    if cov_w[p] is None or cov_w[q] is None:
                        continue
                    hp, hq = hazards[p], hazards[q]
                    theta = shared_intensity(
                        hp, hq, edges, self.rho_within, self.rho_between
                    )
                    if not np.any(theta > 0.0):
                        continue
                    m_eff = cross_severity_exponent(
                        hp.severity, hq.severity, self.sharing
                    )
                    if m_eff <= 0.0:
                        continue
                    # The shared hazard is only shared while both deployments
                    # exist, so it is measured from the later start date.
                    t_start = max(
                        float(self.times[p % self.n_t]),
                        float(self.times[q % self.n_t]),
                    )
                    live = 0.5 * (edges[:-1] + edges[1:]) >= t_start
                    grow = shared_growth(theta * live, edges, m_eff)
                    v = triangle_integral(
                        mean_w[p],
                        mean_w[q].product(grow),
                        kern,
                        self.horizon,
                        edges=edges,
                    ) + triangle_integral(
                        mean_w[q],
                        mean_w[p].product(grow),
                        kern,
                        self.horizon,
                        edges=edges,
                    )
                    cov[p, q] = cov[q, p] = float(v)
            return mu, cov

        if self.weather_spread <= 0.0:
            mu, cov = conditional(1.0)
        else:
            x, w = np.polynomial.hermite_e.hermegauss(self.n_weather)
            w = w / np.sum(w)
            sig = self.weather_spread
            factors = np.exp(sig * x - 0.5 * sig * sig)
            mus, covs = [], []
            for f in factors:
                a, b = conditional(float(f))
                mus.append(a)
                covs.append(b)
            mus = np.array(mus)
            mu = w @ mus
            cov = np.tensordot(w, np.array(covs), axes=(0, 0))
            dev = mus - mu
            cov = cov + (dev * w[:, None]).T @ dev
        vals, vecs = np.linalg.eigh(0.5 * (cov + cov.T))
        cov = (vecs * np.clip(vals, 0.0, None)) @ vecs.T
        self._moments = (mu, 0.5 * (cov + cov.T))
        return self._moments

    # ---------------------------------------------------------------- solver
    def cost(self, u: Sequence[float], discount_rate: float = 0.0) -> float:
        u = np.asarray(u, dtype=float).reshape(self.n_iv, self.n_t)
        total = 0.0
        for i, s in enumerate(self.stores):
            for k, t0 in enumerate(self.times):
                total += float(np.exp(-discount_rate * t0)) * s.cost(u[i, k])
        return float(total)

    def cost_gradient(self, u: Sequence[float], discount_rate: float = 0.0) -> np.ndarray:
        u = np.asarray(u, dtype=float).reshape(self.n_iv, self.n_t)
        out = np.zeros_like(u)
        for i, s in enumerate(self.stores):
            for k, t0 in enumerate(self.times):
                out[i, k] = float(np.exp(-discount_rate * t0)) * s.marginal_cost(
                    u[i, k]
                )
        return out.ravel()

    def solve(
        self,
        confidence: float = 0.9,
        discount_rate: float = 0.0,
        max_rate: Optional[Sequence[float]] = None,
    ) -> ScheduleResult:
        """Least-cost schedule guaranteeing neutrality at ``confidence``."""
        mu, cov = self.moments()
        be = float(self._base.cumulative_warming())
        k = kappa(confidence)

        s = self._base.scales()
        a_ref = s["alpha"] / max(self.n_t, 1)
        cost_ref = self.cost(np.full(self.n_var, a_ref), discount_rate) or 1.0
        row = float(max(abs(be), float(np.max(np.abs(mu))) * a_ref, 1e-300))
        mu_s = mu * a_ref / row
        cov_s = cov * (a_ref / row) ** 2
        var_e = self.pathway_variance / (row * row)
        be_s = be / row

        bounds = []
        for i, st in enumerate(self.stores):
            per = None if max_rate is None else float(max_rate[i]) / a_ref
            bounds.extend([(0.0, per)] * self.n_t)

        cons = [
            {
                "type": "ineq",
                "fun": lambda x: float(
                    mu_s @ x
                    - k * np.sqrt(max(float(x @ cov_s @ x + var_e), 0.0))
                    - be_s
                ),
                "jac": lambda x: (
                    mu_s
                    - k
                    * (cov_s @ x)
                    / max(np.sqrt(max(float(x @ cov_s @ x + var_e), 0.0)), 1e-300)
                ),
            }
        ]
        for i, st in enumerate(self.stores):
            if st.max_scale is None:
                continue
            idx = list(range(i * self.n_t, (i + 1) * self.n_t))
            cap_s = float(st.max_scale) / a_ref
            cons.append(
                {
                    "type": "ineq",
                    "fun": (lambda x, idx=idx, cap_s=cap_s: cap_s - float(np.sum(x[idx]))),
                    "jac": (
                        lambda x, idx=idx: -np.eye(self.n_var)[idx].sum(axis=0)
                    ),
                }
            )

        x0 = np.full(
            self.n_var, max(be_s, 1e-9) / max(float(np.sum(np.abs(mu_s))), 1e-12)
        )
        res = minimize(
            lambda x: self.cost(x * a_ref, discount_rate) / cost_ref,
            np.maximum(x0, 1e-9),
            jac=lambda x: self.cost_gradient(x * a_ref, discount_rate)
            * a_ref
            / cost_ref,
            bounds=bounds,
            constraints=cons,
            method="SLSQP",
            options={"maxiter": 3000, "ftol": 1e-14},
        )
        u = np.maximum(res.x, 0.0) * a_ref
        var = float(u @ cov @ u + self.pathway_variance)
        margin = float(mu @ u - k * np.sqrt(max(var, 0.0)) - be)
        feasible = margin >= -1e-7 * max(abs(be), 1e-300)
        return ScheduleResult(
            times=self.times.copy(),
            deployment=u.reshape(self.n_iv, self.n_t)
            if feasible
            else np.full((self.n_iv, self.n_t), np.nan),
            cost=self.cost(u, 0.0) if feasible else float("inf"),
            discounted_cost=self.cost(u, discount_rate)
            if feasible
            else float("inf"),
            confidence=float(confidence),
            expected_cooling=float(mu @ u),
            cooling_std=float(np.sqrt(max(var, 0.0))),
            required_cooling=be,
            margin=margin,
            feasible=bool(feasible),
            success=bool(res.success and feasible),
            message=str(res.message) if feasible else "schedule infeasible",
            diagnostics={"discount_rate": discount_rate},
        )
