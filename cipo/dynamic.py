"""Dynamic deployment scheduling.

The static problem chooses how much of each measure to deploy. The dynamic
problem chooses *when*. Let :math:`u_{i,n} \\ge 0` be the deployment of
intervention :math:`i` started in year :math:`t_n`. Because the climate
response is linear, a deployment started at :math:`t_n` contributes a shifted
copy of the same unit response, so the net warming remains linear in the
decision variables,

.. math::
    \\Delta T_{\\mathrm{net}}(t)
      = \\Delta T_E(t) + \\sum_{i}\\sum_{n} u_{i,n}\\, r_i(t - t_n) .

The residual warming is therefore a convex quadratic form in :math:`u`, and
the schedule that minimises it subject to deployment rate limits, a budget
path and forcing neutrality is the solution of a convex quadratic programme.
There is no dynamic programming recursion and no local optimum to escape.

This resolves an ambiguity in Eq. (35) of the research plan, which writes the
dynamic objective with :math:`\\Delta T_{\\mathrm{net}}(t;\\alpha(t))` as though
the warming at time :math:`t` depended only on the deployment at that instant.
The climate response has memory, so the warming at :math:`t` depends on the
whole deployment history up to :math:`t`; the formulation above makes that
dependence explicit and stays convex.

A practical consequence follows directly. Deploying a temporary measure early
against a declining emission pathway wastes cooling, because the pool releases
its carbon while emissions are still falling. The optimal schedule delays
temporary deployment and concentrates it where the pathway's warming is
steepest, which the solutions here quantify.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Dict, List, Optional, Sequence, Tuple

import numpy as np
from scipy.optimize import minimize

from .climate import ClimateModel
from .expsum import Signal
from .pathways import EmissionPathway
from .portfolio import Portfolio
from .profiles import Intervention

__all__ = ["DynamicSchedule", "DynamicPortfolio"]


@dataclass
class DynamicSchedule:
    """A deployment schedule and its diagnostics."""

    times: np.ndarray
    deployment: np.ndarray  # shape (n_interventions, n_times)
    cost: float
    deviation: float
    peak: float
    neutrality_residual: float
    success: bool
    message: str = ""
    diagnostics: Dict[str, object] = field(default_factory=dict)

    def totals(self) -> np.ndarray:
        """Total deployment of each intervention over the schedule."""
        return self.deployment.sum(axis=1)

    def frame(self, names: Sequence[str]):
        import pandas as pd

        return pd.DataFrame(
            self.deployment.T, index=self.times, columns=list(names)
        ).rename_axis("year")

    def centre_of_mass(self) -> np.ndarray:
        """Deployment weighted mean start time for each intervention, in years."""
        out = np.full(self.deployment.shape[0], np.nan)
        for i, row in enumerate(self.deployment):
            total = row.sum()
            if total > 0:
                out[i] = float(self.times @ row / total)
        return out


class DynamicPortfolio:
    """Optimal timing of deployment over a decision horizon."""

    def __init__(
        self,
        climate: ClimateModel,
        interventions: Sequence[Intervention],
        pathway: Optional[EmissionPathway],
        horizon: float = 100.0,
        decision_times: Optional[Sequence[float]] = None,
        n_quad: int = 1201,
    ) -> None:
        self.climate = climate
        self.interventions = list(interventions)
        self.pathway = pathway
        self.horizon = float(horizon)
        self.times = np.asarray(
            decision_times
            if decision_times is not None
            else np.arange(0.0, min(60.0, horizon), 10.0),
            dtype=float,
        )
        self.n_quad = int(n_quad)
        self._static = Portfolio(climate, self.interventions, pathway, horizon)

    # --------------------------------------------------------------- geometry
    @property
    def n_iv(self) -> int:
        return len(self.interventions)

    @property
    def n_t(self) -> int:
        return self.times.size

    @property
    def n_var(self) -> int:
        return self.n_iv * self.n_t

    def _basis(self) -> List[Signal]:
        """Unit response of each (intervention, start time) pair."""
        out = []
        for iv in self.interventions:
            sig = iv.response_signal(self.climate)
            for t0 in self.times:
                out.append(sig.shifted(float(t0)))
        return out

    def _quadratic_form(self) -> Tuple[np.ndarray, np.ndarray, float, np.ndarray]:
        """Gram matrix, overlap vector, baseline, and cooling coefficients."""
        basis = self._basis()
        m = len(basis)
        g = np.zeros((m, m))
        for i in range(m):
            for j in range(i, m):
                v = basis[i].continuous_product_integral(basis[j], self.horizon)
                g[i, j] = g[j, i] = v
        if self.pathway is not None:
            dte = self.pathway.response(self.climate)
            q = np.array(
                [dte.continuous_product_integral(s, self.horizon) for s in basis]
            )
            d0 = dte.continuous_product_integral(dte, self.horizon)
        else:
            q = np.zeros(m)
            d0 = 0.0
        cooling = np.array([-float(s.definite(self.horizon)) for s in basis])
        return g, q, float(d0), cooling

    # ------------------------------------------------------------- evaluation
    def net_temperature(self, u: np.ndarray, t) -> np.ndarray:
        u = np.asarray(u, dtype=float).reshape(self.n_iv, self.n_t)
        out = np.zeros(np.asarray(t, dtype=float).shape)
        if self.pathway is not None:
            out = out + self.pathway.temperature(self.climate, t)
        for i, iv in enumerate(self.interventions):
            sig = iv.response_signal(self.climate)
            for k, t0 in enumerate(self.times):
                if u[i, k] != 0.0:
                    out = out + u[i, k] * sig.shifted(float(t0)).eval(t)
        return out

    def cost(self, u: np.ndarray, discount_rate: float = 0.0) -> float:
        """Total deployment cost, optionally discounted to the present."""
        u = np.asarray(u, dtype=float).reshape(self.n_iv, self.n_t)
        total = 0.0
        for i, iv in enumerate(self.interventions):
            for k, t0 in enumerate(self.times):
                disc = float(np.exp(-discount_rate * t0))
                total += disc * iv.cost(u[i, k])
        return float(total)

    def cost_gradient(self, u: np.ndarray, discount_rate: float = 0.0) -> np.ndarray:
        u = np.asarray(u, dtype=float).reshape(self.n_iv, self.n_t)
        out = np.zeros_like(u)
        for i, iv in enumerate(self.interventions):
            for k, t0 in enumerate(self.times):
                out[i, k] = float(np.exp(-discount_rate * t0)) * iv.marginal_cost(
                    u[i, k]
                )
        return out.ravel()

    # ------------------------------------------------------------- optimisation
    def optimise(
        self,
        objective: str = "deviation",
        neutral: bool = False,
        budget_path: Optional[Sequence[float]] = None,
        max_rate: Optional[Sequence[float]] = None,
        discount_rate: float = 0.0,
        total_caps: bool = True,
    ) -> DynamicSchedule:
        """Solve for the optimal schedule.

        Parameters
        ----------
        objective
            ``"deviation"`` minimises the squared residual warming;
            ``"cost"`` minimises discounted outlay subject to neutrality.
        budget_path
            Per period spending ceiling, one entry per decision time.
        max_rate
            Per period deployment ceiling for each intervention.
        total_caps
            Apply each intervention's ``max_scale`` to its schedule total.
        """
        g, q, d0, cooling = self._quadratic_form()
        be = self._static.cumulative_warming()

        # Non-dimensionalise, as in the static problem.
        b_ref = float(np.max(np.abs(cooling))) if np.any(cooling) else 1.0
        u_ref = abs(be) / b_ref if be and b_ref else 1.0
        u_ref = u_ref if u_ref > 0 else 1.0
        dev_ref = d0 if d0 > 0 else float(np.max(np.abs(g))) * u_ref ** 2 or 1.0
        cost_ref = self.cost(np.full(self.n_var, u_ref), discount_rate) or 1.0
        row = float(max(abs(be), b_ref * u_ref))

        g_s = g * u_ref * u_ref / dev_ref
        q_s = q * u_ref / dev_ref
        d0_s = d0 / dev_ref
        cool_s = cooling * u_ref / row
        be_s = be / row

        def dev_s(x: np.ndarray) -> float:
            return float(d0_s + 2.0 * q_s @ x + x @ g_s @ x)

        def dev_jac_s(x: np.ndarray) -> np.ndarray:
            return 2.0 * (g_s @ x + q_s)

        def cost_s(x: np.ndarray) -> float:
            return self.cost(x * u_ref, discount_rate) / cost_ref

        def cost_jac_s(x: np.ndarray) -> np.ndarray:
            return self.cost_gradient(x * u_ref, discount_rate) * u_ref / cost_ref

        cons = []
        if neutral or objective == "cost":
            cons.append(
                {
                    "type": "eq",
                    "fun": lambda x: np.array([x @ cool_s - be_s]),
                    "jac": lambda x: cool_s.reshape(1, -1),
                }
            )
        if budget_path is not None:
            for k, cap in enumerate(budget_path):
                idx = [i * self.n_t + k for i in range(self.n_iv)]
                nrm = max(abs(cap), 1e-300)

                def period_cost(x, idx=idx, cap=cap, nrm=nrm):
                    u = x * u_ref
                    spend = sum(
                        self.interventions[j // self.n_t].cost(u[j]) for j in idx
                    )
                    return np.array([(cap - spend) / nrm])

                cons.append({"type": "ineq", "fun": period_cost})

        bounds = []
        for i, iv in enumerate(self.interventions):
            per_period = None if max_rate is None else float(max_rate[i]) / u_ref
            for _ in range(self.n_t):
                bounds.append((0.0, per_period))
        if total_caps:
            for i, iv in enumerate(self.interventions):
                if iv.max_scale is not None:
                    idx = [i * self.n_t + k for k in range(self.n_t)]
                    cap_s = iv.max_scale / u_ref

                    def total_cap(x, idx=idx, cap_s=cap_s):
                        return np.array([cap_s - float(np.sum(x[idx]))])

                    def total_cap_jac(x, idx=idx):
                        j = np.zeros(self.n_var)
                        j[idx] = -1.0
                        return j.reshape(1, -1)

                    cons.append(
                        {"type": "ineq", "fun": total_cap, "jac": total_cap_jac}
                    )

        obj, jac = (
            (dev_s, dev_jac_s) if objective == "deviation" else (cost_s, cost_jac_s)
        )
        x0 = np.full(self.n_var, (be_s / max(float(np.sum(cool_s)), 1e-12)) if be else 0.0)
        x0 = np.clip(x0, 0.0, None)
        res = minimize(
            obj,
            x0,
            jac=jac,
            bounds=bounds,
            constraints=cons,
            method="SLSQP",
            options={"maxiter": 2000, "ftol": 1e-14},
        )
        u = np.maximum(res.x, 0.0) * u_ref
        t_dense = np.linspace(0.0, self.horizon, self.n_quad)
        traj = self.net_temperature(u, t_dense)
        return DynamicSchedule(
            times=self.times.copy(),
            deployment=u.reshape(self.n_iv, self.n_t),
            cost=self.cost(u, discount_rate),
            deviation=float(d0 + 2.0 * q @ u + u @ g @ u),
            peak=float(np.max(np.abs(traj))),
            neutrality_residual=float(be - u @ cooling),
            success=bool(res.success),
            message=str(res.message),
            diagnostics={
                "trajectory_times": t_dense,
                "trajectory": traj,
                "static_warming": be,
            },
        )

    def static_equivalent(self) -> Portfolio:
        """The corresponding problem with all deployment at time zero."""
        return self._static
