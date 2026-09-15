"""Portfolio construction, forcing neutrality and constrained optimisation.

A portfolio assigns a deployment scale :math:`\\alpha_i \\ge 0` to each
available intervention. Writing :math:`r_i(t) = \\mathrm{AGTP}_{F_i}(t)` for the
temperature response of one unit of intervention :math:`i`, the net warming of
an emission pathway together with its portfolio is

.. math::
    \\Delta T_{\\mathrm{net}}(t;\\alpha)
        = \\Delta T_E(t) + \\sum_{i=1}^N \\alpha_i r_i(t) .

Two functionals of that trajectory drive every optimisation problem here: its
integral over the horizon, which gives the linear forcing neutrality
constraint, and its squared integral, which gives a convex quadratic measure
of residual warming. Both are evaluated exactly, since every response is a
poly exponential signal.

The Hilbert space of the research plan enters through the inner product

.. math:: \\langle f, g \\rangle_w = \\int_0^\\infty f(t) g(t) w(t)\\, dt ,

with a policy weighting :math:`w`. The Gram matrix of the intervention
responses in that inner product is available in closed form, and the quadratic
objective is exactly the squared norm of the net trajectory when
:math:`w = \\mathbf{1}_{[0,TH]}`.

Correction to the research plan
-------------------------------
Theorem 3 of the plan states that the *average* cost per unit cumulative
cooling, :math:`C_i / b_i`, is equalised across deployed interventions. With
costs linear in the deployment scale the cost minimising problem is a linear
programme whose solution is a vertex, so a single intervention is deployed and
no equalisation occurs unless a capacity limit binds. The condition that holds
in general is equalisation of the *marginal* cost per unit cooling,
:math:`C_i'(\\alpha_i) / b_i = \\lambda`, which reduces to the plan's statement
only for linear costs. See :meth:`Portfolio.kkt_report`.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Callable, Dict, List, Optional, Sequence, Tuple

import numpy as np
from scipy.optimize import differential_evolution, linprog, minimize

from .climate import ClimateModel
from .expsum import ExpSum, Signal
from .pathways import EmissionPathway
from .profiles import Intervention, LinearCost, PowerCost

__all__ = [
    "PortfolioResult",
    "Portfolio",
    "weighting_kernel",
]


def weighting_kernel(
    kind: str = "horizon", horizon: float = 100.0, rate: float = 0.02
) -> Tuple[Optional[ExpSum], float]:
    """Return the policy weighting ``w`` and the effective upper limit.

    ``"horizon"`` gives the indicator of ``[0, TH]``, the choice implicit in
    every cumulative metric. ``"discount"`` gives ``exp(-rate * t)`` over an
    unbounded horizon, which values near term outcomes more highly.
    """
    if kind == "horizon":
        return None, float(horizon)
    if kind == "discount":
        return ExpSum.exponential(1.0, rate), float(np.inf)
    raise ValueError("unknown weighting kind: {}".format(kind))


@dataclass
class PortfolioResult:
    """The outcome of one optimisation."""

    alpha: np.ndarray
    objective: float
    cost: float
    deviation: float
    peak: float
    neutrality_residual: float
    success: bool
    message: str = ""
    multiplier: Optional[float] = None
    diagnostics: Dict[str, object] = field(default_factory=dict)

    def shares(self) -> np.ndarray:
        total = float(np.sum(self.alpha))
        return self.alpha / total if total > 0 else self.alpha * 0.0

    def as_dict(self, names: Sequence[str]) -> Dict[str, object]:
        out = {
            "objective": self.objective,
            "cost": self.cost,
            "deviation": self.deviation,
            "peak": self.peak,
            "neutrality_residual": self.neutrality_residual,
            "success": self.success,
        }
        for n, a in zip(names, self.alpha):
            out["alpha_" + n] = float(a)
        return out


class Portfolio:
    """A set of interventions evaluated against one emission pathway.

    Parameters
    ----------
    climate
        The impulse response climate model.
    interventions
        The available measures. Deployment scales are in the natural unit of
        each measure, normally kg of carbon dioxide removed at peak storage,
        or kg of the target gas for a direct species removal.
    pathway
        The emission pathway to be offset. May be ``None`` for studies of the
        interventions alone.
    horizon
        The policy time horizon ``TH`` in years.
    """

    def __init__(
        self,
        climate: ClimateModel,
        interventions: Sequence[Intervention],
        pathway: Optional[EmissionPathway] = None,
        horizon: float = 100.0,
    ) -> None:
        self.climate = climate
        self.interventions = list(interventions)
        self.pathway = pathway
        self.horizon = float(horizon)
        self._responses: Optional[List[Signal]] = None
        self._gram: Optional[np.ndarray] = None
        self._q: Optional[np.ndarray] = None
        self._c0: Optional[float] = None

    # ------------------------------------------------------------- basic data
    @property
    def n(self) -> int:
        return len(self.interventions)

    @property
    def names(self) -> List[str]:
        return [iv.name for iv in self.interventions]

    @property
    def labels(self) -> List[str]:
        return [iv.display for iv in self.interventions]

    def responses(self) -> List[Signal]:
        """The unit temperature response of each intervention."""
        if self._responses is None:
            self._responses = [
                iv.response_signal(self.climate) for iv in self.interventions
            ]
        return self._responses

    def cooling_vector(self) -> np.ndarray:
        """``b_i(TH)``, cumulative cooling per unit deployment, K yr."""
        return np.array(
            [-float(r.definite(self.horizon)) for r in self.responses()]
        )

    def cumulative_warming(self) -> float:
        """``B_E(TH)``, cumulative warming of the pathway, K yr."""
        if self.pathway is None:
            return 0.0
        return float(self.pathway.cumulative_warming(self.climate, self.horizon))

    def bounds(self) -> List[Tuple[float, Optional[float]]]:
        return [(0.0, iv.max_scale) for iv in self.interventions]

    # -------------------------------------------------- internal scaling
    # Cumulative cooling per unit deployment is of order 1e-14 K yr per kg and
    # squared temperature residuals of order 1e-26 K^2 yr. Solvers apply
    # absolute tolerances to constraint violations and to objective decrements,
    # so a problem posed in these units is solved trivially and wrongly: a
    # neutrality constraint of magnitude 1e-12 sits far inside the default
    # feasibility tolerance, and any quadratic objective looks converged at the
    # starting point. Every optimisation below is therefore posed in
    # dimensionless variables x = alpha / alpha_ref, restoring alpha only on
    # return. The scales are properties of the problem, so this changes no
    # solution, only the conditioning.
    def scales(self) -> Dict[str, float]:
        """Reference magnitudes used to non-dimensionalise the optimisations."""
        b = self.cooling_vector()
        b_ref = float(np.max(np.abs(b))) if b.size and np.any(b) else 1.0
        be = abs(self.cumulative_warming())
        if be > 0 and b_ref > 0:
            a_ref = be / b_ref
        else:
            caps = [
                iv.max_scale for iv in self.interventions if iv.max_scale is not None
            ]
            a_ref = float(np.max(caps)) if caps else 1.0
        a_ref = a_ref if a_ref > 0 else 1.0
        cost_ref = self.cost(np.full(self.n, a_ref))
        if not np.isfinite(cost_ref) or cost_ref <= 0:
            cost_ref = 1.0
        dev_ref = self.baseline_deviation()
        if not np.isfinite(dev_ref) or dev_ref <= 0:
            g = self.gram_matrix()
            dev_ref = float(np.max(np.abs(g))) * a_ref * a_ref
        if not np.isfinite(dev_ref) or dev_ref <= 0:
            dev_ref = 1.0
        return {
            "alpha": a_ref,
            "cooling": b_ref,
            "cost": float(cost_ref),
            "deviation": float(dev_ref),
        }

    def _scaled_bounds(self, a_ref: float) -> List[Tuple[float, Optional[float]]]:
        return [
            (0.0, None if iv.max_scale is None else iv.max_scale / a_ref)
            for iv in self.interventions
        ]

    # ------------------------------------------------- Hilbert space geometry
    def gram_matrix(self) -> np.ndarray:
        """``G_ij = <r_i, r_j>`` over ``[0, TH]``, the Gram matrix of responses.

        Symmetric and positive semi definite by construction. Its condition
        number measures how nearly collinear the available responses are, and
        therefore how strongly the cost minimising portfolio is determined.
        """
        if self._gram is None:
            r = self.responses()
            n = len(r)
            g = np.zeros((n, n))
            for i in range(n):
                for j in range(i, n):
                    v = r[i].continuous_product_integral(r[j], self.horizon)
                    g[i, j] = g[j, i] = v
            self._gram = g
        return self._gram

    def overlap_vector(self) -> np.ndarray:
        """``q_i = <Delta T_E, r_i>`` over ``[0, TH]``."""
        if self._q is None:
            if self.pathway is None:
                self._q = np.zeros(self.n)
            else:
                dte = self.pathway.response(self.climate)
                self._q = np.array(
                    [
                        dte.continuous_product_integral(r, self.horizon)
                        for r in self.responses()
                    ]
                )
        return self._q

    def baseline_deviation(self) -> float:
        """``<Delta T_E, Delta T_E>``, the residual with no intervention."""
        if self._c0 is None:
            if self.pathway is None:
                self._c0 = 0.0
            else:
                dte = self.pathway.response(self.climate)
                self._c0 = dte.continuous_product_integral(dte, self.horizon)
        return self._c0

    def condition_number(self) -> float:
        g = self.gram_matrix()
        d = np.sqrt(np.diag(g))
        scale = np.outer(d, d)
        corr = np.where(scale > 0, g / np.where(scale > 0, scale, 1.0), 0.0)
        return float(np.linalg.cond(corr))

    # ------------------------------------------------------------- evaluation
    def net_temperature(self, alpha: Sequence[float], t) -> np.ndarray:
        alpha = np.asarray(alpha, dtype=float)
        out = np.zeros(np.asarray(t, dtype=float).shape)
        if self.pathway is not None:
            out = out + self.pathway.temperature(self.climate, t)
        for a, r in zip(alpha, self.responses()):
            if a != 0.0:
                out = out + a * r.eval(t)
        return out

    def neutrality_residual(self, alpha: Sequence[float]) -> float:
        """``B_E(TH) - sum_i alpha_i b_i(TH)``; zero at forcing neutrality."""
        alpha = np.asarray(alpha, dtype=float)
        return float(self.cumulative_warming() - alpha @ self.cooling_vector())

    def deviation(self, alpha: Sequence[float]) -> float:
        """``int_0^TH [Delta T_net]^2 dt``, in K^2 yr."""
        alpha = np.asarray(alpha, dtype=float)
        return float(
            self.baseline_deviation()
            + 2.0 * self.overlap_vector() @ alpha
            + alpha @ self.gram_matrix() @ alpha
        )

    def peak_deviation(self, alpha: Sequence[float], n: int = 2001) -> float:
        t = np.linspace(0.0, self.horizon, n)
        return float(np.max(np.abs(self.net_temperature(alpha, t))))

    def cost(self, alpha: Sequence[float]) -> float:
        return float(
            sum(
                iv.cost(a)
                for iv, a in zip(self.interventions, np.asarray(alpha, dtype=float))
            )
        )

    def cost_gradient(self, alpha: Sequence[float]) -> np.ndarray:
        return np.array(
            [
                iv.marginal_cost(a)
                for iv, a in zip(self.interventions, np.asarray(alpha, dtype=float))
            ]
        )

    def side_effect_matrix(self, keys: Sequence[str]) -> np.ndarray:
        return np.array(
            [[iv.side_effects.get(k, 0.0) for iv in self.interventions] for k in keys]
        )

    def attainability(self) -> Dict[str, object]:
        """Whether forcing neutrality is reachable within the capacity limits.

        The maximum cumulative cooling available is obtained by deploying
        every measure to its ceiling. When that falls short of the warming to
        be offset, no portfolio is neutral and the optimisation is infeasible;
        reporting the shortfall is more informative for policy than reporting
        a solver failure, since it states how much additional capacity would
        be required and of what kind.
        """
        b = self.cooling_vector()
        caps = np.array(
            [
                np.inf if iv.max_scale is None else iv.max_scale
                for iv in self.interventions
            ]
        )
        per_measure = b * caps
        total = float(np.sum(per_measure))
        required = self.cumulative_warming()
        return {
            "required_cooling": required,
            "max_available_cooling": total,
            "feasible": bool(total >= required) if np.isfinite(total) else True,
            "coverage": total / required if required else np.inf,
            "shortfall": max(required - total, 0.0),
            "per_measure_max_cooling": per_measure,
            "unbounded": bool(np.any(~np.isfinite(caps))),
        }

    def _result(
        self,
        alpha: np.ndarray,
        objective: float,
        success: bool,
        message: str = "",
        multiplier: Optional[float] = None,
        **diagnostics,
    ) -> PortfolioResult:
        return PortfolioResult(
            alpha=alpha,
            objective=float(objective),
            cost=self.cost(alpha),
            deviation=self.deviation(alpha),
            peak=self.peak_deviation(alpha),
            neutrality_residual=self.neutrality_residual(alpha),
            success=bool(success),
            message=message,
            multiplier=multiplier,
            diagnostics=dict(diagnostics),
        )

    # ------------------------------------------------------- cost minimisation
    def minimise_cost(
        self,
        neutral: bool = True,
        budget: Optional[float] = None,
        side_effect_caps: Optional[Dict[str, float]] = None,
        tolerance: float = 0.0,
    ) -> PortfolioResult:
        """Least cost portfolio achieving forcing neutrality.

        Solved as a linear programme when every cost is linear, and by
        sequential quadratic programming with analytic derivatives when any
        cost is strictly convex. ``tolerance`` relaxes neutrality to
        ``|residual| <= tolerance * B_E``.
        """
        b = self.cooling_vector()
        be = self.cumulative_warming()
        caps = side_effect_caps or {}
        linear = all(isinstance(iv.cost_model, LinearCost) for iv in self.interventions)
        s = self.scales()
        a_ref, cost_ref = s["alpha"], s["cost"]
        # Dimensionless neutrality row: (b * a_ref / |B_E|) . x = sign(B_E)
        row_scale = float(max(abs(be), s["cooling"] * a_ref))
        b_s = b * a_ref / row_scale
        be_s = be / row_scale

        if linear and budget is None and tolerance == 0.0:
            unit = np.array([iv.cost_model.unit_cost for iv in self.interventions])
            c_s = unit * a_ref / cost_ref
            a_eq = b_s.reshape(1, -1) if neutral else None
            b_eq = np.array([be_s]) if neutral else None
            a_ub, b_ub = [], []
            if caps:
                keys = list(caps)
                mat = self.side_effect_matrix(keys)
                lim = np.array([caps[k] for k in keys])
                norm = np.maximum(np.abs(lim), 1e-300).reshape(-1, 1)
                a_ub.append(mat * a_ref / norm)
                b_ub.append(lim / norm.ravel())
            res = linprog(
                c_s,
                A_ub=np.vstack(a_ub) if a_ub else None,
                b_ub=np.concatenate(b_ub) if b_ub else None,
                A_eq=a_eq,
                b_eq=b_eq,
                bounds=self._scaled_bounds(a_ref),
                method="highs",
            )
            alpha = (res.x * a_ref) if res.x is not None else np.zeros(self.n)
            lam = None
            if res.success and neutral:
                eq = getattr(res, "eqlin", None)
                if eq is not None and len(np.atleast_1d(eq.marginals)):
                    # Undo the scaling of both the objective and the row.
                    lam = float(
                        np.atleast_1d(eq.marginals)[0] * cost_ref / row_scale
                    )
            return self._result(
                np.asarray(alpha),
                self.cost(alpha) if res.success else np.inf,
                res.success,
                res.message,
                lam,
                solver="linprog",
            )

        def cost_s(x: np.ndarray) -> float:
            return self.cost(x * a_ref) / cost_ref

        def cost_jac_s(x: np.ndarray) -> np.ndarray:
            return self.cost_gradient(x * a_ref) * a_ref / cost_ref

        cons = []
        if neutral:
            if tolerance == 0.0:
                cons.append(
                    {
                        "type": "eq",
                        "fun": lambda x: np.array([x @ b_s - be_s]),
                        "jac": lambda x: b_s.reshape(1, -1),
                    }
                )
            else:
                tol_s = abs(tolerance * be_s)
                cons.append(
                    {
                        "type": "ineq",
                        "fun": lambda x: np.array([tol_s - abs(x @ b_s - be_s)]),
                        "jac": lambda x: -np.sign(x @ b_s - be_s) * b_s.reshape(1, -1),
                    }
                )
        if budget is not None:
            cons.append(
                {
                    "type": "ineq",
                    "fun": lambda x: np.array([(budget - self.cost(x * a_ref)) / cost_ref]),
                    "jac": lambda x: -cost_jac_s(x).reshape(1, -1),
                }
            )
        for k, cap in caps.items():
            row = self.side_effect_matrix([k])[0]
            nrm = max(abs(cap), 1e-300)
            cons.append(
                {
                    "type": "ineq",
                    "fun": lambda x, row=row, cap=cap, nrm=nrm: np.array(
                        [(cap - row @ (x * a_ref)) / nrm]
                    ),
                    "jac": lambda x, row=row, nrm=nrm: -(row * a_ref / nrm).reshape(1, -1),
                }
            )
        x0 = self._feasible_start(b, be) / a_ref
        res = minimize(
            cost_s,
            x0,
            jac=cost_jac_s,
            bounds=self._scaled_bounds(a_ref),
            constraints=cons,
            method="SLSQP",
            options={"maxiter": 1000, "ftol": 1e-14},
        )
        alpha = np.maximum(res.x, 0.0) * a_ref
        return self._result(
            alpha, self.cost(alpha), res.success, res.message, solver="SLSQP"
        )

    def _feasible_start(self, b: np.ndarray, be: float) -> np.ndarray:
        """A starting point that satisfies neutrality if it is attainable."""
        x0 = np.zeros(self.n)
        if be <= 0 or not np.any(b > 0):
            return x0
        share = be / max(int(np.sum(b > 0)), 1)
        for i in range(self.n):
            if b[i] > 0:
                cap = self.interventions[i].max_scale
                val = share / b[i]
                x0[i] = val if cap is None else min(val, cap)
        return x0

    def analytic_power_cost_solution(self) -> Optional[PortfolioResult]:
        """Closed form least cost portfolio for identical convex power costs.

        With :math:`C_i(\\alpha_i) = u_i (\\alpha_i/\\alpha_{\\mathrm{ref}})^\\gamma
        \\alpha_{\\mathrm{ref}}/\\gamma` and :math:`\\gamma > 1`, stationarity of
        the Lagrangian gives

        .. math::
            \\alpha_i(\\lambda) = \\alpha_{\\mathrm{ref}}
                \\Big(\\frac{\\lambda b_i}{u_i}\\Big)^{1/(\\gamma-1)} ,

        clipped to any capacity limit, and :math:`\\lambda` is fixed by the
        neutrality constraint. The map is monotone in :math:`\\lambda`, so a
        bisection solves it. Returns ``None`` when the cost models are not of
        this common form; used to verify the numerical optimiser.
        """
        models = [iv.cost_model for iv in self.interventions]
        if not all(isinstance(m, PowerCost) for m in models):
            return None
        gammas = {m.gamma for m in models}
        if len(gammas) != 1:
            return None
        gamma = gammas.pop()
        if gamma <= 1.0:
            return None
        b = self.cooling_vector()
        be = self.cumulative_warming()
        u = np.array([m.unit_cost for m in models])
        ref = np.array([m.reference_scale for m in models])
        caps = np.array(
            [
                np.inf if iv.max_scale is None else iv.max_scale
                for iv in self.interventions
            ]
        )
        expo = 1.0 / (gamma - 1.0)

        def alpha_of(lam: float) -> np.ndarray:
            with np.errstate(invalid="ignore"):
                a = ref * np.power(np.maximum(lam * b / u, 0.0), expo)
            return np.minimum(a, caps)

        lo, hi = 0.0, 1.0
        for _ in range(400):
            if alpha_of(hi) @ b >= be:
                break
            hi *= 2.0
        else:
            return None
        for _ in range(300):
            mid = 0.5 * (lo + hi)
            if alpha_of(mid) @ b < be:
                lo = mid
            else:
                hi = mid
        lam = 0.5 * (lo + hi)
        alpha = alpha_of(lam)
        return self._result(
            alpha, self.cost(alpha), True, "analytic", lam, solver="analytic"
        )

    # -------------------------------------------------- deviation minimisation
    def minimise_deviation(
        self,
        neutral: bool = False,
        budget: Optional[float] = None,
    ) -> PortfolioResult:
        """Portfolio minimising the squared residual warming over the horizon.

        A convex quadratic programme in :math:`\\alpha`, since the Gram matrix
        of responses is positive semi definite.
        """
        g = self.gram_matrix()
        q = self.overlap_vector()
        b = self.cooling_vector()
        be = self.cumulative_warming()
        s = self.scales()
        a_ref, dev_ref, cost_ref = s["alpha"], s["deviation"], s["cost"]
        # Dimensionless quadratic: x -> (d0 + 2 q.x a_ref + x.G.x a_ref^2)/dev_ref
        g_s = g * a_ref * a_ref / dev_ref
        q_s = q * a_ref / dev_ref
        d0_s = self.baseline_deviation() / dev_ref
        row_scale = float(max(abs(be), s["cooling"] * a_ref))
        b_s = b * a_ref / row_scale
        be_s = be / row_scale

        def obj(x: np.ndarray) -> float:
            return float(d0_s + 2.0 * q_s @ x + x @ g_s @ x)

        def jac(x: np.ndarray) -> np.ndarray:
            return 2.0 * (g_s @ x + q_s)

        cons = []
        if neutral:
            cons.append(
                {
                    "type": "eq",
                    "fun": lambda x: np.array([x @ b_s - be_s]),
                    "jac": lambda x: b_s.reshape(1, -1),
                }
            )
        if budget is not None:
            cons.append(
                {
                    "type": "ineq",
                    "fun": lambda x: np.array(
                        [(budget - self.cost(x * a_ref)) / cost_ref]
                    ),
                    "jac": lambda x: -(
                        self.cost_gradient(x * a_ref) * a_ref / cost_ref
                    ).reshape(1, -1),
                }
            )
        x0 = (self._feasible_start(b, be) / a_ref) if neutral else np.zeros(self.n)
        res = minimize(
            obj,
            x0,
            jac=jac,
            bounds=self._scaled_bounds(a_ref),
            constraints=cons,
            method="SLSQP",
            options={"maxiter": 1000, "ftol": 1e-16},
        )
        alpha = np.maximum(res.x, 0.0) * a_ref
        return self._result(
            alpha, self.deviation(alpha), res.success, res.message, solver="SLSQP"
        )

    def minimise_peak(
        self, neutral: bool = False, n_grid: int = 401
    ) -> PortfolioResult:
        """Portfolio minimising the largest absolute residual warming.

        Formulated as a linear programme through the standard epigraph
        variable, with the trajectory sampled on a uniform grid.
        """
        t = np.linspace(0.0, self.horizon, n_grid)
        base = (
            self.pathway.temperature(self.climate, t)
            if self.pathway is not None
            else np.zeros_like(t)
        )
        r = np.array([sig.eval(t) for sig in self.responses()])
        b = self.cooling_vector()
        be = self.cumulative_warming()

        # Non-dimensionalise: x = alpha / a_ref and temperatures in units of
        # the largest baseline excursion, so the epigraph variable is O(1).
        s = self.scales()
        a_ref = s["alpha"]
        t_ref = float(np.max(np.abs(base))) if np.any(base) else 1.0
        if not np.isfinite(t_ref) or t_ref <= 0:
            t_ref = float(np.max(np.abs(r))) * a_ref or 1.0
        base_s = base / t_ref
        r_s = r * a_ref / t_ref
        row_scale = float(max(abs(be), s["cooling"] * a_ref))
        b_s = b * a_ref / row_scale

        # Variables: [x (n), z]
        c = np.zeros(self.n + 1)
        c[-1] = 1.0
        a_ub = np.vstack(
            [
                np.hstack([r_s.T, -np.ones((n_grid, 1))]),
                np.hstack([-r_s.T, -np.ones((n_grid, 1))]),
            ]
        )
        b_ub = np.concatenate([-base_s, base_s])
        a_eq = np.hstack([b_s, [0.0]]).reshape(1, -1) if neutral else None
        b_eq = np.array([be / row_scale]) if neutral else None
        bnds = self._scaled_bounds(a_ref) + [(0.0, None)]
        res = linprog(
            c, A_ub=a_ub, b_ub=b_ub, A_eq=a_eq, b_eq=b_eq, bounds=bnds, method="highs"
        )
        alpha = (res.x[: self.n] * a_ref) if res.x is not None else np.zeros(self.n)
        alpha = np.maximum(np.asarray(alpha), 0.0)
        return self._result(
            alpha,
            self.peak_deviation(alpha) if res.success else np.inf,
            res.success,
            res.message,
            solver="linprog-minimax",
        )

    # ----------------------------------------------------------- Pareto fronts
    def pareto_frontier(
        self,
        n_points: int = 25,
        neutral: bool = False,
        cost_range: Optional[Tuple[float, float]] = None,
    ) -> Dict[str, np.ndarray]:
        """Trade off between deployment cost and residual warming.

        Generated by the epsilon constraint method: residual warming is
        minimised subject to a cost ceiling, and the ceiling is swept.
        """
        if cost_range is None:
            free = self.minimise_deviation(neutral=neutral)
            hi = max(free.cost, 1.0)
            caps = np.linspace(0.0, hi, n_points)
        else:
            caps = np.linspace(cost_range[0], cost_range[1], n_points)
        rows = []
        for cap in caps:
            res = self.minimise_deviation(neutral=neutral, budget=float(cap))
            rows.append(
                (
                    cap,
                    res.cost,
                    res.deviation,
                    res.peak,
                    res.neutrality_residual,
                    res.alpha,
                )
            )
        return {
            "cost_cap": np.array([r[0] for r in rows]),
            "cost": np.array([r[1] for r in rows]),
            "deviation": np.array([r[2] for r in rows]),
            "peak": np.array([r[3] for r in rows]),
            "residual": np.array([r[4] for r in rows]),
            "alpha": np.array([r[5] for r in rows]),
        }

    # -------------------------------------------------- parameter optimisation
    def optimise_parameters(
        self,
        rebuild: Callable[[np.ndarray], Sequence[Intervention]],
        param_bounds: Sequence[Tuple[float, float]],
        objective: str = "cost",
        seed: int = 0,
        maxiter: int = 60,
        popsize: int = 15,
    ) -> Dict[str, object]:
        """Joint optimisation over deployment scales and design parameters.

        The inner problem in :math:`\\alpha` is convex and is solved exactly;
        the outer problem in the design parameters :math:`\\theta` is not, and
        is handled by differential evolution. ``rebuild`` maps a parameter
        vector to a fresh list of interventions.
        """

        def outer(theta: np.ndarray) -> float:
            trial = Portfolio(
                self.climate, rebuild(theta), self.pathway, self.horizon
            )
            res = (
                trial.minimise_cost(neutral=True)
                if objective == "cost"
                else trial.minimise_deviation(neutral=False)
            )
            if not res.success:
                return 1e30
            return res.cost if objective == "cost" else res.deviation

        out = differential_evolution(
            outer,
            list(param_bounds),
            seed=seed,
            maxiter=maxiter,
            popsize=popsize,
            polish=True,
            tol=1e-10,
        )
        best = Portfolio(self.climate, rebuild(out.x), self.pathway, self.horizon)
        inner = (
            best.minimise_cost(neutral=True)
            if objective == "cost"
            else best.minimise_deviation(neutral=False)
        )
        return {
            "theta": out.x,
            "value": float(out.fun),
            "portfolio": best,
            "result": inner,
            "success": bool(out.success),
        }

    # ------------------------------------------------------ optimality reports
    def kkt_report(self, alpha: Sequence[float], tol: float = 1e-6) -> Dict[str, object]:
        """Verify the first order conditions at a candidate portfolio.

        For the cost minimising problem with a neutrality constraint the
        stationarity condition on the interior is

        .. math:: \\frac{C_i'(\\alpha_i)}{b_i(TH)} = \\lambda ,

        the equalised marginal cost per unit cumulative cooling. Interventions
        held at a bound instead satisfy the corresponding inequality: a measure
        left undeployed must be no cheaper at the margin than the active ones,
        and a measure at its capacity limit must be no more expensive.
        """
        alpha = np.asarray(alpha, dtype=float)
        b = self.cooling_vector()
        marg = self.cost_gradient(alpha)
        ratio = np.where(b > 0, marg / np.where(b > 0, b, 1.0), np.inf)
        caps = np.array(
            [
                np.inf if iv.max_scale is None else iv.max_scale
                for iv in self.interventions
            ]
        )
        interior = (alpha > tol) & (alpha < caps * (1.0 - 1e-9))
        at_zero = alpha <= tol
        at_cap = alpha >= caps * (1.0 - 1e-9)
        lam = float(np.mean(ratio[interior])) if np.any(interior) else float("nan")
        if not np.any(interior):
            active = alpha > tol
            lam = float(np.min(ratio[active])) if np.any(active) else float("nan")
        spread = (
            float(np.max(ratio[interior]) - np.min(ratio[interior]))
            if np.count_nonzero(interior) > 1
            else 0.0
        )
        return {
            "multiplier": lam,
            "ratio": ratio,
            "interior": interior,
            "at_zero": at_zero,
            "at_capacity": at_cap,
            "equalisation_spread": spread,
            "relative_spread": spread / lam if lam and np.isfinite(lam) else 0.0,
            "zero_condition_satisfied": bool(
                np.all(ratio[at_zero] >= lam - 1e-6 * max(abs(lam), 1.0))
                if np.any(at_zero) and np.isfinite(lam)
                else True
            ),
            "neutrality_residual": self.neutrality_residual(alpha),
        }

    def deviation_optimality_report(self, alpha: Sequence[float]) -> Dict[str, object]:
        """Check the orthogonality condition for deviation minimisation.

        Theorem 4 of the research plan requires that each deployed response be
        orthogonal to the net trajectory,
        :math:`\\int_0^{TH} r_i \\Delta T_{\\mathrm{net}}\\,dt = 0`. That is the
        unconstrained stationarity condition; with a neutrality constraint the
        overlaps are instead proportional to :math:`b_i`.
        """
        alpha = np.asarray(alpha, dtype=float)
        g = self.gram_matrix()
        q = self.overlap_vector()
        overlaps = g @ alpha + q
        b = self.cooling_vector()
        scale = np.sqrt(np.abs(np.diag(g)) * max(self.deviation(alpha), 1e-300))
        return {
            "overlap": overlaps,
            "normalised_overlap": np.where(scale > 0, overlaps / scale, 0.0),
            "proportionality": np.where(b > 0, overlaps / b, np.nan),
        }

    # ----------------------------------------------------------------- helpers
    def compensation_ratio(self, species: str) -> np.ndarray:
        """Deployment of each intervention needed per kg of one species.

        The single intervention, single species reduction of the neutrality
        constraint: :math:`\\alpha = \\mathrm{iAGTP}_s(TH) / b_i(TH)`.
        """
        num = float(self.climate.iagtp(species, self.horizon))
        return num / self.cooling_vector()

    def summary_frame(self, alpha: Sequence[float]):
        import pandas as pd

        alpha = np.asarray(alpha, dtype=float)
        b = self.cooling_vector()
        return pd.DataFrame(
            {
                "intervention": self.labels,
                "alpha": alpha,
                "share": alpha / np.sum(alpha) if np.sum(alpha) > 0 else alpha,
                "cooling_per_unit": b,
                "cooling_delivered": alpha * b,
                "cost": [iv.cost(a) for iv, a in zip(self.interventions, alpha)],
                "marginal_cost": self.cost_gradient(alpha),
                "marginal_cost_per_cooling": np.where(
                    b > 0, self.cost_gradient(alpha) / np.where(b > 0, b, 1.0), np.inf
                ),
                "mean_storage_time": [
                    iv.mean_storage_time for iv in self.interventions
                ],
            }
        )
