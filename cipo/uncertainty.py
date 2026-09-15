"""Uncertainty propagation, robust design and global sensitivity analysis.

Three complementary treatments of parameter uncertainty are provided.

*Monte Carlo* draws climate and intervention parameters from assigned
distributions, re solves the portfolio problem for each draw, and reports the
resulting spread. It answers how uncertain a recommendation is.

*Distributionally robust design* asks a different question: what deployment
guarantees neutrality when the parameters are adversarial within an ambiguity
set. Only the mean and the covariance of the uncertain cooling coefficients
are assumed known. For the linear neutrality functional the worst case over
all distributions with a given mean and covariance is available in closed
form. Writing :math:`\\mu` and :math:`\\Sigma` for the mean and covariance of
the cooling vector :math:`b`, the requirement that neutrality hold with
probability at least :math:`1-\\eta` under every such distribution is exactly
the second order cone constraint

.. math::
    \\mu^{\\mathsf T}\\alpha
      - \\kappa(\\eta)\\,\\sqrt{\\alpha^{\\mathsf T}\\Sigma\\,\\alpha}
      \\;\\ge\\; B_E(TH) ,
    \\qquad \\kappa(\\eta) = \\sqrt{\\tfrac{\\eta}{1-\\eta}} ,

the multiplier being the tight one sided Chebyshev bound of Cantelli, which
guarantees :math:`\\mathbb{P}(X < \\mu - \\kappa\\sigma) \\le 1/(1+\\kappa^2)
= 1-\\eta`. This replaces the
intractable formulation of the research plan, Eq. (33), whose inner supremum
over distributions is not computable as written, with an equivalent convex
programme. Robustness costs a premium, quantified below as the extra outlay
relative to the deployment that is neutral only on average.

*Sobol indices* attribute the variance of any scalar output to individual
parameters, using the Saltelli estimator on a scrambled Sobol sequence. First
order indices measure the effect of a parameter alone and total order indices
include all its interactions, so the gap between them exposes interaction
structure that a one at a time sensitivity scan cannot see.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Callable, Dict, List, Optional, Sequence, Tuple

import numpy as np
from scipy.optimize import minimize
from scipy.stats import qmc

from .climate import ClimateModel
from .portfolio import Portfolio

__all__ = [
    "ParameterSpec",
    "max_guaranteed_confidence",
    "attainable_confidence",
    "monte_carlo",
    "robust_portfolio",
    "robustness_premium",
    "sobol_indices",
    "cooling_moments",
]


@dataclass
class ParameterSpec:
    """An uncertain parameter, given as a multiplicative or additive factor.

    Parameters
    ----------
    name
        Identifier used in reports.
    kind
        ``"thermal"`` scales the thermal response magnitudes, ``"a0"`` scales
        the permanent airborne fraction, ``"gamma"`` scales the climate carbon
        feedback strength, ``"re"`` scales one species radiative efficiency,
        ``"tau"`` scales one species lifetime, and ``"storage"`` scales the
        mean storage time of one intervention.
    low, high
        Bounds of the distribution.
    target
        Species name for ``"re"`` and ``"tau"``, or intervention index for
        ``"storage"``.
    distribution
        ``"uniform"`` or ``"lognormal"``. A lognormal draw is centred so that
        its median is the geometric mean of the bounds and its 95 per cent
        interval matches them.
    """

    name: str
    kind: str
    low: float
    high: float
    target: Optional[object] = None
    distribution: str = "uniform"

    def sample(self, u: np.ndarray) -> np.ndarray:
        """Map uniform variates on ``[0, 1]`` to parameter values."""
        u = np.asarray(u, dtype=float)
        if self.distribution == "uniform":
            return self.low + u * (self.high - self.low)
        if self.distribution == "lognormal":
            from scipy.stats import norm

            log_lo, log_hi = np.log(self.low), np.log(self.high)
            mu = 0.5 * (log_lo + log_hi)
            sigma = (log_hi - log_lo) / (2.0 * 1.959963984540054)
            return np.exp(mu + sigma * norm.ppf(np.clip(u, 1e-12, 1 - 1e-12)))
        raise ValueError("unknown distribution: {}".format(self.distribution))


def _apply(
    base: Portfolio, specs: Sequence[ParameterSpec], values: Sequence[float]
) -> Portfolio:
    """Rebuild a portfolio under one parameter draw."""
    thermal_scale = 1.0
    a0_scale = 1.0
    gamma_scale = 1.0
    re_scale: Dict[str, float] = {}
    tau_scale: Dict[str, float] = {}
    storage: Dict[int, float] = {}
    for spec, v in zip(specs, values):
        if spec.kind == "thermal":
            thermal_scale *= float(v)
        elif spec.kind == "a0":
            a0_scale *= float(v)
        elif spec.kind == "gamma":
            gamma_scale *= float(v)
        elif spec.kind == "re":
            re_scale[str(spec.target)] = re_scale.get(str(spec.target), 1.0) * float(v)
        elif spec.kind == "tau":
            tau_scale[str(spec.target)] = tau_scale.get(str(spec.target), 1.0) * float(v)
        elif spec.kind == "storage":
            storage[int(spec.target)] = storage.get(int(spec.target), 1.0) * float(v)
        else:
            raise ValueError("unknown parameter kind: {}".format(spec.kind))

    cm = base.climate.with_perturbation(
        thermal_scale=thermal_scale,
        re_scale=re_scale or None,
        tau_scale=tau_scale or None,
        a0_scale=a0_scale,
        gamma_scale=gamma_scale,
    )
    interventions = list(base.interventions)
    for idx, factor in storage.items():
        interventions[idx] = _rescale_storage(interventions[idx], factor)
    return Portfolio(cm, interventions, base.pathway, base.horizon)


def _rescale_storage(iv, factor: float):
    """Return a copy of an intervention with its timescales scaled."""
    import copy

    out = copy.copy(iv)
    for attr in ("tau", "release_time", "tau_disturbance"):
        if hasattr(out, attr):
            setattr(out, attr, getattr(out, attr) * float(factor))
    return out


def monte_carlo(
    base: Portfolio,
    specs: Sequence[ParameterSpec],
    n_samples: int = 512,
    solver: str = "cost",
    seed: int = 0,
    **solver_kwargs,
):
    """Propagate parameter uncertainty through the portfolio optimisation."""
    import pandas as pd

    sampler = qmc.Sobol(d=len(specs), scramble=True, seed=seed)
    u = sampler.random(n_samples)
    rows: List[Dict[str, float]] = []
    for k in range(n_samples):
        values = [spec.sample(u[k, j]) for j, spec in enumerate(specs)]
        trial = _apply(base, specs, values)
        res = (
            trial.minimise_cost(neutral=True, **solver_kwargs)
            if solver == "cost"
            else trial.minimise_deviation(**solver_kwargs)
        )
        row = {spec.name: float(v) for spec, v in zip(specs, values)}
        row.update(res.as_dict(base.names))
        row["cooling_total"] = float(res.alpha @ trial.cooling_vector())
        rows.append(row)
    return pd.DataFrame(rows)


def cooling_moments(
    base: Portfolio,
    specs: Sequence[ParameterSpec],
    n_samples: int = 512,
    seed: int = 0,
) -> Tuple[np.ndarray, np.ndarray, float, float]:
    """Mean and covariance of the cooling vector, and of the pathway warming.

    These are the only inputs the robust formulation requires, which is what
    makes it distributionally robust: no shape is assumed for the underlying
    distribution beyond these two moments.
    """
    sampler = qmc.Sobol(d=len(specs), scramble=True, seed=seed)
    u = sampler.random(n_samples)
    cooling = np.zeros((n_samples, base.n))
    warming = np.zeros(n_samples)
    for k in range(n_samples):
        values = [spec.sample(u[k, j]) for j, spec in enumerate(specs)]
        trial = _apply(base, specs, values)
        cooling[k] = trial.cooling_vector()
        warming[k] = trial.cumulative_warming()
    return (
        cooling.mean(axis=0),
        np.cov(cooling, rowvar=False).reshape(base.n, base.n),
        float(warming.mean()),
        float(warming.var(ddof=1)),
    )


def max_guaranteed_confidence(
    base: Portfolio,
    mean_b: np.ndarray,
    cov_b: np.ndarray,
    n_restart: int = 24,
    seed: int = 0,
) -> Dict[str, float]:
    """The highest confidence at which neutrality can be guaranteed at all.

    The robust constraint is positively homogeneous in :math:`\\alpha`, so
    along a ray :math:`\\alpha = s\\,d` with :math:`d \\ge 0` its left side is
    :math:`s\\,(\\mu^{\\mathsf T} d - \\kappa\\sqrt{d^{\\mathsf T}\\Sigma d})`.
    Scaling up therefore helps only when that bracket is positive, so a
    guarantee is attainable for some finite deployment if and only if

    .. math::
        \\kappa \\;<\\; \\kappa_{\\max}
        \\;=\\; \\max_{d \\ge 0,\\ d \\ne 0}
                \\frac{\\mu^{\\mathsf T} d}{\\sqrt{d^{\\mathsf T}\\Sigma d}} ,

    giving a maximum confidence
    :math:`\\eta_{\\max} = \\kappa_{\\max}^2/(1+\\kappa_{\\max}^2)`.

    This is a hard limit and no budget removes it: when the uncertainty in
    delivered cooling is proportional to the cooling itself, buying more
    deployment buys proportionally more uncertainty. Reporting it is more
    useful than reporting a solver failure, since it states the confidence
    beyond which the guarantee is unattainable in principle.
    """
    n = base.n
    rng = np.random.default_rng(seed)

    def ratio(d: np.ndarray) -> float:
        d = np.maximum(d, 0.0)
        nrm = float(np.sqrt(d @ cov_b @ d))
        if nrm <= 0:
            return 0.0
        return float(mean_b @ d / nrm)

    best, best_d = 0.0, np.ones(n) / n
    starts = [np.ones(n) / n] + list(np.eye(n)) + [
        rng.dirichlet(np.ones(n)) for _ in range(n_restart)
    ]
    for d0 in starts:
        res = minimize(
            lambda d: -ratio(d),
            np.maximum(np.asarray(d0, dtype=float), 1e-9),
            bounds=[(0.0, 1.0)] * n,
            constraints=[{"type": "eq", "fun": lambda d: np.sum(d) - 1.0}],
            method="SLSQP",
            options={"maxiter": 400, "ftol": 1e-12},
        )
        val = ratio(res.x)
        if val > best:
            best, best_d = val, np.maximum(res.x, 0.0)
    return {
        "kappa_max": best,
        "confidence_max": best * best / (1.0 + best * best),
        "direction": best_d / max(float(np.sum(best_d)), 1e-30),
    }


def attainable_confidence(
    base: Portfolio,
    mean_b: np.ndarray,
    cov_b: np.ndarray,
    target_warming: float,
    warming_variance: float = 0.0,
    n_bisect: int = 24,
) -> float:
    """The highest confidence at which a guarantee is actually attainable.

    :func:`max_guaranteed_confidence` gives the limit set by the covariance
    structure alone, ignoring capacity. This function respects the deployment
    ceilings and the uncertainty in the pathway warming as well, and so returns
    the operationally meaningful ceiling, which is the lower of the two. Found
    by bisection on feasibility, which is monotone in the confidence because
    the required margin grows with it.
    """
    lo, hi = 0.5, 1.0 - 1e-6
    if not robust_portfolio(
        base, mean_b, cov_b, target_warming, warming_variance, lo
    )["feasible"]:
        return float("nan")
    for _ in range(n_bisect):
        mid = 0.5 * (lo + hi)
        if robust_portfolio(
            base, mean_b, cov_b, target_warming, warming_variance, mid
        )["feasible"]:
            lo = mid
        else:
            hi = mid
    return float(lo)


def robust_portfolio(
    base: Portfolio,
    mean_b: np.ndarray,
    cov_b: np.ndarray,
    target_warming: float,
    warming_variance: float = 0.0,
    confidence: float = 0.9,
) -> Dict[str, object]:
    """Least cost deployment meeting neutrality under moment ambiguity.

    Solves

    .. math::
        \\min_\\alpha \\sum_i C_i(\\alpha_i)
        \\quad\\text{s.t.}\\quad
        \\mu^{\\mathsf T}\\alpha
          - \\kappa \\sqrt{\\alpha^{\\mathsf T}\\Sigma\\alpha + \\sigma_E^2}
          \\ge B_E ,
        \\quad \\alpha \\ge 0 ,

    a convex programme, since the constraint function is concave. The
    multiplier :math:`\\kappa = \\sqrt{\\eta/(1-\\eta)}` is the tight Cantelli
    value, so the guarantee holds for every distribution with the stated
    moments and does not rely on normality. It grows without bound as the
    required confidence approaches one, which is the price of insisting on a
    guarantee under an adversarial distribution.
    """
    eta = float(min(max(confidence, 1e-9), 1.0 - 1e-9))
    kappa = float(np.sqrt(eta / (1.0 - eta)))
    s = base.scales()
    a_ref, cost_ref = s["alpha"], s["cost"]
    row = float(max(abs(target_warming), s["cooling"] * a_ref))
    mu_s = mean_b * a_ref / row
    cov_s = cov_b * (a_ref / row) ** 2
    var_e = warming_variance / (row * row)
    be_s = target_warming / row

    def margin(x: np.ndarray) -> float:
        var = float(x @ cov_s @ x + var_e)
        return float(mu_s @ x - kappa * np.sqrt(max(var, 0.0)) - be_s)

    def margin_jac(x: np.ndarray) -> np.ndarray:
        var = float(x @ cov_s @ x + var_e)
        if var <= 0:
            return mu_s
        return mu_s - kappa * (cov_s @ x) / np.sqrt(var)

    def cost_s(x: np.ndarray) -> float:
        return base.cost(x * a_ref) / cost_ref

    def cost_jac_s(x: np.ndarray) -> np.ndarray:
        return base.cost_gradient(x * a_ref) * a_ref / cost_ref

    x0 = np.full(base.n, max(be_s, 1e-6) / max(float(np.sum(mu_s)), 1e-12))
    res = minimize(
        cost_s,
        np.maximum(x0, 1e-9),
        jac=cost_jac_s,
        bounds=base._scaled_bounds(a_ref),
        constraints=[{"type": "ineq", "fun": margin, "jac": margin_jac}],
        method="SLSQP",
        options={"maxiter": 1200, "ftol": 1e-14},
    )
    alpha = np.maximum(res.x, 0.0) * a_ref
    var = float(alpha @ cov_b @ alpha + warming_variance)
    margin = float(
        mean_b @ alpha - kappa * np.sqrt(max(var, 0.0)) - target_warming
    )
    # A converged solver result is only meaningful if the guarantee actually
    # holds. Beyond the attainable confidence the feasible set is empty and the
    # optimiser stops at an arbitrary point, which must not be reported as a
    # cheap portfolio.
    tol = 1e-8 * max(abs(target_warming), 1.0)
    feasible = margin >= -tol
    return {
        "alpha": alpha if feasible else np.full_like(alpha, np.nan),
        "cost": base.cost(alpha) if feasible else float("inf"),
        "success": bool(res.success and feasible),
        "feasible": bool(feasible),
        "message": str(res.message)
        if feasible
        else "guarantee unattainable at this confidence",
        "kappa": kappa,
        "expected_cooling": float(mean_b @ alpha),
        "cooling_std": float(np.sqrt(max(var, 0.0))),
        "guaranteed_margin": margin,
    }


def robustness_premium(
    base: Portfolio,
    mean_b: np.ndarray,
    cov_b: np.ndarray,
    target_warming: float,
    warming_variance: float = 0.0,
    confidences: Sequence[float] = (0.5, 0.75, 0.9, 0.95, 0.99),
):
    """Cost of a neutrality guarantee as a function of the confidence level.

    At confidence one half the constraint reduces to neutrality in
    expectation, so the premium is measured against that reference.
    """
    import pandas as pd

    rows = []
    reference = None
    limit = max_guaranteed_confidence(base, mean_b, cov_b)
    for eta in confidences:
        out = robust_portfolio(
            base, mean_b, cov_b, target_warming, warming_variance, eta
        )
        if reference is None and out["feasible"]:
            reference = out["cost"]
        rows.append(
            {
                "confidence": eta,
                "kappa": out["kappa"],
                "cost": out["cost"],
                "feasible": out["feasible"],
                "confidence_max": limit["confidence_max"],
                "premium_fraction": out["cost"] / reference - 1.0
                if reference and out["feasible"]
                else np.nan,
                "expected_cooling": out["expected_cooling"],
                "cooling_std": out["cooling_std"],
                "success": out["success"],
                **{
                    "alpha_" + n: float(a)
                    for n, a in zip(base.names, out["alpha"])
                },
            }
        )
    return pd.DataFrame(rows)


def sobol_indices(
    fn: Callable[[np.ndarray], float],
    specs: Sequence[ParameterSpec],
    n_base: int = 512,
    seed: int = 0,
):
    """First and total order Sobol indices by the Saltelli estimator.

    ``fn`` receives a vector of parameter values and returns a scalar. The
    estimator needs ``n_base * (d + 2)`` evaluations.
    """
    import pandas as pd

    d = len(specs)
    sampler = qmc.Sobol(d=2 * d, scramble=True, seed=seed)
    u = sampler.random(n_base)
    ua, ub = u[:, :d], u[:, d:]

    def to_values(rows: np.ndarray) -> np.ndarray:
        out = np.empty_like(rows)
        for j, spec in enumerate(specs):
            out[:, j] = spec.sample(rows[:, j])
        return out

    va, vb = to_values(ua), to_values(ub)
    ya = np.array([fn(v) for v in va])
    yb = np.array([fn(v) for v in vb])
    both = np.concatenate([ya, yb])
    var = float(np.var(both, ddof=1))
    rows = []
    if var <= 0:
        for spec in specs:
            rows.append(
                {"parameter": spec.name, "kind": spec.kind, "S1": 0.0, "ST": 0.0,
                 "interaction": 0.0}
            )
        return pd.DataFrame(rows)

    # The outputs are centred before forming the products. The first order
    # estimator is a covariance, and evaluating it on uncentred values
    # subtracts two nearly equal quantities of order (mean)^2: with a mean
    # several times the standard deviation, as here, that cancellation
    # dominates the result and drives the estimates negative. Centring costs
    # nothing and removes the bias. The total order estimator depends only on
    # differences of outputs and is unaffected.
    mu = float(np.mean(both))
    ca, cb = ya - mu, yb - mu
    for j, spec in enumerate(specs):
        vab = va.copy()
        vab[:, j] = vb[:, j]
        yab = np.array([fn(v) for v in vab])
        cab = yab - mu
        # Saltelli et al. (2010) estimators, Table 2.
        s1 = float(np.mean(cb * (cab - ca)) / var)
        st = float(np.mean((ca - cab) ** 2) / (2.0 * var))
        rows.append(
            {
                "parameter": spec.name,
                "kind": spec.kind,
                "S1": s1,
                "ST": st,
                "interaction": st - s1,
            }
        )
    df = pd.DataFrame(rows)
    df.attrs["variance"] = var
    df.attrs["n_evaluations"] = n_base * (d + 2)
    return df


def portfolio_output(
    base: Portfolio,
    specs: Sequence[ParameterSpec],
    quantity: str = "cost",
    **solver_kwargs,
) -> Callable[[np.ndarray], float]:
    """Build a scalar output function of the parameters, for Sobol analysis."""

    def fn(values: np.ndarray) -> float:
        trial = _apply(base, specs, values)
        res = trial.minimise_cost(neutral=True, **solver_kwargs)
        if quantity == "cost":
            return float(res.cost)
        if quantity == "deployment":
            return float(np.sum(res.alpha))
        if quantity == "deviation":
            return float(res.deviation)
        if quantity.startswith("alpha_"):
            idx = base.names.index(quantity[len("alpha_") :])
            return float(res.alpha[idx])
        raise ValueError("unknown quantity: {}".format(quantity))

    return fn
