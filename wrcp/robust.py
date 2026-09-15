"""Robust portfolio optimisation under fire risk.

The deterministic framework requires forcing neutrality exactly,
:math:`\\sum_i \\alpha_i b_i = B_E`. Under fire risk the delivered cooling
:math:`b_i` is a random variable, so the equality is replaced by a chance
constraint: the portfolio must deliver at least the warming of the pathway with
probability at least :math:`\\eta`. Requiring that for *every* distribution with
the mean and covariance computed in :mod:`wrcp.stochastic` is equivalent, by the
tight one-sided Chebyshev bound of Cantelli, to the second-order cone constraint

.. math::
    \\mu^{\\mathsf T}\\alpha
      - \\kappa(\\eta)\\sqrt{\\alpha^{\\mathsf T}\\Sigma\\,\\alpha + \\sigma_E^2}
    \\;\\ge\\; B_E ,
    \\qquad \\kappa(\\eta) = \\sqrt{\\eta/(1-\\eta)} .

The left side is concave, so the least-cost problem is convex and has a unique
optimum whenever the cost functions are convex. No distributional assumption
enters beyond the two moments, which matters because burn-severity distributions
are fat tailed: at tail index :math:`1.5` the chance of an event ten times the
mean is some three orders of magnitude above the exponential value, and a
Gaussian or log-normal formulation would price that away.

Three results follow and are computed here.

The **fire premium** is the extra cost of a guarantee at confidence
:math:`\\eta` relative to deterministic accounting. It is decomposed into the
part attributable to the mean loss of cooling, which deterministic accounting
simply over-credits, and the part attributable to the variance, which is the
price of the guarantee itself. That split matters for policy: the first part is
a correction any honest inventory must make, while the second is a choice about
how much confidence to buy.

The **confidence ceiling** is the highest :math:`\\eta` any finite portfolio can
guarantee. Because the constraint is positively homogeneous in :math:`\\alpha`,
scaling a portfolio up multiplies both the mean and the standard deviation, so
the signal-to-noise ratio is unchanged and no budget raises the ceiling. Under
fire risk the mean falls and the variance rises, so the ceiling falls relative
to the deterministic case; capacity limits and pathway uncertainty can lower it
further, and both bounds are reported.

The **cost-effectiveness condition** is the first-order condition of the convex
programme. Equation (49) of the research plan prints it with
:math:`\\sigma_i^2 \\alpha_i^*` in the numerator of the risk term, which
omits the off-diagonal covariance and so cannot be the derivative of the
constraint whenever measures are correlated. Differentiating the constraint
gives the marginal risk of measure :math:`i` as
:math:`(\\Sigma\\alpha)_i / \\sqrt{\\alpha^{\\mathsf T}\\Sigma\\alpha
+ \\sigma_E^2}`, and it is the full matrix-vector product that appears, not the
diagonal alone. With correlated fire risk the difference is first order:
a measure in a region already heavily represented in the portfolio is penalised
through the off-diagonal terms even when its own variance is small.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Dict, List, Optional, Sequence, Tuple

import numpy as np
from scipy.optimize import minimize

from cipo.climate import ClimateModel
from cipo.pathways import EmissionPathway
from cipo.portfolio import Portfolio
from cipo.profiles import LinearCost

from .stochastic import FireExposedStore, FireMoments, cooling_moments

__all__ = [
    "RobustResult",
    "FirePortfolio",
    "kappa",
    "confidence_from_kappa",
]


def kappa(eta: float) -> float:
    """The Cantelli multiplier ``sqrt(eta / (1 - eta))``."""
    e = float(np.clip(eta, 1e-12, 1.0 - 1e-12))
    return float(np.sqrt(e / (1.0 - e)))


def confidence_from_kappa(k: float) -> float:
    """Invert :func:`kappa`: ``eta = k^2 / (1 + k^2)``."""
    k = float(max(k, 0.0))
    return float(k * k / (1.0 + k * k))


@dataclass
class RobustResult:
    """One solved robust portfolio."""

    alpha: np.ndarray
    cost: float
    confidence: float
    kappa: float
    expected_cooling: float
    cooling_std: float
    required_cooling: float
    margin: float
    feasible: bool
    success: bool
    message: str = ""
    multiplier: Optional[float] = None
    diagnostics: Dict[str, object] = field(default_factory=dict)

    def shares(self) -> np.ndarray:
        total = float(np.sum(self.alpha))
        return self.alpha / total if total > 0 else self.alpha * 0.0

    def as_dict(self, names: Sequence[str]) -> Dict[str, object]:
        out = {
            "confidence": self.confidence,
            "kappa": self.kappa,
            "cost": self.cost,
            "expected_cooling": self.expected_cooling,
            "cooling_std": self.cooling_std,
            "required_cooling": self.required_cooling,
            "margin": self.margin,
            "feasible": self.feasible,
            "success": self.success,
        }
        for n, a in zip(names, self.alpha):
            out["alpha_" + n] = float(a)
        return out


class FirePortfolio:
    """A set of fire-exposed measures optimised against one emission pathway.

    The deterministic machinery of :class:`cipo.portfolio.Portfolio` is reused
    for the pathway warming, the costs and the numerical conditioning; the
    moments of delivered cooling come from :mod:`wrcp.stochastic`.

    Every optimisation is posed in dimensionless variables. Cooling per unit
    deployment is of order :math:`10^{-14}` K yr per kilogram, so a constraint
    written in natural units sits far inside any solver's feasibility tolerance
    and is satisfied trivially and wrongly. The reference scales are properties
    of the problem, so this changes no solution, only its conditioning.
    """

    def __init__(
        self,
        climate: ClimateModel,
        stores: Sequence[FireExposedStore],
        pathway: Optional[EmissionPathway] = None,
        horizon: float = 100.0,
        n_block: int = 10,
        rho_within: float = 0.6,
        rho_between: float = 0.15,
        sharing: float = 0.5,
        weather_spread: float = 0.35,
        n_weather: int = 9,
        pathway_variance: float = 0.0,
    ) -> None:
        self.climate = climate
        self.stores = list(stores)
        self.pathway = pathway
        self.horizon = float(horizon)
        self.n_block = int(n_block)
        self.rho_within = float(rho_within)
        self.rho_between = float(rho_between)
        self.sharing = float(sharing)
        self.weather_spread = float(weather_spread)
        self.n_weather = int(n_weather)
        self.pathway_variance = float(pathway_variance)
        self._moments: Optional[FireMoments] = None
        self._base = Portfolio(
            climate,
            [s.intervention for s in self.stores],
            pathway,
            self.horizon,
        )

    # ------------------------------------------------------------- basic data
    @property
    def n(self) -> int:
        return len(self.stores)

    @property
    def names(self) -> List[str]:
        return [s.name for s in self.stores]

    @property
    def labels(self) -> List[str]:
        return [s.label for s in self.stores]

    @property
    def deterministic(self) -> Portfolio:
        """The same problem with fire switched off, for the premium baseline."""
        return self._base

    def moments(self) -> FireMoments:
        """Mean and covariance of the cooling vector, cached."""
        if self._moments is None:
            self._moments = cooling_moments(
                self.stores,
                self.climate,
                self.horizon,
                n_block=self.n_block,
                rho_within=self.rho_within,
                rho_between=self.rho_between,
                sharing=self.sharing,
                weather_spread=self.weather_spread,
                n_weather=self.n_weather,
            )
        return self._moments

    def required_cooling(self) -> float:
        """``B_E(TH)``, the cumulative warming to be offset."""
        return float(self._base.cumulative_warming())

    def caps(self) -> np.ndarray:
        return np.array(
            [
                np.inf if s.max_scale is None else float(s.max_scale)
                for s in self.stores
            ]
        )

    def cost(self, alpha: Sequence[float]) -> float:
        return float(
            sum(
                s.cost(a)
                for s, a in zip(self.stores, np.asarray(alpha, dtype=float))
            )
        )

    def cost_gradient(self, alpha: Sequence[float]) -> np.ndarray:
        return np.array(
            [
                s.marginal_cost(a)
                for s, a in zip(self.stores, np.asarray(alpha, dtype=float))
            ]
        )

    # --------------------------------------------------------------- geometry
    def margin(self, alpha: Sequence[float], confidence: float) -> float:
        """Slack in the chance constraint, in K yr. Non-negative iff feasible."""
        a = np.asarray(alpha, dtype=float)
        m = self.moments()
        var = float(a @ m.covariance @ a + self.pathway_variance)
        return float(
            m.mean @ a
            - kappa(confidence) * np.sqrt(max(var, 0.0))
            - self.required_cooling()
        )

    def attainability(self) -> Dict[str, object]:
        """Whether the guarantee is reachable, and what limits it.

        Two distinct ceilings are reported. The *covariance* ceiling follows
        from the signal-to-noise ratio alone and ignores capacity; it is the
        fundamental limit of Eq. (57) of the research plan. The *capacity*
        ceiling additionally respects the deployment limits and the pathway
        uncertainty, and is the one a policy maker faces. The binding ceiling is
        the lower of the two.
        """
        m = self.moments()
        cov = m.covariance
        caps = self.caps()
        n = self.n

        def ratio(d: np.ndarray) -> float:
            d = np.maximum(d, 0.0)
            nrm = float(np.sqrt(max(d @ cov @ d, 0.0)))
            if nrm <= 0.0:
                return np.inf if float(m.mean @ d) > 0 else 0.0
            return float(m.mean @ d / nrm)

        # A direction of zero variance gives an unbounded ratio, which is the
        # correct answer: a portfolio of fire-proof measures can guarantee
        # neutrality at any confidence, provided it has the capacity.
        best, best_d = 0.0, np.ones(n) / n
        rng = np.random.default_rng(0)
        starts = [np.ones(n) / n] + [e for e in np.eye(n)] + [
            rng.dirichlet(np.ones(n)) for _ in range(32)
        ]
        for d0 in starts:
            r = ratio(np.asarray(d0, dtype=float))
            if not np.isfinite(r):
                best, best_d = np.inf, np.asarray(d0, dtype=float)
                break
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

        unbounded = bool(np.any(~np.isfinite(caps) & (m.mean > 0.0)))
        finite = np.isfinite(caps)
        max_mean = float(np.sum(m.mean[finite] * caps[finite]))
        return {
            "kappa_max": best,
            "confidence_max": 1.0 if not np.isfinite(best) else confidence_from_kappa(best),
            "direction": best_d / max(float(np.sum(best_d)), 1e-30),
            "required_cooling": self.required_cooling(),
            "max_expected_cooling": np.inf if unbounded else max_mean,
            "capacity_sufficient": bool(
                unbounded or max_mean >= self.required_cooling()
            ),
        }

    def restricted(self, keep) -> "FirePortfolio":
        """The same problem with only the selected measures available.

        ``keep`` is a boolean mask or a sequence of names. Used to isolate the
        confidence ceiling of a portfolio that has no fire-proof option, which
        is where the ceiling genuinely binds.
        """
        if isinstance(keep, (list, tuple, np.ndarray)) and np.asarray(keep).dtype == bool:
            mask = np.asarray(keep, dtype=bool)
        else:
            wanted = set(keep)
            mask = np.array([n in wanted for n in self.names], dtype=bool)
        return FirePortfolio(
            self.climate,
            [s for s, m in zip(self.stores, mask) if m],
            self.pathway,
            self.horizon,
            n_block=self.n_block,
            rho_within=self.rho_within,
            rho_between=self.rho_between,
            sharing=self.sharing,
            weather_spread=self.weather_spread,
            n_weather=self.n_weather,
            pathway_variance=self.pathway_variance,
        )

    def exposed_only(self) -> "FirePortfolio":
        """The sub-portfolio of measures that fire can actually reach."""
        return self.restricted(
            np.array([not s.hazard.is_inert for s in self.stores], dtype=bool)
        )

    def ceiling_report(self) -> Dict[str, object]:
        """The confidence ceiling, and the condition under which it binds.

        Equation (57) of the research plan presents :math:`\\eta_{\\max}` as a
        fundamental limit on guaranteed neutrality, on the grounds that scaling
        a portfolio up buys uncertainty in proportion to cooling. That argument
        is correct but its premise is not always met, and the distinction is
        the most policy-relevant thing the ceiling has to say.

        The homogeneity argument bounds the ratio
        :math:`\\mu^{\\mathsf T}d / \\sqrt{d^{\\mathsf T}\\Sigma d}` over
        non-negative directions. If even one available measure is beyond the
        reach of fire, that measure spans a direction with zero variance and
        positive mean, the ratio is unbounded along it, and there is no ceiling
        at all: neutrality can be guaranteed at any confidence by buying enough
        of it. The ceiling therefore does not arise from fire risk as such. It
        arises from the *absence or exhaustion of a fire-proof option*, and so
        it is a statement about capacity and cost, not about physics.

        Three numbers are returned. The ceiling of the full portfolio is
        typically unity and is reported for completeness. The ceiling of the
        fire-exposed sub-portfolio is the fundamental limit the plan describes,
        and it is finite. The attainable ceiling respects capacity limits and
        is the one a crediting scheme faces.
        """
        full = self.attainability()
        exposed = self.exposed_only()
        ex = exposed.attainability()
        return {
            "confidence_max_full": full["confidence_max"],
            "kappa_max_full": full["kappa_max"],
            "confidence_max_exposed_only": ex["confidence_max"],
            "kappa_max_exposed_only": ex["kappa_max"],
            "exposed_direction": ex["direction"],
            "exposed_names": exposed.labels,
            "attainable_with_capacity": self.attainable_confidence(),
            "attainable_exposed_only": exposed.attainable_confidence(),
            "fireproof_available": bool(
                any(s.hazard.is_inert for s in self.stores)
            ),
        }

    def attainable_confidence(self, n_bisect: int = 30) -> float:
        """The highest confidence at which a portfolio is actually feasible.

        Respects capacity and pathway uncertainty as well as the covariance,
        so it is the operationally meaningful ceiling. Feasibility is monotone
        in the confidence because the required margin grows with it, which is
        what makes a bisection valid.
        """
        lo, hi = 0.5, 1.0 - 1e-7
        if not self.solve(lo).feasible:
            return float("nan")
        if self.solve(hi).feasible:
            return float(hi)
        for _ in range(int(n_bisect)):
            mid = 0.5 * (lo + hi)
            if self.solve(mid).feasible:
                lo = mid
            else:
                hi = mid
        return float(lo)

    # ------------------------------------------------------------ the solver
    def solve(self, confidence: float = 0.9) -> RobustResult:
        """Least-cost portfolio guaranteeing neutrality at ``confidence``."""
        m = self.moments()
        be = self.required_cooling()
        k = kappa(confidence)

        s = self._base.scales()
        a_ref, cost_ref = s["alpha"], s["cost"]
        row = float(max(abs(be), s["cooling"] * a_ref, 1e-300))
        mu_s = m.mean * a_ref / row
        cov_s = m.covariance * (a_ref / row) ** 2
        var_e = self.pathway_variance / (row * row)
        be_s = be / row
        bounds = [
            (0.0, None if s_i.max_scale is None else float(s_i.max_scale) / a_ref)
            for s_i in self.stores
        ]

        def margin_s(x: np.ndarray) -> float:
            var = float(x @ cov_s @ x + var_e)
            return float(mu_s @ x - k * np.sqrt(max(var, 0.0)) - be_s)

        def margin_jac(x: np.ndarray) -> np.ndarray:
            var = float(x @ cov_s @ x + var_e)
            if var <= 0.0:
                return mu_s
            return mu_s - k * (cov_s @ x) / np.sqrt(var)

        def cost_s(x: np.ndarray) -> float:
            return self.cost(x * a_ref) / cost_ref

        def cost_jac_s(x: np.ndarray) -> np.ndarray:
            return self.cost_gradient(x * a_ref) * a_ref / cost_ref

        x0 = self._start(mu_s, be_s, bounds)
        res = minimize(
            cost_s,
            x0,
            jac=cost_jac_s,
            bounds=bounds,
            constraints=[
                {"type": "ineq", "fun": margin_s, "jac": margin_jac}
            ],
            method="SLSQP",
            options={"maxiter": 2000, "ftol": 1e-14},
        )
        alpha = np.maximum(res.x, 0.0) * a_ref
        var = float(alpha @ m.covariance @ alpha + self.pathway_variance)
        margin = float(m.mean @ alpha - k * np.sqrt(max(var, 0.0)) - be)
        # A converged solver result means nothing unless the guarantee holds.
        # Beyond the attainable confidence the feasible set is empty and the
        # optimiser stops wherever it happens to be; reporting that as a cheap
        # portfolio would invert the conclusion of the whole study.
        tol = 1e-7 * max(abs(be), 1e-300)
        feasible = margin >= -tol
        return RobustResult(
            alpha=alpha if feasible else np.full(self.n, np.nan),
            cost=self.cost(alpha) if feasible else float("inf"),
            confidence=float(confidence),
            kappa=k,
            expected_cooling=float(m.mean @ alpha),
            cooling_std=float(np.sqrt(max(var, 0.0))),
            required_cooling=be,
            margin=margin,
            feasible=bool(feasible),
            success=bool(res.success and feasible),
            message=str(res.message)
            if feasible
            else "guarantee unattainable at this confidence",
            diagnostics={"iterations": int(getattr(res, "nit", -1))},
        )

    def _start(
        self, mu_s: np.ndarray, be_s: float, bounds: Sequence[Tuple[float, Optional[float]]]
    ) -> np.ndarray:
        """A starting point that meets neutrality in the mean if it can."""
        x0 = np.zeros(self.n)
        live = mu_s > 0.0
        if be_s <= 0.0 or not np.any(live):
            return x0
        share = be_s / float(np.count_nonzero(live))
        for i in range(self.n):
            if not live[i]:
                continue
            hi = bounds[i][1]
            val = share / mu_s[i]
            x0[i] = val if hi is None else min(val, hi)
        return np.maximum(x0, 1e-9)

    # ------------------------------------------------------- the fire premium
    def deterministic_solution(self):
        """Least-cost portfolio under deterministic durability, the baseline.

        This is what current accounting would buy: it credits every measure
        with the cooling it would deliver if no fire occurred.
        """
        return self._base.minimise_cost(neutral=True)

    def premium(
        self, confidences: Sequence[float] = (0.5, 0.75, 0.9, 0.95, 0.99)
    ):
        """Fire premium against the deterministic baseline, by confidence.

        Two references are reported for each confidence level. The premium
        against the *deterministic* cost is the total correction, and is the
        number a crediting standard would have to absorb. The premium against
        the cost at confidence one half, where the constraint reduces to
        neutrality in expectation, isolates the cost of the guarantee from the
        cost of the mean over-credit. Their difference is the decomposition the
        research plan asks for.
        """
        import pandas as pd

        det = self.deterministic_solution()
        det_cost = float(det.cost) if det.success else float("nan")
        ceiling = self.attainability()
        rows = []
        mean_ref = None
        for eta in confidences:
            out = self.solve(float(eta))
            if mean_ref is None and abs(eta - 0.5) < 1e-12 and out.feasible:
                mean_ref = out.cost
            rows.append(
                {
                    "confidence": float(eta),
                    "kappa": out.kappa,
                    "cost": out.cost,
                    "feasible": out.feasible,
                    "premium_vs_deterministic": out.cost / det_cost - 1.0
                    if out.feasible and np.isfinite(det_cost) and det_cost > 0
                    else np.nan,
                    "expected_cooling": out.expected_cooling,
                    "cooling_std": out.cooling_std,
                    "confidence_max": ceiling["confidence_max"],
                    **{
                        "alpha_" + n: float(a)
                        for n, a in zip(self.names, out.alpha)
                    },
                }
            )
        frame = pd.DataFrame(rows)
        if mean_ref is None:
            half = self.solve(0.5)
            mean_ref = half.cost if half.feasible else np.nan
        frame["premium_vs_mean_neutral"] = np.where(
            frame["feasible"] & np.isfinite(mean_ref) & (mean_ref > 0),
            frame["cost"] / mean_ref - 1.0,
            np.nan,
        )
        frame.attrs["deterministic_cost"] = det_cost
        frame.attrs["mean_neutral_cost"] = float(mean_ref)
        # At confidence one half the Cantelli multiplier is exactly one, so the
        # constraint is NOT neutrality in expectation: it still subtracts one
        # standard deviation. The cost of neutrality in expectation is obtained
        # by solving with the variance switched off, which is the only way to
        # separate the mean over-credit from the price of any guarantee at all.
        expectation_cost = self._expectation_only_cost()
        frame.attrs["expectation_cost"] = expectation_cost
        frame.attrs["mean_over_credit"] = (
            float(expectation_cost / det_cost - 1.0)
            if np.isfinite(det_cost) and det_cost > 0 and np.isfinite(expectation_cost)
            else np.nan
        )
        frame["premium_vs_expectation"] = np.where(
            frame["feasible"] & np.isfinite(expectation_cost) & (expectation_cost > 0),
            frame["cost"] / expectation_cost - 1.0,
            np.nan,
        )
        return frame

    def _expectation_only_cost(self) -> float:
        """Least cost of meeting neutrality in expectation, ignoring variance.

        The reference against which the mean over-credit of deterministic
        accounting is measured. It is the same programme with the covariance
        set to zero, so it isolates the effect of crediting a store for cooling
        it does not on average deliver, with no allowance for uncertainty.
        """
        m = self.moments()
        twin = FirePortfolio(
            self.climate,
            self.stores,
            self.pathway,
            self.horizon,
            n_block=self.n_block,
            rho_within=self.rho_within,
            rho_between=self.rho_between,
            sharing=self.sharing,
            weather_spread=self.weather_spread,
            n_weather=self.n_weather,
        )
        twin._moments = FireMoments(
            mean=m.mean,
            covariance=np.zeros_like(m.covariance),
            idiosyncratic=np.zeros_like(m.covariance),
            common=np.zeros_like(m.covariance),
            deterministic=m.deterministic,
            names=list(m.names),
        )
        res = twin.solve(0.5)
        return float(res.cost) if res.feasible else float("nan")

    # ------------------------------------------------------ optimality report
    def kkt_report(
        self,
        alpha: Sequence[float],
        confidence: float = 0.9,
        tol: float = 1e-9,
    ) -> Dict[str, object]:
        """Verify the first-order conditions of the robust programme.

        At an interior optimum the marginal cost per unit of *risk-adjusted*
        cooling is equalised,

        .. math::
            \\frac{C_i'(\\alpha_i^*)}
                  {\\mu_i - \\kappa(\\eta)\\,
                   (\\Sigma\\alpha^*)_i / \\sqrt{\\alpha^{*\\mathsf T}\\Sigma
                   \\alpha^* + \\sigma_E^2}} = \\lambda ,

        with the inequality in the corresponding direction at each bound. The
        denominator is the *effective* cooling of the measure: its mean
        contribution less the marginal risk it adds. A measure whose effective
        cooling is non-positive cannot help at any price, however cheap; that
        is the precise sense in which fire-resistant measures are favoured, and
        it is reported as ``effective_cooling``.
        """
        a = np.asarray(alpha, dtype=float)
        m = self.moments()
        eta = float(confidence)
        k = kappa(eta)
        var = float(a @ m.covariance @ a + self.pathway_variance)
        marginal_risk = (
            (m.covariance @ a) / np.sqrt(var) if var > 0.0 else np.zeros_like(a)
        )
        effective = m.mean - k * marginal_risk
        marg = self.cost_gradient(a)
        with np.errstate(divide="ignore", invalid="ignore"):
            ratio = np.where(effective > 0.0, marg / effective, np.inf)
        caps = self.caps()
        interior = (a > tol) & (a < caps * (1.0 - 1e-9))
        at_zero = a <= tol
        at_cap = a >= caps * (1.0 - 1e-9)
        if np.any(interior):
            lam = float(np.mean(ratio[interior]))
            spread = (
                float(np.max(ratio[interior]) - np.min(ratio[interior]))
                if np.count_nonzero(interior) > 1
                else 0.0
            )
        else:
            active = a > tol
            lam = float(np.min(ratio[active])) if np.any(active) else float("nan")
            spread = 0.0
        return {
            "multiplier": lam,
            "ratio": ratio,
            "effective_cooling": effective,
            "marginal_risk": marginal_risk,
            "mean_cooling": m.mean,
            "interior": interior,
            "at_zero": at_zero,
            "at_capacity": at_cap,
            "equalisation_spread": spread,
            "relative_spread": spread / lam if lam and np.isfinite(lam) else 0.0,
            "zero_condition_satisfied": bool(
                np.all(ratio[at_zero] >= lam * (1.0 - 1e-6))
                if np.any(at_zero) and np.isfinite(lam)
                else True
            ),
            "margin": self.margin(a, eta),
        }

    def summary_frame(self, alpha: Sequence[float], confidence: float = 0.9):
        import pandas as pd

        a = np.asarray(alpha, dtype=float)
        m = self.moments()
        k = kappa(confidence)
        var = float(a @ m.covariance @ a + self.pathway_variance)
        marginal_risk = (
            (m.covariance @ a) / np.sqrt(var) if var > 0.0 else np.zeros_like(a)
        )
        total = float(np.sum(a))
        return pd.DataFrame(
            {
                "measure": self.labels,
                "alpha": a,
                "share": a / total if total > 0 else a,
                "cooling_deterministic": m.deterministic,
                "cooling_mean": m.mean,
                "cooling_std": m.std,
                "effective_cooling": m.mean - k * marginal_risk,
                "loss_fraction": m.loss_fraction,
                "cost": [s.cost(v) for s, v in zip(self.stores, a)],
                "marginal_cost": self.cost_gradient(a),
                "hazard_lambda0": [s.hazard.lambda0 for s in self.stores],
                "region": [s.hazard.region for s in self.stores],
                "mean_storage_time": [
                    s.intervention.mean_storage_time for s in self.stores
                ],
            }
        )
