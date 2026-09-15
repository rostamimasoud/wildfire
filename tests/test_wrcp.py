"""Test suite for the fire-integrated framework.

The tests are organised by what could go wrong rather than by module. Each one
compares against something obtained a different way: an analytic limit, an
independent numerical method, a simulation of the underlying process, or a
structural property the mathematics guarantees.
"""

from __future__ import annotations

import numpy as np
import pytest

from cipo import (
    AR6,
    Afforestation,
    DelayedPulse,
    ExponentialRelease,
    LinearRelease,
    PermanentRemoval,
    PowerCost,
)
from cipo.expsum import ExpSum, Signal
from wrcp.algebra import (
    Piecewise,
    breakpoints,
    merge_edges,
    rebase,
    reverse_weighted_integral,
    scaled_definite,
    signal_product,
    stock_signal,
    triangle_integral,
    triangle_integral_quadrature,
)
from wrcp.calibration import (
    beta_from_intensification,
    lambda_from_return_interval,
    severity_from_class_fractions,
    spread_from_extreme,
)
from wrcp.hazard import (
    NO_FIRE,
    STAND_REPLACING,
    BetaSeverity,
    FireHazard,
    TruncatedParetoSeverity,
    TwoPointSeverity,
    cross_severity_exponent,
    drought_path,
    temperature_path,
)
from wrcp.regions import BOREAL_CANADA, WESTERN_US, build_stores
from wrcp.robust import FirePortfolio, confidence_from_kappa, kappa
from wrcp.spike import spike_amplitude, spike_trajectory, worst_case_spike
from wrcp.stochastic import FireExposedStore, block_edges, cooling_moments

HORIZON = 100.0


@pytest.fixture(scope="module")
def cm():
    return AR6()


@pytest.fixture(scope="module")
def kern(cm):
    return cm.agtp_kernel("CO2")


@pytest.fixture(scope="module")
def hazard():
    return FireHazard(
        lambda0=1.0 / 70.0,
        beta_T=0.34,
        beta_D=0.25,
        severity=BetaSeverity(0.6, 4.0),
        temperature=temperature_path(1.4, 4.0, 55.0),
        drought=drought_path(0.0, 1.2, HORIZON),
        region="test",
    )


FAMILIES = [
    ("exponential", ExponentialRelease(tau=50.0)),
    ("delayed pulse", DelayedPulse(tau=50.0)),
    ("linear release", LinearRelease(release_time=100.0)),
    ("afforestation", Afforestation()),
    ("permanent", PermanentRemoval()),
]


# ===================================================================== algebra
class TestScaledDefinite:
    """The one primitive everything else is built on."""

    @staticmethod
    def _reference(p, r, w, n=900):
        x, wq = np.polynomial.legendre.leggauss(n)
        t = 0.5 * w * (x + 1.0)
        return 0.5 * w * float(np.sum(wq * t ** p * np.exp(-r * t)))

    @pytest.mark.parametrize("p", [0, 1, 3, 8, 20, 40])
    @pytest.mark.parametrize(
        "r", [0.0, 1e-14, 1e-7, 1e-3, 0.02, 0.05, 0.3, 1.0, 2.0]
    )
    @pytest.mark.parametrize("w", [1.0, 10.0, 16.666, 50.0])
    def test_matches_quadrature(self, p, r, w):
        got = float(scaled_definite(p, r, w, 0.0)[0])
        ref = self._reference(p, r, w)
        assert np.isfinite(got)
        assert got == pytest.approx(ref, rel=1e-11)

    @pytest.mark.parametrize("p", [0, 1, 3, 8])
    @pytest.mark.parametrize("r", [-1e-12, -1e-4, -0.01, -0.05])
    @pytest.mark.parametrize("w", [1.0, 10.0, 16.666])
    def test_negative_rates(self, p, r, w):
        """Time reversal produces negative rates; they must be exact too."""
        got = float(scaled_definite(p, r, w, 0.0)[0])
        assert got == pytest.approx(self._reference(p, r, w), rel=1e-11)

    def test_zero_rate_limit(self):
        """The series branch must reproduce the elementary zero-rate answer."""
        for p in (0, 2, 7, 25):
            got = float(scaled_definite(p, 0.0, 12.0, 0.0)[0])
            assert got == pytest.approx(12.0 ** (p + 1) / (p + 1), rel=1e-14)

    def test_log_scale_is_a_factor(self):
        a = float(scaled_definite(3, 0.05, 10.0, 0.0)[0])
        b = float(scaled_definite(3, 0.05, 10.0, -4.0)[0])
        assert b == pytest.approx(a * np.exp(-4.0), rel=1e-14)

    def test_no_overflow_at_long_horizon(self):
        """The prefactor and the integral must never be formed separately."""
        out = scaled_definite(
            np.array([0, 5, 12]), np.array([-0.29, 0.29, 1e-11]), 100.0,
            np.array([-29.0, 0.0, 0.0]),
        )
        assert np.all(np.isfinite(out))

    def test_vectorised_matches_scalar(self):
        p = np.array([0, 3, 8, 20])
        r = np.array([0.05, -0.01, 1e-9, 0.3])
        ls = np.array([0.0, -1.0, -2.0, -3.0])
        vec = scaled_definite(p, r, 16.7, ls)
        one = [
            float(scaled_definite(int(a), float(b), 16.7, float(c))[0])
            for a, b, c in zip(p, r, ls)
        ]
        assert np.allclose(vec, one, rtol=1e-14)


class TestSignalAlgebra:
    def test_rebase_shifts_the_origin(self):
        es = ExpSum([(0.03, [1.0, 2.0, 0.5])])
        t = np.linspace(0.0, 30.0, 61)
        assert np.allclose(rebase(es, 7.0).eval(t), es.eval(t + 7.0), rtol=1e-13)

    def test_signal_product_is_pointwise(self):
        a = Signal.from_expsum(ExpSum.exponential(2.0, 0.05), 0.0)
        b = Signal.from_expsum(ExpSum.exponential(3.0, 0.02), 12.0)
        t = np.linspace(0.0, 60.0, 241)
        assert np.allclose(signal_product(a, b).eval(t), a.eval(t) * b.eval(t), atol=1e-14)

    def test_signal_product_rejects_atoms(self):
        with pytest.raises(ValueError):
            signal_product(Signal.delta(1.0), Signal.delta(1.0))

    @pytest.mark.parametrize("label,iv", FAMILIES)
    def test_stock_signal_matches_numeric_integral(self, label, iv):
        flux = iv.flux()
        t = np.linspace(0.0, 200.0, 801)
        assert np.allclose(stock_signal(flux).eval(t), -flux.definite(t), atol=1e-12)

    @pytest.mark.parametrize("label,iv", FAMILIES)
    def test_stock_is_non_negative(self, label, iv):
        t = np.linspace(0.0, 400.0, 1601)
        assert np.min(stock_signal(iv.flux()).eval(t)) > -1e-12


class TestPiecewise:
    def test_from_signal_reproduces_values(self):
        sig = stock_signal(Afforestation().flux())
        edges = block_edges(HORIZON, 10)
        pw = Piecewise.from_signal(sig, edges)
        t = np.linspace(0.5, HORIZON - 0.5, 300)
        assert np.allclose(pw.eval(t), sig.eval(t), rtol=1e-12)

    def test_product_is_pointwise(self):
        edges = block_edges(HORIZON, 8)
        a = Piecewise.from_signal(stock_signal(ExponentialRelease(tau=40.0).flux()), edges)
        b = Piecewise.from_signal(stock_signal(DelayedPulse(tau=70.0).flux()), edges)
        t = np.linspace(0.5, HORIZON - 0.5, 200)
        assert np.allclose(a.product(b).eval(t), a.eval(t) * b.eval(t), atol=1e-14)

    def test_refine_preserves_the_function(self):
        edges = block_edges(HORIZON, 5)
        pw = Piecewise.from_signal(stock_signal(Afforestation().flux()), edges)
        fine = pw.refine(block_edges(HORIZON, 20))
        t = np.linspace(0.5, HORIZON - 0.5, 200)
        assert np.allclose(fine.eval(t), pw.eval(t), rtol=1e-12)

    def test_mismatched_blocks_rejected(self):
        a = Piecewise.zero(block_edges(HORIZON, 4))
        b = Piecewise.zero(block_edges(HORIZON, 5))
        with pytest.raises(ValueError):
            a + b


class TestCoolingIdentity:
    """Cooling as a linear functional of the stock, against the reference route."""

    @pytest.mark.parametrize("label,iv", FAMILIES)
    @pytest.mark.parametrize("th", [50.0, 100.0, 500.0])
    def test_matches_deterministic_framework(self, cm, kern, label, iv, th):
        got = reverse_weighted_integral(stock_signal(iv.flux()), kern, th)
        assert got == pytest.approx(float(iv.cooling(cm, th)), rel=2e-12)

    def test_fixed_term_store_closed_form(self, cm, kern):
        """The plan's own Table 1 entry, which also fixes the sign."""
        tau, th = 40.0, 100.0
        got = reverse_weighted_integral(
            stock_signal(DelayedPulse(tau=tau).flux()), kern, th
        )
        psi = float(cm.iagtp("CO2", th)) - float(cm.iagtp("CO2", max(th - tau, 0.0)))
        assert got == pytest.approx(psi, rel=1e-12)
        assert got > 0.0


class TestTriangleIntegral:
    @pytest.mark.parametrize("label,iv", FAMILIES[:4])
    @pytest.mark.parametrize("n_block,th", [(1, 100.0), (10, 100.0), (10, 500.0)])
    def test_matches_quadrature(self, cm, kern, hazard, label, iv, n_block, th):
        stock = stock_signal(iv.flux())
        edges = merge_edges(block_edges(th, n_block), breakpoints(stock, horizon=th))
        a = Piecewise.from_signal(stock, edges)
        f = a.product(hazard.survival(edges))
        g = f.product(hazard.growth(edges, hazard.severity.m2))
        exact = triangle_integral(f, g, kern, th, edges=edges)
        quad = triangle_integral_quadrature(f, g, kern, th, edges=edges, n=240)
        assert exact == pytest.approx(quad, rel=5e-10)

    def test_vanishes_for_zero_integrand(self, kern):
        edges = block_edges(HORIZON, 6)
        z = Piecewise.zero(edges)
        f = Piecewise.from_signal(stock_signal(Afforestation().flux()), edges)
        assert triangle_integral(f, z, kern, HORIZON, edges=edges) == 0.0
        assert triangle_integral(z, f, kern, HORIZON, edges=edges) == 0.0

    def test_bilinear(self, cm, kern, hazard):
        edges = block_edges(HORIZON, 6)
        a = Piecewise.from_signal(stock_signal(ExponentialRelease(tau=50.0).flux()), edges)
        f = a.product(hazard.survival(edges))
        g = f.product(hazard.growth(edges, hazard.severity.m2))
        base = triangle_integral(f, g, kern, HORIZON, edges=edges)
        scaled = triangle_integral(f.scaled(3.0), g, kern, HORIZON, edges=edges)
        assert scaled == pytest.approx(3.0 * base, rel=1e-12)

    def test_block_count_does_not_change_a_constant_hazard(self, cm, kern):
        """With no climate trend the answer cannot depend on the blocking."""
        hz = FireHazard(lambda0=0.01, severity=BetaSeverity(0.6, 4.0))
        iv = ExponentialRelease(tau=50.0)
        vals = []
        for nb in (1, 4, 13):
            edges = merge_edges(
                block_edges(HORIZON, nb),
                breakpoints(stock_signal(iv.flux()), horizon=HORIZON),
            )
            a = Piecewise.from_signal(stock_signal(iv.flux()), edges)
            f = a.product(hz.survival(edges))
            g = f.product(hz.growth(edges, hz.severity.m2))
            vals.append(triangle_integral(f, g, kern, HORIZON, edges=edges))
        assert vals[1] == pytest.approx(vals[0], rel=1e-10)
        assert vals[2] == pytest.approx(vals[0], rel=1e-10)


# ===================================================================== hazard
class TestSeverity:
    @pytest.mark.parametrize(
        "sev",
        [
            BetaSeverity(0.6, 4.0),
            BetaSeverity(0.3, 12.0),
            TwoPointSeverity(0.3, 0.35),
            TruncatedParetoSeverity(1.5, 0.05),
            STAND_REPLACING,
        ],
    )
    def test_moments_match_sampler(self, sev):
        rng = np.random.default_rng(11)
        x = sev.sample(rng, 400000)
        assert np.all((x >= -1e-12) & (x <= 1.0 + 1e-12))
        assert float(x.mean()) == pytest.approx(sev.m1, abs=4e-3)
        assert float((x ** 2).mean()) == pytest.approx(sev.m2, abs=4e-3)

    @pytest.mark.parametrize(
        "sev",
        [BetaSeverity(0.6, 4.0), TwoPointSeverity(0.3, 0.35), TruncatedParetoSeverity()],
    )
    def test_moment_inequalities(self, sev):
        """A variable on the unit interval satisfies m1^2 <= m2 <= m1."""
        assert sev.m1 ** 2 <= sev.m2 + 1e-12
        assert sev.m2 <= sev.m1 + 1e-12
        assert sev.variance >= -1e-12

    def test_fat_tail_has_higher_second_moment(self):
        """Why the bound must be distribution free.

        The comparison is against a *concentrated* Beta of the same mean,
        which is what an analyst who assumed a well-behaved severity would
        fit. A diffuse Beta is not the right control: at concentration four the
        Beta is more dispersed than the truncated Pareto, so the test would
        pass or fail on the arbitrary choice of concentration rather than on
        the tail. Stating the comparison against the concentrated case makes
        the claim the one that matters, namely that assuming a tight severity
        distribution understates the second moment the guarantee depends on.
        """
        pareto = TruncatedParetoSeverity(alpha=1.5, floor=0.05)
        for concentration in (20.0, 40.0, 200.0):
            tight = BetaSeverity(pareto.m1, concentration)
            assert tight.m1 == pytest.approx(pareto.m1, rel=1e-12)
            assert pareto.m2 > tight.m2

    def test_second_moment_drives_the_bound_not_the_shape(self):
        """Two different laws with matched moments must give one guarantee."""
        pareto = TruncatedParetoSeverity(alpha=1.5, floor=0.05)
        # Concentration chosen so the Beta matches both moments of the Pareto.
        conc = (pareto.m1 - pareto.m2) / (pareto.m2 - pareto.m1 ** 2)
        matched = BetaSeverity(pareto.m1, conc)
        assert matched.m1 == pytest.approx(pareto.m1, rel=1e-12)
        assert matched.m2 == pytest.approx(pareto.m2, rel=1e-10)


class TestFireHazard:
    def test_present_day_rate_is_the_return_interval(self, hazard):
        assert float(np.asarray(hazard.intensity(0.0)).ravel()[0]) == pytest.approx(
            hazard.lambda0, rel=1e-13
        )

    def test_intensity_increases_with_climate(self, hazard):
        t = np.linspace(0.0, HORIZON, 50)
        assert np.all(np.diff(hazard.intensity(t)) > 0)

    def test_block_discretisation_converges(self, hazard):
        errs = [hazard.refinement_error(HORIZON, n) for n in (5, 10, 40)]
        assert errs[0] > errs[1] > errs[2]
        assert errs[1] < 1e-3

    def test_survival_is_monotone_and_bounded(self, hazard):
        edges = block_edges(HORIZON, 10)
        s = hazard.survival(edges).eval(np.linspace(0.0, HORIZON, 400))
        assert s[0] == pytest.approx(1.0, rel=1e-12)
        assert np.all(np.diff(s) <= 1e-14)
        assert np.all((s > 0.0) & (s <= 1.0 + 1e-12))

    def test_growth_starts_at_zero_and_rises(self, hazard):
        edges = block_edges(HORIZON, 10)
        g = hazard.growth(edges, hazard.severity.m2).eval(
            np.linspace(0.0, HORIZON, 400)
        )
        assert g[0] == pytest.approx(0.0, abs=1e-14)
        assert np.all(np.diff(g) >= -1e-14)

    def test_inert_hazard_never_burns(self):
        assert NO_FIRE.is_inert
        assert FireHazard(lambda0=0.02, exposure=0.0).is_inert

    def test_scaled_changes_only_the_rate(self, hazard):
        h2 = hazard.scaled(3.0)
        assert h2.lambda0 == pytest.approx(3.0 * hazard.lambda0)
        assert h2.beta_T == hazard.beta_T
        assert h2.severity is hazard.severity

    def test_cross_severity_exponent_recovers_the_marginal(self):
        sev = BetaSeverity(0.6, 4.0)
        assert cross_severity_exponent(sev, sev, 1.0) == pytest.approx(sev.m2, rel=1e-13)
        assert cross_severity_exponent(sev, sev, 0.0) == pytest.approx(
            sev.m1 ** 2, rel=1e-13
        )


# ================================================================== moments
class TestFireMoments:
    """The central claim: the closed forms match the simulated process."""

    @staticmethod
    def _simulate(iv, hz, cm, horizon, n=120000, seed=5, n_step=400):
        rng = np.random.default_rng(seed)
        edges = np.linspace(0.0, horizon, n_step + 1)
        mid = 0.5 * (edges[:-1] + edges[1:])
        expected = hz.intensity(mid) * np.diff(edges)
        weight = (
            stock_signal(iv.flux()).eval(mid)
            * cm.agtp_kernel("CO2").eval(horizon - mid)
            * np.diff(edges)
        )
        out = np.empty(n)
        for k in range(0, n, 20000):
            m = min(20000, n - k)
            counts = rng.poisson(expected, size=(m, n_step))
            phi = hz.severity.sample(rng, size=(m, n_step))
            factor = np.where(counts > 0, (1.0 - phi) ** counts, 1.0)
            out[k : k + m] = np.cumprod(factor, axis=1) @ weight
        return out

    @pytest.mark.parametrize(
        "iv",
        [
            ExponentialRelease(tau=60.0),
            Afforestation(tau_growth=28.0, tau_disturbance=140.0),
            DelayedPulse(tau=55.0),
        ],
    )
    def test_mean_and_variance_against_simulation(self, cm, hazard, iv):
        store = FireExposedStore(iv, hazard)
        mu = store.mean_cooling(cm, HORIZON)
        sd = np.sqrt(store.variance_cooling(cm, HORIZON))
        sample = self._simulate(iv, hazard, cm, HORIZON)
        assert mu == pytest.approx(float(sample.mean()), rel=6e-3)
        assert sd == pytest.approx(float(sample.std(ddof=1)), rel=1.5e-2)

    def test_mean_below_deterministic(self, cm, hazard):
        """Fire can only remove cooling, never add it."""
        for _, iv in FAMILIES[:4]:
            store = FireExposedStore(iv, hazard)
            assert store.mean_cooling(cm, HORIZON) < store.deterministic_cooling(
                cm, HORIZON
            )
            assert store.mean_loss_fraction(cm, HORIZON) > 0.0

    def test_inert_store_is_deterministic(self, cm):
        store = FireExposedStore(PermanentRemoval(), NO_FIRE)
        assert store.mean_cooling(cm, HORIZON) == pytest.approx(
            store.deterministic_cooling(cm, HORIZON), rel=1e-14
        )
        assert store.variance_cooling(cm, HORIZON) == 0.0

    def test_variance_is_non_monotone_in_the_hazard(self, cm, hazard):
        """Uncertainty peaks at intermediate hazard and vanishes at both ends.

        This is a property of the framework worth asserting rather than a
        quirk. At a vanishing hazard the store keeps its carbon with certainty
        and there is nothing to be uncertain about; at an overwhelming hazard
        it loses the carbon with certainty and there is again nothing to be
        uncertain about. The variance is therefore largest in between, near a
        cumulative hazard of order one, which is precisely the range the
        calibrated regimes occupy. A portfolio is hardest to guarantee not
        where fire is worst but where it is most uncertain.
        """
        iv = Afforestation()
        factors = np.array([1e-3, 0.05, 0.25, 0.5, 1.0, 2.0, 5.0, 20.0])
        v = np.array(
            [
                FireExposedStore(iv, hazard.scaled(f)).variance_cooling(cm, HORIZON)
                for f in factors
            ]
        )
        assert np.all(v >= 0.0)
        peak = int(np.argmax(v))
        assert 0 < peak < len(factors) - 1
        assert np.all(np.diff(v[: peak + 1]) > 0)
        assert np.all(np.diff(v[peak:]) < 0)
        # Both limits are negligible against the peak.
        assert v[0] < 1e-2 * v[peak]
        assert v[-1] < 1e-2 * v[peak]

    def test_variance_grows_with_hazard_in_the_calibrated_range(self, cm):
        """Below the peak, more fire means more uncertainty."""
        iv = Afforestation()
        weak = FireHazard(
            lambda0=1.0 / 400.0, severity=BetaSeverity(0.6, 4.0), region="test"
        )
        v = [
            FireExposedStore(iv, weak.scaled(f)).variance_cooling(cm, HORIZON)
            for f in (0.5, 1.0, 2.0)
        ]
        assert v[0] < v[1] < v[2]

    def test_mean_falls_with_hazard(self, cm, hazard):
        iv = Afforestation()
        m = [
            FireExposedStore(iv, hazard.scaled(f)).mean_cooling(cm, HORIZON)
            for f in (0.5, 1.0, 2.0)
        ]
        assert m[0] > m[1] > m[2]

    def test_exposure_scales_the_loss(self, cm, hazard):
        iv = ExponentialRelease(tau=50.0)
        full = FireExposedStore(iv, hazard).mean_loss_fraction(cm, HORIZON)
        part = FireExposedStore(
            iv, hazard.replace(exposure=0.25)
        ).mean_loss_fraction(cm, HORIZON)
        assert 0.0 < part < full

    def test_deterministic_limit(self, cm, hazard):
        iv = Afforestation()
        det = float(iv.cooling(cm, HORIZON))
        store = FireExposedStore(iv, hazard.replace(lambda0=1e-9))
        assert store.mean_cooling(cm, HORIZON) == pytest.approx(det, rel=1e-6)
        assert store.variance_cooling(cm, HORIZON) / det ** 2 < 1e-6


class TestCovariance:
    @pytest.fixture(scope="class")
    def moments(self, cm):
        return cooling_moments(
            build_stores(WESTERN_US), cm, HORIZON, n_block=6, weather_spread=0.3,
            n_weather=5,
        )

    def test_positive_semidefinite(self, moments):
        vals = np.linalg.eigvalsh(moments.covariance)
        assert np.min(vals) >= -1e-12 * max(float(np.max(vals)), 1e-300)

    def test_symmetric(self, moments):
        assert np.allclose(moments.covariance, moments.covariance.T, rtol=1e-12)

    def test_correlations_within_unit_interval(self, moments):
        c = moments.correlation()
        assert np.all(np.abs(c) <= 1.0 + 1e-9)

    def test_decomposition_sums_to_total(self, moments):
        total = moments.idiosyncratic + moments.common
        assert np.allclose(moments.covariance, total, atol=1e-12 * np.max(np.abs(total)))

    def test_common_component_is_not_diversifiable(self, moments):
        """The floor exists: the common part does not fall with portfolio size."""
        assert np.any(np.diag(moments.common) > 0.0)
        assert 0.0 < moments.diversifiable_share < 1.0

    def test_protected_measures_carry_no_variance(self, cm, moments):
        stores = build_stores(WESTERN_US)
        inert = np.array([s.hazard.is_inert for s in stores])
        assert np.all(np.diag(moments.covariance)[inert] < 1e-40)

    def test_no_weather_spread_removes_the_common_part(self, cm):
        m = cooling_moments(
            build_stores(WESTERN_US), cm, HORIZON, n_block=6, weather_spread=0.0
        )
        assert np.allclose(m.common, 0.0)


# ================================================================== portfolio
@pytest.fixture(scope="module")
def portfolio(cm):
    return FirePortfolio(
        cm,
        build_stores(WESTERN_US),
        WESTERN_US.pathway(),
        horizon=WESTERN_US.horizon,
        n_block=6,
        weather_spread=WESTERN_US.weather_spread,
        n_weather=5,
    )


class TestRobustPortfolio:
    def test_kappa_round_trip(self):
        for eta in (0.5, 0.75, 0.9, 0.95, 0.99):
            assert confidence_from_kappa(kappa(eta)) == pytest.approx(eta, rel=1e-12)

    def test_kappa_at_half_is_one(self):
        assert kappa(0.5) == pytest.approx(1.0, rel=1e-14)

    def test_cost_rises_with_confidence(self, portfolio):
        costs = [portfolio.solve(e).cost for e in (0.5, 0.75, 0.9, 0.95)]
        assert all(np.isfinite(costs))
        assert all(b > a for a, b in zip(costs, costs[1:]))

    def test_premium_is_positive(self, portfolio):
        det = portfolio.deterministic_solution()
        assert portfolio.solve(0.5).cost > det.cost

    def test_constraint_is_tight_at_the_optimum(self, portfolio):
        for eta in (0.6, 0.9):
            res = portfolio.solve(eta)
            assert res.feasible
            assert abs(res.margin) < 1e-7 * abs(res.required_cooling)

    def test_optimality_conditions(self, portfolio):
        for eta in (0.75, 0.9):
            res = portfolio.solve(eta)
            rep = portfolio.kkt_report(res.alpha, confidence=eta)
            assert rep["relative_spread"] < 2e-3
            assert rep["zero_condition_satisfied"]

    def test_capacity_respected(self, portfolio):
        res = portfolio.solve(0.9)
        caps = portfolio.caps()
        assert np.all(res.alpha <= caps * (1.0 + 1e-9))
        assert np.all(res.alpha >= -1e-12)

    def test_shift_towards_protected_measures(self, portfolio):
        """The central qualitative claim of the paper."""
        m = portfolio.moments()
        protected = np.array([s.hazard.is_inert for s in portfolio.stores])
        shares = []
        for eta in (0.5, 0.9, 0.95):
            res = portfolio.solve(eta)
            cooling = res.alpha * m.mean
            shares.append(float(cooling[protected].sum() / cooling.sum()))
        assert shares[0] < shares[1] < shares[2]

    def test_infeasible_reported_not_hidden(self, cm):
        """Beyond the ceiling the solver must not report a cheap portfolio."""
        stores = [s for s in build_stores(WESTERN_US) if not s.hazard.is_inert]
        fp = FirePortfolio(
            cm, stores, WESTERN_US.pathway(), WESTERN_US.horizon, n_block=6,
            weather_spread=WESTERN_US.weather_spread, n_weather=5,
        )
        res = fp.solve(0.999999)
        assert not res.feasible
        assert not np.isfinite(res.cost)
        assert np.all(np.isnan(res.alpha))

    def test_ceiling_is_finite_without_a_fireproof_option(self, portfolio):
        rep = portfolio.ceiling_report()
        assert rep["fireproof_available"]
        assert rep["confidence_max_exposed_only"] < 1.0
        assert rep["confidence_max_exposed_only"] > 0.5

    def test_deterministic_limit_of_the_portfolio(self, cm):
        quiet = [
            FireExposedStore(s.intervention, s.hazard.replace(lambda0=1e-10), s.label)
            for s in build_stores(WESTERN_US)
        ]
        fp = FirePortfolio(
            cm, quiet, WESTERN_US.pathway(), WESTERN_US.horizon, n_block=5,
            weather_spread=0.0,
        )
        ref = fp.deterministic.analytic_power_cost_solution()
        assert ref is not None
        assert fp.solve(0.9).cost == pytest.approx(float(ref.cost), rel=3e-3)


# ====================================================================== spike
class TestSpike:
    def test_trajectory_is_a_shifted_kernel(self, cm):
        sig = spike_trajectory(cm, 1.0e12, 30.0)
        t = np.linspace(0.0, 100.0, 401)
        expected = 1.0e12 * cm.agtp_kernel("CO2").eval(t - 30.0) * (t >= 30.0)
        assert np.allclose(sig.eval(t), expected, rtol=1e-12)

    def test_no_warming_before_the_fire(self, cm):
        sig = spike_trajectory(cm, 1.0e12, 40.0)
        assert np.all(np.abs(sig.eval(np.linspace(0.0, 39.9, 50))) < 1e-30)

    def test_amplitude_scales_linearly(self, cm):
        a1, lag1 = spike_amplitude(cm, 1.0e12)
        a2, lag2 = spike_amplitude(cm, 3.0e12)
        assert a2 == pytest.approx(3.0 * a1, rel=1e-12)
        assert lag1 == pytest.approx(lag2)

    def test_worst_case_spike_raises_the_peak(self, portfolio):
        res = portfolio.solve(0.9)
        spike = worst_case_spike(portfolio, res.alpha, n_grid=401)
        assert spike.peak_with_spike >= spike.peak_baseline
        assert spike.released > 0.0
        assert 0.0 <= spike.fire_time <= portfolio.horizon


# ================================================================ calibration
class TestCalibration:
    def test_lambda_from_return_interval(self):
        assert lambda_from_return_interval(70.0) == pytest.approx(1.0 / 70.0)
        with pytest.raises(ValueError):
            lambda_from_return_interval(0.0)

    def test_beta_inversion_reproduces_the_target(self):
        beta = beta_from_intensification(3.3, 1.4, 4.0, beta_D=0.25, drought_end=1.2)
        realised = np.exp(beta * (4.0 - 1.4) + 0.25 * 1.2)
        assert realised == pytest.approx(3.3, rel=1e-12)

    def test_spread_from_extreme_is_increasing(self):
        s = [spread_from_extreme(r, 100.0) for r in (1.5, 3.0, 6.0)]
        assert all(b > a for a, b in zip(s, s[1:]))

    def test_spread_reproduces_the_quantile(self):
        from scipy.stats import norm

        r, p = 6.0, 100.0
        sig = spread_from_extreme(r, p)
        z = norm.ppf(1.0 - 1.0 / p)
        assert np.exp(sig * z - 0.5 * sig ** 2) == pytest.approx(r, rel=1e-9)

    def test_spread_is_zero_for_no_anomaly(self):
        assert spread_from_extreme(1.0, 100.0) == 0.0

    def test_severity_from_classes(self):
        m1, m2 = severity_from_class_fractions(0.32, 0.34, 0.34)
        assert 0.0 < m1 < 1.0
        assert m1 ** 2 <= m2 <= m1

    def test_stand_replacing_dominates_mixed(self):
        high, _ = severity_from_class_fractions(0.82, 0.13, 0.05)
        mixed, _ = severity_from_class_fractions(0.32, 0.34, 0.34)
        assert high > mixed


class TestRegions:
    @pytest.mark.parametrize("region", [WESTERN_US, BOREAL_CANADA])
    def test_intensification_target_met(self, region):
        hz = region.hazard_template()
        assert hz.intensification(region.horizon) == pytest.approx(
            region.intensification, rel=1e-9
        )

    @pytest.mark.parametrize("region", [WESTERN_US, BOREAL_CANADA])
    def test_severity_moments_recovered(self, region):
        m1, m2 = region._severity_moments
        assert region.severity.m1 == pytest.approx(m1, rel=1e-9)
        assert region.severity.m2 == pytest.approx(m2, rel=1e-6)

    @pytest.mark.parametrize("region", [WESTERN_US, BOREAL_CANADA])
    def test_capacity_covers_the_pathway(self, cm, region):
        """Composition must be decided by cost and durability, not scarcity."""
        stores = build_stores(region)
        fp = FirePortfolio(
            cm, stores, region.pathway(), region.horizon, n_block=5,
            weather_spread=0.0,
        )
        caps = np.array([s.max_scale for s in stores], dtype=float)
        coverage = float(fp.moments().mean @ caps) / fp.required_cooling()
        assert coverage > 1.2

    def test_boreal_severity_exceeds_western(self):
        assert BOREAL_CANADA.severity.m1 > WESTERN_US.severity.m1

    def test_western_hazard_exceeds_boreal(self):
        assert (
            WESTERN_US.hazard_template().lambda0
            > BOREAL_CANADA.hazard_template().lambda0
        )

    def test_boreal_variability_exceeds_western(self):
        assert BOREAL_CANADA.weather_spread > WESTERN_US.weather_spread

    def test_measures_are_temporary_or_permanent_as_declared(self):
        for s in build_stores(WESTERN_US):
            if isinstance(s.intervention, PermanentRemoval):
                assert s.hazard.is_inert
            else:
                assert not s.hazard.is_inert
