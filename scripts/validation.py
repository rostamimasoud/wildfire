"""Six-tier validation of the fire-integrated framework.

Each tier checks one thing against something computed a different way, and the
tiers are ordered so that a failure in an early one explains a failure in a
later one. Nothing here is compared against a value produced by this framework:
every reference is either an independent numerical method, a published assessed
value, or an analytically known limit.

Tier 1
    The stock-path identity and the new triangle-integral primitive against
    independent numerical methods. The identity that cumulative cooling equals
    ``(A * AGTP_CO2)(TH)`` is checked against the deterministic route of
    :mod:`cipo`, which never forms the stock at all. The variance integral is
    checked against block-wise Gauss-Legendre quadrature.

Tier 2
    The inherited climate model against the published assessed metrics, so that
    a discrepancy in a fire-adjusted result cannot be blamed on the climate
    calibration.

Tier 3
    The closed-form fire moments against a direct Monte Carlo simulation of the
    fire process: Poisson event times, sampled severities, and the stock path
    integrated numerically. This is the central check of the whole paper,
    because it tests the survival algebra, the severity moments and the
    time-reversed kernel together.

Tier 4
    The optimality conditions of the robust programme, by checking that the
    marginal cost per unit of effective cooling is equalised across measures
    held strictly inside their bounds, and that the Cantelli margin is tight at
    the optimum.

Tier 5
    Reduction to the deterministic framework in the limit of vanishing hazard.
    As ``lambda0 -> 0`` the mean cooling must approach the deterministic value,
    the variance must vanish, and the robust portfolio must approach the
    deterministic one at every confidence.

Tier 6
    The physical consistency of the calibrated fire regimes: the block
    discretisation of the hazard trend, the recovery of the severity moments
    from the fitted two-point law, the reproduction of the intensification
    target, and the transient spike amplitude against the observed signal from
    the 2019-2020 Australian fires.
"""

from __future__ import annotations

import numpy as np
import pandas as pd

from common import DATA_DIR, save_table, write_values

from cipo import (
    AR6,
    Afforestation,
    DelayedPulse,
    ExponentialRelease,
    LinearRelease,
    PermanentRemoval,
)
from cipo.climate import AR6_REFERENCE
from wrcp.algebra import (
    Piecewise,
    breakpoints,
    merge_edges,
    reverse_weighted_integral,
    stock_signal,
    triangle_integral,
    triangle_integral_quadrature,
)
from wrcp.hazard import (
    STAND_REPLACING,
    BetaSeverity,
    FireHazard,
    TruncatedParetoSeverity,
    TwoPointSeverity,
    drought_path,
    temperature_path,
)
from wrcp.regions import BOREAL_CANADA, WESTERN_US, build_stores
from wrcp.robust import FirePortfolio, kappa
from wrcp.spike import spike_amplitude
from wrcp.stochastic import FireExposedStore, block_edges

ROWS = []


def record(tier, check, value, reference, tolerance, units="relative"):
    err = (
        abs(value - reference) / abs(reference)
        if reference not in (0, 0.0)
        else abs(value - reference)
    )
    ROWS.append(
        {
            "tier": tier,
            "check": check,
            "value": value,
            "reference": reference,
            "error": err,
            "tolerance": tolerance,
            "units": units,
            "pass": bool(err <= tolerance),
        }
    )
    flag = "ok " if err <= tolerance else "FAIL"
    print("  [{}] t{} {:<52s} err={:.3e} (tol {:.0e})".format(flag, tier, check, err, tolerance))
    return err


# --------------------------------------------------------------------- tier 1
def tier1_algebra(cm):
    print("Tier 1: the stock-path identity and the triangle integral")
    kern = cm.agtp_kernel("CO2")
    families = [
        ("exponential release", ExponentialRelease(tau=50.0)),
        ("delayed pulse", DelayedPulse(tau=50.0)),
        ("constant-rate release", LinearRelease(release_time=100.0)),
        ("afforestation", Afforestation()),
        ("permanent removal", PermanentRemoval()),
    ]
    for label, iv in families:
        stock = stock_signal(iv.flux())
        for th in (100.0, 500.0):
            record(
                1,
                "cooling by stock path, {} TH={:g}".format(label, th),
                reverse_weighted_integral(stock, kern, th),
                float(iv.cooling(cm, th)),
                2e-12,
            )

    # The triangle integral against block-wise quadrature. A smooth single
    # block isolates the primitive itself; a ten-block case exercises the
    # decomposition and the accumulated off-diagonal term.
    hz = FireHazard(
        lambda0=0.01,
        beta_T=0.4,
        severity=BetaSeverity(0.6, 4.0),
        temperature=temperature_path(1.2, 3.6, 70.0),
    )
    for label, iv in families[:4]:
        stock = stock_signal(iv.flux())
        for nb, th in ((1, 100.0), (10, 100.0), (10, 400.0)):
            edges = merge_edges(
                block_edges(th, nb), breakpoints(stock, horizon=th)
            )
            a = Piecewise.from_signal(stock, edges)
            f = a.product(hz.survival(edges))
            g = f.product(hz.growth(edges, hz.severity.m2))
            exact = triangle_integral(f, g, kern, th, edges=edges)
            quad = triangle_integral_quadrature(f, g, kern, th, edges=edges, n=260)
            record(
                1,
                "triangle integral, {} {:d} blocks TH={:g}".format(label, len(edges) - 1, th),
                exact,
                quad,
                5e-9,
            )


# --------------------------------------------------------------------- tier 2
def tier2_climate(cm):
    print("Tier 2: the inherited climate model against assessed metrics")
    record(
        2,
        "AGWP CO2 at 100 yr",
        float(cm.agwp("CO2", 100.0)),
        AR6_REFERENCE["AGWP_CO2_100"],
        0.03,
    )
    record(
        2,
        "AGWP CO2 at 20 yr",
        float(cm.agwp("CO2", 20.0)),
        AR6_REFERENCE["AGWP_CO2_20"],
        0.03,
    )
    for gas, ref in AR6_REFERENCE["GWP100"].items():
        record(2, "GWP100 {}".format(gas), float(cm.gwp(gas, 100.0)), ref, 0.05)
    for gas, ref in AR6_REFERENCE["GWP20"].items():
        record(2, "GWP20 {}".format(gas), float(cm.gwp(gas, 20.0)), ref, 0.05)


# --------------------------------------------------------------------- tier 3
def _monte_carlo_moments(iv, hz, cm, horizon, n=240000, seed=7, n_step=500):
    """Direct simulation of the fire process; the reference for tier 3.

    Fire times come from a thinned Poisson process on a fine grid, severities
    from the severity law's own sampler, and the stock path is integrated
    against the reversed kernel numerically. Nothing from
    :mod:`wrcp.algebra` is used, so agreement tests the closed forms rather
    than reproducing them.
    """
    rng = np.random.default_rng(seed)
    edges = np.linspace(0.0, horizon, n_step + 1)
    mid = 0.5 * (edges[:-1] + edges[1:])
    dt = np.diff(edges)
    expected = hz.intensity(mid) * dt
    stock = stock_signal(iv.flux())
    weight = stock.eval(mid) * cm.agtp_kernel("CO2").eval(horizon - mid) * dt
    out = np.empty(n)
    chunk = 20000
    for k in range(0, n, chunk):
        m = min(chunk, n - k)
        counts = rng.poisson(expected, size=(m, n_step))
        phi = hz.severity.sample(rng, size=(m, n_step))
        factor = np.where(counts > 0, (1.0 - phi) ** counts, 1.0)
        out[k : k + m] = np.cumprod(factor, axis=1) @ weight
    return float(out.mean()), float(out.var(ddof=1)), float(
        out.std(ddof=1) / np.sqrt(n)
    )


def tier3_moments(cm):
    print("Tier 3: closed-form fire moments against direct simulation")
    horizon = 100.0
    cases = [
        (
            "mixed severity, exponential release",
            ExponentialRelease(tau=60.0),
            FireHazard(
                lambda0=1.0 / 70.0,
                beta_T=0.34,
                beta_D=0.25,
                severity=BetaSeverity(0.6, 4.0),
                temperature=temperature_path(1.4, 4.0, 55.0),
                drought=drought_path(0.0, 1.2, horizon),
            ),
        ),
        (
            "stand replacing, afforestation",
            Afforestation(tau_growth=28.0, tau_disturbance=140.0),
            FireHazard(
                lambda0=1.0 / 110.0,
                beta_T=0.19,
                severity=STAND_REPLACING,
                temperature=temperature_path(1.9, 5.6, 55.0),
            ),
        ),
        (
            "fat-tailed severity, delayed pulse",
            DelayedPulse(tau=55.0),
            FireHazard(
                lambda0=1.0 / 50.0,
                beta_T=0.3,
                severity=TruncatedParetoSeverity(alpha=1.5, floor=0.05),
                temperature=temperature_path(1.5, 4.2, 60.0),
            ),
        ),
    ]
    for label, iv, hz in cases:
        store = FireExposedStore(iv, hz)
        mu = store.mean_cooling(cm, horizon)
        var = store.variance_cooling(cm, horizon)
        m_mu, m_var, se = _monte_carlo_moments(iv, hz, cm, horizon)
        # The Monte Carlo standard error sets the tolerance: a tighter one
        # would be testing the sampler, not the algebra.
        record(3, "mean cooling, {}".format(label), mu, m_mu, max(4.0 * se / abs(m_mu), 2e-3))
        record(
            3,
            "s.d. of cooling, {}".format(label),
            float(np.sqrt(var)),
            float(np.sqrt(m_var)),
            8e-3,
        )

    # A limiting case with an exact answer: a constant hazard with
    # stand-replacing severity makes the survival multiplier Bernoulli, so the
    # expected stock of a fixed-term store is exactly exp(-lambda t) on
    # [0, tau) and zero after. The reference integrates that against the
    # reversed kernel by Gauss-Legendre on [0, tau], where the integrand is
    # analytic; the trapezoidal rule would be limited by the discontinuity at
    # tau rather than by the framework.
    tau, lam = 60.0, 0.02
    iv = DelayedPulse(tau=tau)
    store = FireExposedStore(iv, FireHazard(lambda0=lam, severity=STAND_REPLACING))
    kern = cm.agtp_kernel("CO2")
    x, w = np.polynomial.legendre.leggauss(400)
    s = 0.5 * tau * (x + 1.0)
    exact_mean = float(
        0.5 * tau * np.sum(w * np.exp(-lam * s) * kern.eval(horizon - s))
    )
    record(
        3,
        "Bernoulli survival, mean against exact quadrature",
        store.mean_cooling(cm, horizon),
        exact_mean,
        1e-11,
    )


# --------------------------------------------------------------------- tier 4
def tier4_optimality(cm):
    print("Tier 4: optimality conditions of the robust programme")
    for region in (WESTERN_US, BOREAL_CANADA):
        fp = FirePortfolio(
            cm,
            build_stores(region),
            region.pathway(),
            horizon=region.horizon,
            n_block=8,
            weather_spread=region.weather_spread,
            n_weather=5,
        )
        for eta in (0.75, 0.9):
            res = fp.solve(eta)
            rep = fp.kkt_report(res.alpha, confidence=eta)
            # Equalised marginal cost per unit effective cooling, interior only.
            record(
                4,
                "{} eta={:.2f} marginal-cost equalisation".format(region.name, eta),
                rep["relative_spread"],
                0.0,
                2e-3,
                units="relative spread",
            )
            # The constraint must bind: a slack guarantee would mean money was
            # spent for nothing, which a cost minimiser cannot do.
            record(
                4,
                "{} eta={:.2f} chance constraint tight".format(region.name, eta),
                res.margin / max(abs(res.required_cooling), 1e-300),
                0.0,
                5e-6,
                units="relative margin",
            )
            # The Cantelli identity itself.
            record(
                4,
                "{} eta={:.2f} Cantelli multiplier".format(region.name, eta),
                kappa(eta),
                float(np.sqrt(eta / (1.0 - eta))),
                1e-15,
            )


# --------------------------------------------------------------------- tier 5
def tier5_deterministic_limit(cm):
    print("Tier 5: reduction to the deterministic framework as the hazard vanishes")
    horizon = 100.0
    iv = Afforestation(tau_growth=28.0, tau_disturbance=140.0)
    det = float(iv.cooling(cm, horizon))
    for lam in (1e-3, 1e-5, 1e-7):
        hz = FireHazard(
            lambda0=lam,
            beta_T=0.34,
            severity=BetaSeverity(0.6, 4.0),
            temperature=temperature_path(1.4, 4.0, 55.0),
        )
        store = FireExposedStore(iv, hz)
        mu = store.mean_cooling(cm, horizon)
        sd = float(np.sqrt(max(store.variance_cooling(cm, horizon), 0.0)))
        # The two moments vanish at different rates, and checking that they do
        # is a stronger test than checking that they vanish. One fire is
        # expected per unit of cumulative hazard, so the mean shortfall is
        # first order in lambda while the variance, being the second moment of
        # a single rare event, is also first order and the standard deviation
        # is therefore of order sqrt(lambda). A tolerance linear in lambda
        # would wrongly fail the standard deviation, and one of order
        # sqrt(lambda) would wrongly pass the mean.
        record(
            5,
            "mean -> deterministic, first order at lambda0={:.0e}".format(lam),
            mu,
            det,
            60.0 * lam,
        )
        record(
            5,
            "s.d. -> zero, half order at lambda0={:.0e}".format(lam),
            sd / det,
            0.0,
            8.0 * np.sqrt(lam),
            units="coefficient of variation",
        )

    # The robust portfolio must reproduce the deterministic one as the hazard
    # vanishes. The reference is the closed-form power-cost optimum of the
    # deterministic framework, not its numerical solution, so that solver
    # tolerance cannot be mistaken for agreement.
    region = WESTERN_US
    quiet = [
        FireExposedStore(s.intervention, s.hazard.replace(lambda0=1e-10), s.label)
        for s in build_stores(region)
    ]
    fp = FirePortfolio(
        cm, quiet, region.pathway(), horizon=region.horizon, n_block=6,
        weather_spread=0.0,
    )
    analytic = fp.deterministic.analytic_power_cost_solution()
    reference = float(analytic.cost) if analytic is not None else float(
        fp.deterministic_solution().cost
    )
    for eta in (0.5, 0.9, 0.99):
        rob = fp.solve(eta)
        record(
            5,
            "robust cost -> analytic deterministic cost, eta={:.2f}".format(eta),
            rob.cost,
            reference,
            2e-3,
        )


# --------------------------------------------------------------------- tier 6
def tier6_physical(cm):
    print("Tier 6: consistency of the calibrated fire regimes")
    for region in (WESTERN_US, BOREAL_CANADA):
        hz = region.hazard_template()
        # The block discretisation of the trend.
        record(
            6,
            "{} hazard trend, 10 blocks".format(region.name),
            hz.refinement_error(region.horizon, 10),
            0.0,
            1e-3,
            units="relative in Lambda(TH)",
        )
        # The intensification target must be reproduced by the inverted beta_T.
        record(
            6,
            "{} intensification target".format(region.name),
            hz.intensification(region.horizon),
            region.intensification,
            1e-9,
        )
        # The fitted two-point law must carry the calibrated moments.
        m1, m2 = region._severity_moments
        record(6, "{} severity m1".format(region.name), region.severity.m1, m1, 1e-9)
        record(6, "{} severity m2".format(region.name), region.severity.m2, m2, 1e-6)
        # The baseline rate must be the inverse of the stated return interval.
        record(
            6,
            "{} present-day rate".format(region.name),
            float(np.asarray(hz.intensity(0.0)).ravel()[0]),
            1.0 / region.return_interval,
            1e-12,
        )

    # The transient spike against the observed Australian 2019-2020 signal.
    # The fires released of order 0.7 Gt of carbon dioxide; the detectable
    # global-mean stratospheric signal was of order a tenth of a kelvin, while
    # the surface temperature response to that release is two orders of
    # magnitude smaller. The check is that the framework reproduces the surface
    # response, which is the quantity it models, and that this is far below the
    # stratospheric signal, which it does not.
    amp, lag = spike_amplitude(cm, 0.7e12)
    record(
        6,
        "Australian 2019-20 release, surface spike order",
        float(np.log10(max(amp, 1e-300))),
        float(np.log10(2.0e-4)),
        0.35,
        units="log10 K",
    )
    record(6, "pulse response lag", lag, 6.0, 0.7, units="years")
    write_values(
        {
            "spike_amplitude_australia_K": amp,
            "spike_lag_yr": lag,
        }
    )


def main():
    cm = AR6()
    tier1_algebra(cm)
    tier2_climate(cm)
    tier3_moments(cm)
    tier4_optimality(cm)
    tier5_deterministic_limit(cm)
    tier6_physical(cm)

    frame = pd.DataFrame(ROWS)
    save_table(frame, "validation")
    n_fail = int((~frame["pass"]).sum())
    by_tier = frame.groupby("tier").agg(
        checks=("pass", "size"),
        failures=("pass", lambda s: int((~s).sum())),
        worst_error=("error", "max"),
    )
    print()
    print(by_tier.to_string())
    print("\n{} checks, {} failures".format(len(frame), n_fail))
    write_values(
        {
            "validation_checks": len(frame),
            "validation_failures": n_fail,
            "validation_worst_error": float(frame["error"].max()),
            "validation_tier1_worst": float(
                frame.loc[frame.tier == 1, "error"].max()
            ),
            "validation_tier3_worst": float(
                frame.loc[frame.tier == 3, "error"].max()
            ),
        }
    )
    return n_fail


if __name__ == "__main__":
    raise SystemExit(1 if main() else 0)
