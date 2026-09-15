"""Figure 1: what fire does to a temporary store.

Four panels establish the mechanism before any optimisation appears.

a   The hazard. Fire intensity rises with the regional climate trajectory, and
    the survival probability of a store falls accordingly. Both case-study
    regimes are shown, so the reader sees that the western United States has
    the higher mean hazard while boreal Canada has the more severe fire.

b   The stock. The deterministic stock of an afforestation project, the mean
    stock under fire, and the spread of the survival multiplier. The gap
    between the first two curves is the over-credit that deterministic
    accounting takes.

c   The credit. Cumulative cooling as a function of the horizon, with and
    without fire, for a fully exposed and a largely protected measure. The
    non-monotone shape is inherited from the deterministic framework; fire
    lowers the curve and, more importantly, widens it.

d   The distribution. The delivered cooling of an exposed store is not
    symmetric about its mean: most realisations lose a little, a few lose
    nearly everything. That is the shape the Chebyshev bound is designed to
    tolerate without being told about it.
"""

from __future__ import annotations

import numpy as np

from common import (
    COLOURS,
    DOUBLE,
    band,
    configure,
    halo,
    light_grid,
    panel_label,
    save,
    spine_style,
    write_values,
)

from cipo import AR6, Afforestation, ExponentialRelease
from wrcp.algebra import stock_signal
from wrcp.hazard import FireHazard
from wrcp.regions import BOREAL_CANADA, EXPOSURE, WESTERN_US
from wrcp.stochastic import FireExposedStore

HORIZON = 100.0


def _survival_samples(hz, iv, cm, horizon, n=40000, seed=3, n_step=400):
    """Realised delivered cooling under simulated fire, for panel d."""
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
    for k in range(0, n, 10000):
        m = min(10000, n - k)
        counts = rng.poisson(expected, size=(m, n_step))
        phi = hz.severity.sample(rng, size=(m, n_step))
        factor = np.where(counts > 0, (1.0 - phi) ** counts, 1.0)
        out[k : k + m] = np.cumprod(factor, axis=1) @ weight
    return out


def main():
    configure()
    import matplotlib.pyplot as plt

    cm = AR6()
    fig, axes = plt.subplots(2, 2, figsize=(DOUBLE, 0.62 * DOUBLE))
    (ax_a, ax_b), (ax_c, ax_d) = axes
    t = np.linspace(0.0, HORIZON, 1201)

    # ---------------------------------------------------------------- panel a
    for region in (WESTERN_US, BOREAL_CANADA):
        hz = region.hazard_template()
        colour = COLOURS[region.name]
        ax_a.plot(
            t,
            hz.intensity(t) * 1000.0,
            color=colour,
            path_effects=halo(),
            label="{}, {:.0f} yr interval".format(
                region.label, region.return_interval
            ),
        )
    ax_a.set_xlabel("year from present")
    ax_a.set_ylabel("fire intensity (per thousand yr$^{-1}$)")
    ax_a.set_xlim(0, HORIZON)
    ax_a.set_ylim(bottom=0)
    ax_a.legend(loc="upper left")
    light_grid(ax_a, "y")
    spine_style(ax_a)
    panel_label(ax_a, "a")

    ax_a2 = ax_a.twinx()
    for region in (WESTERN_US, BOREAL_CANADA):
        hz = region.hazard_template()
        edges = np.linspace(0.0, HORIZON, 401)
        cum = hz.cumulative(edges)
        ax_a2.plot(
            edges,
            np.exp(-cum),
            color=COLOURS[region.name],
            ls=(0, (3, 1.6)),
            lw=0.9,
        )
    ax_a2.set_ylabel("probability of no fire", color="#555555")
    ax_a2.set_ylim(0, 1.02)
    ax_a2.tick_params(axis="y", colors="#555555")
    ax_a2.spines["top"].set_visible(False)
    ax_a2.text(
        0.97,
        0.30,
        "dashed: survival",
        transform=ax_a2.transAxes,
        ha="right",
        fontsize=6.0,
        color="#555555",
    )

    # ---------------------------------------------------------------- panel b
    region = WESTERN_US
    iv = Afforestation(tau_growth=28.0, tau_disturbance=140.0)
    hz = region.hazard_template(EXPOSURE["afforestation"])
    store = FireExposedStore(iv, hz)
    edges = store.edges(HORIZON, 40)
    det_stock = stock_signal(iv.flux()).eval(t)
    mean_stock = store.mean_stock_weight(edges).eval(t)

    ax_b.plot(
        t,
        det_stock,
        color=COLOURS["deterministic"],
        ls=(0, (4, 1.5)),
        label="deterministic stock",
    )
    ax_b.plot(
        t,
        mean_stock,
        color=COLOURS["afforestation"],
        path_effects=halo(),
        label="mean stock under fire",
    )
    # The spread of the survival multiplier, as a one-standard-deviation band
    # on the stock. Second moment of the multiplier from the same algebra.
    cum = np.interp(t, edges, hz.cumulative(edges))
    m1, m2 = hz.severity.m1, hz.severity.m2
    var_mult = np.exp(-(2.0 * m1 - m2) * cum) - np.exp(-2.0 * m1 * cum)
    sd_stock = det_stock * np.sqrt(np.clip(var_mult, 0.0, None))
    # The multiplier cannot exceed one, so the stock cannot exceed its
    # deterministic value. A symmetric band around a skewed mean would breach
    # that bound, so the band is clipped to it and to zero; the clipping is
    # itself informative, because it shows how far the distribution is from
    # symmetric.
    band(
        ax_b,
        t,
        np.clip(mean_stock - sd_stock, 0.0, None),
        np.minimum(mean_stock + sd_stock, det_stock),
        COLOURS["afforestation"],
        alpha=0.22,
    )
    ax_b.set_xlabel("year from establishment")
    ax_b.set_ylabel("stored stock (per unit at peak)")
    ax_b.set_xlim(0, HORIZON)
    ax_b.set_ylim(0, 1.05)
    ax_b.legend(loc="upper right")
    light_grid(ax_b, "y")
    spine_style(ax_b)
    panel_label(ax_b, "b")
    ax_b.annotate(
        "over-credit",
        xy=(62.0, 0.5 * (det_stock[t.searchsorted(62.0)] + mean_stock[t.searchsorted(62.0)])),
        xytext=(70.0, 0.80),
        fontsize=6.2,
        color=COLOURS["fire"],
        arrowprops=dict(arrowstyle="-", lw=0.5, color=COLOURS["fire"]),
    )

    # ---------------------------------------------------------------- panel c
    horizons = np.linspace(5.0, 200.0, 60)
    pairs = [
        ("Afforestation", Afforestation(tau_growth=28.0, tau_disturbance=140.0),
         EXPOSURE["afforestation"], COLOURS["afforestation"]),
        ("Biochar", ExponentialRelease(tau=350.0), EXPOSURE["biochar"],
         COLOURS["biochar"]),
    ]
    for label, measure, exposure, colour in pairs:
        det = np.array([float(measure.cooling(cm, h)) for h in horizons])
        st = FireExposedStore(measure, region.hazard_template(exposure))
        mu = np.array([st.mean_cooling(cm, h, n_block=8) for h in horizons])
        sd = np.array(
            [np.sqrt(max(st.variance_cooling(cm, h, n_block=8), 0.0)) for h in horizons]
        )
        ax_c.plot(horizons, det * 1e14, color=colour, ls=(0, (4, 1.5)), lw=0.9)
        ax_c.plot(horizons, mu * 1e14, color=colour, path_effects=halo(), label=label)
        band(ax_c, horizons, (mu - sd) * 1e14, (mu + sd) * 1e14, colour, alpha=0.20)
    ax_c.set_xlabel("time horizon (yr)")
    ax_c.set_ylabel("cumulative cooling ($10^{-14}$ K yr kg$^{-1}$)")
    ax_c.set_xlim(0, 200)
    ax_c.set_ylim(bottom=0)
    ax_c.legend(
        loc="upper left", title="dashed: no fire", title_fontsize=6.0,
        borderaxespad=0.3,
    )
    light_grid(ax_c, "y")
    spine_style(ax_c)
    panel_label(ax_c, "c")

    # ---------------------------------------------------------------- panel d
    iv_d = Afforestation(tau_growth=28.0, tau_disturbance=140.0)
    hz_d = region.hazard_template(EXPOSURE["afforestation"])
    samples = _survival_samples(hz_d, iv_d, cm, HORIZON) * 1e14
    det_d = float(iv_d.cooling(cm, HORIZON)) * 1e14
    store_d = FireExposedStore(iv_d, hz_d)
    mu_d = store_d.mean_cooling(cm, HORIZON) * 1e14
    sd_d = np.sqrt(max(store_d.variance_cooling(cm, HORIZON), 0.0)) * 1e14

    ax_d.hist(
        samples,
        bins=70,
        color=COLOURS["afforestation"],
        alpha=0.55,
        density=True,
        lw=0,
    )
    kappa90 = np.sqrt(0.9 / 0.1)
    level90 = mu_d - kappa90 * sd_d
    ax_d.axvline(det_d, color=COLOURS["deterministic"], ls=(0, (4, 1.5)), lw=1.0)
    ax_d.axvline(mu_d, color=COLOURS["net"], lw=1.1)
    # The Chebyshev guarantee at ninety per cent lies below zero for a single
    # exposed store, because its coefficient of variation exceeds one third.
    # That is not a defect of the bound; it is the statement that no quantity
    # of one exposed measure guarantees any positive cooling at that
    # confidence, which is precisely why the robust portfolios of Figure 3
    # must hold something fire can not reach. The axis is extended to show it.
    lo = min(0.0, level90) - 0.25
    ax_d.set_xlim(lo, det_d * 1.06)
    if level90 > lo:
        ax_d.axvline(level90, color=COLOURS["fire"], lw=1.0, ls=(0, (1.4, 1.2)))
    ax_d.axvspan(lo, 0.0, color="#000000", alpha=0.045, lw=0)
    ax_d.set_xlabel("delivered cooling ($10^{-14}$ K yr kg$^{-1}$)")
    ax_d.set_ylabel("density")
    light_grid(ax_d, "y")
    spine_style(ax_d)
    panel_label(ax_d, "d")

    top = ax_d.get_ylim()[1]
    ax_d.annotate(
        "90% Chebyshev\nguarantee",
        xy=(level90, top * 0.42),
        xytext=(level90 + 0.15, top * 0.72),
        fontsize=5.8,
        color=COLOURS["fire"],
        ha="left",
        arrowprops=dict(arrowstyle="-", lw=0.45, color=COLOURS["fire"]),
    )
    ax_d.text(
        mu_d - 0.06,
        top * 0.97,
        "mean ",
        rotation=90,
        va="top",
        ha="right",
        fontsize=5.8,
        color=COLOURS["net"],
    )
    ax_d.text(
        det_d - 0.06,
        top * 0.97,
        "credited without fire ",
        rotation=90,
        va="top",
        ha="right",
        fontsize=5.8,
        color=COLOURS["deterministic"],
    )

    fig.tight_layout(pad=0.6, w_pad=1.4, h_pad=1.1)
    save(fig, "fig1_mechanism")

    frac_below = float(np.mean(samples < mu_d - kappa90 * sd_d))
    write_values(
        {
            "fig1_afforestation_loss_fraction": 1.0 - mu_d / det_d,
            "fig1_afforestation_cv": sd_d / mu_d,
            "fig1_chebyshev_90_level": mu_d - kappa90 * sd_d,
            "fig1_mass_below_90_guarantee": frac_below,
            "fig1_wus_intensification": WESTERN_US.hazard_template().intensification(
                HORIZON
            ),
            "fig1_boreal_intensification": BOREAL_CANADA.hazard_template().intensification(
                HORIZON
            ),
            "fig1_wus_survival_100yr": float(
                np.exp(-WESTERN_US.hazard_template().cumulative(
                    np.linspace(0, HORIZON, 401))[-1])
            ),
            "fig1_boreal_survival_100yr": float(
                np.exp(-BOREAL_CANADA.hazard_template().cumulative(
                    np.linspace(0, HORIZON, 401))[-1])
            ),
        }
    )
    print(
        "  afforestation loses {:.1%} of its credit on average; "
        "coefficient of variation {:.2f}".format(1.0 - mu_d / det_d, sd_d / mu_d)
    )
    print(
        "  Chebyshev 90% level sits at {:.2f}, with {:.2%} of simulated mass below it"
        .format(mu_d - kappa90 * sd_d, frac_below)
    )


if __name__ == "__main__":
    main()
