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

e   The equivalent storage time. Inverting the credit onto a fire-free store
    of the same credit expresses fire as a shortening of the storage time an
    offset protocol would assume.
f   The penalty across measures, which is uneven, so a buffer pool set as one
    percentage of every measure misprices all of them.

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

from matplotlib.lines import Line2D

from scenarios import portfolio

from cipo import AR6, Afforestation, ExponentialRelease
from wrcp.algebra import stock_signal
from wrcp.hazard import FireHazard
from wrcp.regions import BOREAL_CANADA, EXPOSURE, REGIONS, WESTERN_US
from wrcp.stochastic import FireExposedStore

HORIZON = 100.0

SHORT = {
    "Afforestation": "afforest.",
    "Improved forest management": "forest mgmt",
    "Soil carbon": "soil C",
    "Harvested wood products": "wood prod.",
    "Biochar": "biochar",
}


def equivalent_storage_time(cm, credit, horizon, lo=0.5, hi=2.0e4, n=60):
    """Storage time of a fire-free store that earns the same credit.

    Cumulative cooling rises monotonically with the mean storage time of a
    first-order store, so the map from storage time to credit is invertible.
    Inverting it expresses a measure in the unit an offset protocol already
    uses, the years a store is assumed to hold its carbon, and makes the effect
    of fire a change in that number rather than a change in an abstract credit.
    """

    def excess(tau):
        return float(ExponentialRelease(tau=tau).cooling(cm, horizon)) - credit

    if excess(lo) > 0.0:
        return lo
    if excess(hi) < 0.0:
        return hi
    for _ in range(n):
        mid = 0.5 * (lo + hi)
        if excess(mid) < 0.0:
            lo = mid
        else:
            hi = mid
    return 0.5 * (lo + hi)


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
    fig = plt.figure(figsize=(DOUBLE, 0.64 * DOUBLE))
    gs = fig.add_gridspec(
        2, 3, hspace=0.52, wspace=0.60, left=0.062, right=0.972,
        top=0.940, bottom=0.095,
    )
    ax_a = fig.add_subplot(gs[0, 0])
    ax_b = fig.add_subplot(gs[0, 1])
    ax_c = fig.add_subplot(gs[0, 2])
    ax_d = fig.add_subplot(gs[1, 0])
    ax_e = fig.add_subplot(gs[1, 1])
    ax_f = fig.add_subplot(gs[1, 2])
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
    ax_a2.set_ylabel("escape probability", color="#555555")
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
        label="without fire",
    )
    ax_b.plot(
        t,
        mean_stock,
        color=COLOURS["afforestation"],
        path_effects=halo(),
        label="mean under fire",
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
    ax_b.set_ylabel("stored stock (per peak unit)")
    ax_b.set_xlim(0, HORIZON)
    # Headroom above the deterministic stock, which the embedded panel uses.
    ax_b.set_ylim(0, 1.58)
    ax_b.set_yticks([0.0, 0.25, 0.5, 0.75, 1.0])
    ax_b.legend(
        loc="upper right", bbox_to_anchor=(1.005, 1.005), fontsize=5.8,
        labelspacing=0.28, handlelength=1.5, handletextpad=0.5,
        borderaxespad=0.0, borderpad=0.3,
    )
    light_grid(ax_b, "y")
    spine_style(ax_b)
    panel_label(ax_b, "b")
    # An embedded panel isolating the factor fire contributes. The mean stock
    # is the product of the deterministic stock and this multiplier, so the
    # host panel shows the product and the inset shows the cause.
    ax_b_in = ax_b.inset_axes([0.070, 0.680, 0.35, 0.28], zorder=6)
    ax_b_in.set_facecolor("white")
    ax_b_in.patch.set_alpha(1.0)
    for reg in (WESTERN_US, BOREAL_CANADA):
        hz_r = reg.hazard_template(EXPOSURE["afforestation"])
        edges_r = np.linspace(0.0, HORIZON, 401)
        cum_r = np.interp(t, edges_r, hz_r.cumulative(edges_r))
        ax_b_in.plot(
            t, np.exp(-hz_r.severity.m1 * cum_r),
            color=COLOURS[reg.name], lw=0.9,
        )
    ax_b_in.set_xlim(0, HORIZON)
    ax_b_in.set_ylim(0, 1.02)
    ax_b_in.set_xticks([0, 50, 100])
    ax_b_in.set_yticks([0, 0.5, 1.0])
    ax_b_in.tick_params(labelsize=4.6, length=1.8, pad=1.0)
    ax_b_in.text(
        0.04, 0.06, "survival multiplier $\\mathbb{E}[S(t)]$",
        transform=ax_b_in.transAxes, fontsize=4.6, ha="left", va="bottom",
    )
    spine_style(ax_b_in)

    ax_b.annotate(
        "over-credit",
        xy=(78.0, 0.5 * (det_stock[t.searchsorted(78.0)] + mean_stock[t.searchsorted(78.0)])),
        xytext=(60.0, 0.80),
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
    ax_d.axvline(mu_d, color=COLOURS["net"], lw=1.1, ymax=0.47)
    # The Chebyshev guarantee at ninety per cent lies below zero for a single
    # exposed store, because its coefficient of variation exceeds one third.
    # That is not a defect of the bound; it is the statement that no quantity
    # of one exposed measure guarantees any positive cooling at that
    # confidence, which is precisely why the robust portfolios of Figure 4
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
        xy=(level90, top * 0.16),
        xytext=(level90 + 0.10, top * 0.30),
        fontsize=5.8,
        color=COLOURS["fire"],
        ha="left",
        arrowprops=dict(arrowstyle="-", lw=0.45, color=COLOURS["fire"]),
    )
    ax_d.text(
        mu_d + 0.07,
        top * 0.44,
        "mean",
        rotation=90,
        va="top",
        ha="left",
        fontsize=5.8,
        color=COLOURS["net"],
    )
    # An embedded panel on the left tail, drawn with a logarithmic density so
    # the rare outcomes are visible at all. The tail is the reason the
    # guarantee is posed over every distribution with the computed moments
    # instead of over a named one.
    ax_d_in = ax_d.inset_axes([0.335, 0.505, 0.40, 0.42], zorder=6)
    ax_d_in.set_facecolor("white")
    ax_d_in.patch.set_alpha(1.0)
    # The cumulative form reads better here than a logarithmic histogram,
    # which saturates: it gives the probability of falling short of any given
    # credit, which is the quantity a guarantee is written against.
    ordered = np.sort(samples)
    grid = np.linspace(0.0, mu_d, 240)
    shortfall = np.searchsorted(ordered, grid, side="right") / ordered.size
    ax_d_in.plot(grid, np.clip(shortfall, 1e-5, None),
                 color=COLOURS["afforestation"], lw=1.0)
    ax_d_in.set_yscale("log")
    ax_d_in.set_xlim(0.0, mu_d)
    ax_d_in.set_ylim(1e-4, 1.4)
    ax_d_in.set_xticks([0.0, 1.0, 2.0])
    ax_d_in.set_yticks([1e-4, 1e-2, 1.0])
    ax_d_in.set_yticklabels(["$10^{-4}$", "$10^{-2}$", "1"])
    ax_d_in.minorticks_off()
    ax_d_in.tick_params(labelsize=4.4, length=1.8, pad=1.0)
    ax_d_in.text(
        0.04, 0.93, "P(credit $\\leq x$)", transform=ax_d_in.transAxes,
        fontsize=4.9, ha="left", va="top",
    )
    light_grid(ax_d_in, "y")
    spine_style(ax_d_in)

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


    # ---------------------------------------------------------------- panels e, f
    # Both panels read the portfolio moments, which average over fire-weather
    # years. That is the basis the optimisation uses, so the per-measure
    # numbers here agree with the tables and with Figures 2 to 5.
    order, rows = None, {}
    for key in ("western_us", "boreal_canada"):
        fp = portfolio(key)
        m = fp.moments()
        keep = [i for i, st in enumerate(fp.stores) if not st.hazard.is_inert]
        if order is None:
            order = [fp.stores[i].label for i in keep]
        rows[key] = [
            (
                fp.names[i],
                equivalent_storage_time(cm, float(m.deterministic[i]), HORIZON),
                equivalent_storage_time(cm, float(m.mean[i]), HORIZON),
                float(m.loss_fraction[i]),
            )
            for i in keep
        ]

    pos = np.arange(len(order))
    offsets = {"western_us": -0.18, "boreal_canada": 0.18}
    labels = [SHORT.get(lbl, lbl) for lbl in order]

    for key, dy in offsets.items():
        colour = COLOURS[key]
        for j, (_, nominal, effective, _) in enumerate(rows[key]):
            y = pos[j] + dy
            ax_e.plot([effective, nominal], [y, y], color=colour, lw=1.0,
                      solid_capstyle="round", zorder=2)
            ax_e.plot([nominal], [y], marker="o", ms=3.0, mfc="white",
                      mec=colour, mew=0.9, ls="none", zorder=3)
            ax_e.plot([effective], [y], marker="o", ms=3.0, color=colour,
                      ls="none", zorder=3)
    ax_e.set_xscale("log")
    ax_e.set_yticks(pos)
    ax_e.set_yticklabels(labels, fontsize=5.3)
    ax_e.invert_yaxis()
    ax_e.set_xlabel("equivalent storage time (yr)")
    ax_e.set_ylim(len(order) - 0.5, -0.5)
    handles = [
        Line2D([], [], marker="o", ms=3.0, mfc="white", mec=COLOURS["net"],
               mew=0.9, ls="none", label="without fire"),
        Line2D([], [], marker="o", ms=3.0, color=COLOURS["net"], ls="none",
               label="under fire"),
    ]
    ax_e.legend(
        handles=handles, fontsize=5.2, loc="lower right", frameon=False,
        labelspacing=0.25, handlelength=1.0, borderaxespad=0.3,
    )
    light_grid(ax_e, "x")
    spine_style(ax_e)
    panel_label(ax_e, "e", dx=-0.34, dy=1.10)

    width = 0.38
    for key, dy in offsets.items():
        ax_f.barh(
            pos + dy, [100.0 * r[3] for r in rows[key]], height=width,
            color=COLOURS[key], edgecolor="white", linewidth=0.3,
            label=REGIONS[key].label,
        )
    ax_f.set_yticks(pos)
    ax_f.set_yticklabels(labels, fontsize=5.3)
    ax_f.invert_yaxis()
    ax_f.set_xlabel("credit lost to fire (%)")
    ax_f.set_ylim(len(order) - 0.5, -0.5)
    ax_f.legend(
        fontsize=5.2, loc="lower right", frameon=False, labelspacing=0.25,
        handlelength=1.0, borderaxespad=0.3,
    )
    light_grid(ax_f, "x")
    spine_style(ax_f)
    panel_label(ax_f, "f", dx=-0.34, dy=1.10)

    save(fig, "fig1_mechanism")

    frac_below = float(np.mean(samples < mu_d - kappa90 * sd_d))
    write_values(
        {
            # Panel d simulates the fire process at a central fire-weather
            # year, so these two are conditional on that year. The mean credit
            # is convex in the year's severity, so the unconditional loss,
            # which the portfolio and the tables use, is smaller. The names
            # record which quantity is which.
            "fig1_afforestation_loss_fraction_central_year": 1.0 - mu_d / det_d,
            "fig1_afforestation_cv_central_year": sd_d / mu_d,
            "fig1_chebyshev_90_level": mu_d - kappa90 * sd_d,
            "fig1_equiv_storage_nominal_afforestation": rows["western_us"][0][1],
            "fig1_equiv_storage_under_fire_afforestation": rows["western_us"][0][2],
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
        "  afforestation loses {:.1%} of its credit at a central fire-weather "
        "year; coefficient of variation {:.2f}".format(
            1.0 - mu_d / det_d, sd_d / mu_d
        )
    )
    print(
        "  Chebyshev 90% level sits at {:.2f}, with {:.2%} of simulated mass below it"
        .format(mu_d - kappa90 * sd_d, frac_below)
    )


if __name__ == "__main__":
    main()
