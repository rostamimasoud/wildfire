"""Figure 2: the fire premium and the confidence ceiling.

a   The premium against confidence, for both regions, decomposed. The lower
    band is the correction for the *mean* over-credit, which any honest
    inventory owes regardless of how much confidence it wants. The upper band
    is the cost of the guarantee itself. Separating the two matters because the
    first is not a choice and the second is.

b   Where the variance comes from. The diagonal of the covariance split into
    the part that diversification removes, which is the timing and severity of
    fire at one site, and the part it cannot, which is the common fire-weather
    year. The second sets a floor: a portfolio of a hundred forests in one
    region is not a hundred times safer than one.

c   The confidence ceiling. The signal-to-noise ratio attainable along
    non-negative deployment directions, as a function of how much fire-proof
    capacity is available. With none, the ceiling is finite and is the
    fundamental limit the research plan describes. With any, it is unbounded,
    and the limit becomes one of cost and capacity rather than of physics. This
    panel is the correction to that part of the plan.

d   Sensitivity of the premium to the two least constrained calibration
    choices: the climate sensitivity of fire frequency, inverted from the
    projected intensification, and the interannual spread, inverted from the
    largest anomaly in the record. The premium is reported across the plausible
    range rather than at a point.
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
    measure_colours,
    panel_label,
    save,
    save_table,
    spine_style,
    write_values,
)
from scenarios import CONFIDENCES, climate, portfolio, solved

from wrcp.regions import BOREAL_CANADA, REGIONS, WESTERN_US, build_stores
from wrcp.robust import FirePortfolio, confidence_from_kappa
from wrcp.stochastic import FireExposedStore


def main():
    configure()
    import matplotlib.pyplot as plt

    fig, axes = plt.subplots(2, 2, figsize=(DOUBLE, 0.66 * DOUBLE))
    (ax_a, ax_b), (ax_c, ax_d) = axes
    records = {}

    # ---------------------------------------------------------------- panel a
    for region in (WESTERN_US, BOREAL_CANADA):
        data = solved(region.name)
        prem = data["premium"]
        ok = prem["feasible"].to_numpy(dtype=bool)
        eta = prem["confidence"].to_numpy()[ok]
        total = prem["premium_vs_deterministic"].to_numpy()[ok]
        guarantee = prem["premium_vs_expectation"].to_numpy()[ok]
        mean_only = prem.attrs["mean_over_credit"]
        colour = COLOURS[region.name]
        ax_a.plot(
            eta,
            100.0 * total,
            color=colour,
            path_effects=halo(),
            label=region.label,
        )
        ax_a.axhline(
            100.0 * mean_only,
            color=colour,
            ls=(0, (1.6, 1.4)),
            lw=0.8,
        )
        ax_a.fill_between(
            eta,
            100.0 * mean_only,
            100.0 * total,
            color=colour,
            alpha=0.11,
            lw=0,
        )
        records[region.name] = dict(
            mean_over_credit=mean_only,
            premium=dict(zip(eta.round(3), total)),
            guarantee=dict(zip(eta.round(3), guarantee)),
        )
    ax_a.set_xlabel("required confidence $\\eta$")
    ax_a.set_ylabel("fire premium (% of deterministic)")
    ax_a.set_xlim(0.48, 1.0)
    ax_a.set_ylim(bottom=0)
    ax_a.annotate(
        "mean over-credit alone (dashed):\nowed at any confidence",
        xy=(0.57, 100.0 * records["western_us"]["mean_over_credit"]),
        xytext=(0.52, 24.0),
        fontsize=6.0,
        color="#555555",
        arrowprops=dict(arrowstyle="-", lw=0.45, color="#888888"),
    )
    ax_a.annotate(
        "price of the guarantee",
        xy=(0.93, 0.5 * 100.0 * (records["western_us"]["premium"][0.9]
                                 + records["western_us"]["mean_over_credit"])),
        xytext=(0.66, 330.0),
        fontsize=6.0,
        color=COLOURS["western_us"],
        arrowprops=dict(arrowstyle="-", lw=0.45, color=COLOURS["western_us"]),
    )
    ax_a.legend(loc="upper left", borderaxespad=0.3)
    light_grid(ax_a, "y")
    spine_style(ax_a)
    panel_label(ax_a, "a", dx=-0.185)

    # ---------------------------------------------------------------- panel b
    region = WESTERN_US
    data = solved(region.name)
    m = data["moments"]
    fp = data["portfolio"]
    exposed = np.array([not s.hazard.is_inert for s in fp.stores])
    idx = np.where(exposed)[0]
    labels = [
        fp.labels[i].split(",")[0].replace("Improved forest", "Improved\nforest")
        .replace("Harvested wood", "Harvested\nwood")
        for i in idx
    ]
    idio = np.diag(m.idiosyncratic)[idx]
    comm = np.diag(m.common)[idx]
    scale = 1e29
    x = np.arange(len(idx))
    ax_b.bar(
        x,
        idio * scale,
        color=COLOURS["exposed"],
        label="site timing and severity (diversifiable)",
        lw=0,
    )
    ax_b.bar(
        x,
        comm * scale,
        bottom=idio * scale,
        color=COLOURS["fire"],
        label="common fire-weather year (not diversifiable)",
        lw=0,
    )
    ax_b.set_xticks(x)
    ax_b.set_xticklabels(labels, rotation=0, fontsize=5.6)
    ax_b.set_ylabel("variance of cooling ($10^{-29}$ K$^2$ yr$^2$ kg$^{-2}$)")
    ax_b.set_ylim(0, float(np.max(idio + comm) * scale) * 1.62)
    ax_b.legend(
        loc="upper center", fontsize=5.6, borderaxespad=0.2,
        handlelength=0.9, labelspacing=0.22,
    )
    light_grid(ax_b, "y")
    spine_style(ax_b)
    panel_label(ax_b, "b", dx=-0.155)
    ax_b.text(
        0.97,
        0.70,
        "equal-weight portfolio:\n{:.0%} diversifiable".format(m.diversifiable_share),
        transform=ax_b.transAxes,
        fontsize=6.0,
        ha="right",
        color=COLOURS["net"],
    )

    # ---------------------------------------------------------------- panel c
    # The ceiling as fire-proof capacity is admitted, from none upwards. The
    # relevant axis is the share of the required cooling that fire-proof
    # measures could supply if deployed to their ceilings.
    for region in (WESTERN_US, BOREAL_CANADA):
        base = portfolio(region.name)
        m_r = base.moments()
        protected = np.array([s.hazard.is_inert for s in base.stores])
        caps = np.array(
            [0.0 if s.max_scale is None else float(s.max_scale) for s in base.stores]
        )
        be = base.required_cooling()
        full_share = float(m_r.mean[protected] @ caps[protected] / be)
        shares = np.linspace(0.0, min(full_share, 2.5), 11)
        ceilings = []
        kappas = []
        for sh in shares:
            stores = []
            for s, prot, cap in zip(base.stores, protected, caps):
                if prot:
                    frac = sh / full_share if full_share > 0 else 0.0
                    iv = s.intervention
                    if frac <= 0.0:
                        continue
                    import copy

                    iv2 = copy.copy(iv)
                    iv2.max_scale = float(cap * frac)
                    stores.append(FireExposedStore(iv2, s.hazard, s.label))
                else:
                    stores.append(s)
            trial = FirePortfolio(
                climate(),
                stores,
                base.pathway,
                base.horizon,
                n_block=8,
                weather_spread=region.weather_spread,
                n_weather=5,
            )
            k = trial.attainability()["kappa_max"]
            kappas.append(k)
            ceilings.append(1.0 if not np.isfinite(k) else confidence_from_kappa(k))
        # The ceiling itself saturates at one as soon as any fire-proof
        # capacity is admitted, so plotting it would show a step and hide the
        # magnitude. The signal-to-noise ratio kappa_max is the quantity that
        # carries the information: it is what the required confidence must fall
        # below, and it diverges the moment a zero-variance direction exists.
        kap = np.array(kappas, dtype=float)
        finite = np.isfinite(kap)
        # Only the first point has no fire-proof capacity at all. Every other
        # point admits a zero-variance direction, so kappa_max is unbounded
        # there and the confidence ceiling is one. Plotting the divergence as a
        # curve would waste the axis on an asymptote; it is marked instead.
        ax_c.plot(
            shares[finite],
            kap[finite],
            color=COLOURS[region.name],
            marker="o",
            ms=3.4,
            ls="none",
            path_effects=halo(),
            label=region.label,
        )
        records[region.name]["ceiling_no_fireproof"] = float(ceilings[0])
        records[region.name]["kappa_no_fireproof"] = float(kap[0])
    x_hi = 1.15
    ax_c.axvspan(0.045, x_hi, color=COLOURS["protected"], alpha=0.07, lw=0)
    eta_ticks = (0.9, 0.95, 0.98, 0.99)
    for eta in eta_ticks:
        ax_c.axhline(
            np.sqrt(eta / (1.0 - eta)), color="#aaaaaa", lw=0.5, ls=(0, (2.5, 2))
        )
    ax_c.set_yscale("log")
    ax_c.set_xlim(-0.30, x_hi)
    ax_c.set_ylim(2.5, 40.0)
    ax_c.set_yticks([3, 5, 7, 10, 15, 25, 35])
    ax_c.set_yticklabels(["3", "5", "7", "10", "15", "25", "35"])
    ax_c.minorticks_off()
    # The confidence each ratio corresponds to, on the right-hand axis: the
    # mapping is monotone, so the same gridlines carry both readings.
    ax_c2 = ax_c.twinx()
    ax_c2.set_yscale("log")
    ax_c2.set_ylim(ax_c.get_ylim())
    ax_c2.set_yticks([np.sqrt(e / (1.0 - e)) for e in eta_ticks])
    ax_c2.set_yticklabels(["{:g}".format(e) for e in eta_ticks], fontsize=6.0)
    ax_c2.minorticks_off()
    ax_c2.set_ylabel("guaranteed confidence $\\eta$", color="#777777", fontsize=6.4)
    ax_c2.tick_params(axis="y", colors="#777777")
    for side in ("top", "left"):
        ax_c2.spines[side].set_visible(False)
    ax_c.set_xlabel("fire-proof capacity (multiple of required cooling)")
    ax_c.set_ylabel("attainable $\\kappa_{\\max}$", labelpad=1.0)
    ax_c.legend(loc="lower left", borderaxespad=0.4, fontsize=5.8)
    light_grid(ax_c, "y")
    spine_style(ax_c)
    panel_label(ax_c, "c", dx=-0.185)
    ax_c.annotate(
        "none available:\n$\\eta_{\\max}$ = "
        + "{:.3f}".format(records["western_us"]["ceiling_no_fireproof"]),
        xy=(0.0, records["western_us"]["kappa_no_fireproof"]),
        xytext=(-0.28, 22.0),
        fontsize=6.0,
        color=COLOURS["fire"],
        arrowprops=dict(arrowstyle="-", lw=0.45, color=COLOURS["fire"]),
    )
    ax_c.text(
        0.60,
        0.62,
        "any fire-proof capacity:\n$\\kappa_{\\max}$ unbounded, $\\eta_{\\max} = 1$",
        transform=ax_c.transAxes,
        fontsize=6.0,
        ha="center",
        color=COLOURS["protected"],
    )

    # ---------------------------------------------------------------- panel d
    # The two calibration choices have different units and different plausible
    # ranges, so each is plotted against its own absolute value on its own
    # axis. Normalising both to "multiples of the adopted value" would put an
    # intensification of 5 next to an anomaly ratio of 8 as though they were
    # comparable perturbations, which they are not.
    region = WESTERN_US
    eta_ref = 0.9
    import copy

    specs = [
        (
            "intensification",
            "projected intensification",
            np.array([1.5, 2.0, 2.6, 3.3, 4.0, 5.0]),
            region.intensification,
            COLOURS["afforestation"],
            "o",
            ax_d,
        ),
        (
            "anomaly_ratio",
            "largest anomaly (multiple of mean)",
            np.array([2.0, 3.0, 3.5, 4.5, 6.0, 8.0]),
            region.anomaly_ratio,
            COLOURS["biochar"],
            "s",
            ax_d.twiny(),
        ),
    ]
    handles_d = []
    for attr, axis_label, values, adopted, colour, marker, ax in specs:
        prem = []
        for v in values:
            r2 = copy.copy(region)
            setattr(r2, attr, float(v))
            fp2 = FirePortfolio(
                climate(),
                build_stores(r2),
                r2.pathway(),
                r2.horizon,
                n_block=8,
                weather_spread=r2.weather_spread,
                n_weather=5,
            )
            det = fp2.deterministic_solution()
            res = fp2.solve(eta_ref)
            prem.append(100.0 * (res.cost / det.cost - 1.0) if res.feasible else np.nan)
        (line,) = ax.plot(
            values,
            prem,
            color=colour,
            marker=marker,
            ms=2.8,
            path_effects=halo(),
            label=axis_label,
        )
        handles_d.append(line)
        ax.scatter(
            [adopted],
            [float(np.interp(adopted, values, prem))],
            s=26,
            facecolor="white",
            edgecolor=colour,
            zorder=6,
            lw=0.9,
        )
        ax.set_xlabel(axis_label, color=colour)
        ax.tick_params(axis="x", colors=colour)
        ax.spines["right"].set_visible(False)
        records.setdefault("sensitivity", {})[attr] = dict(
            zip(values.round(3), np.round(prem, 2))
        )
    ax_d.set_ylabel("fire premium at $\\eta = 0.9$ (per cent)")
    ax_d.spines["top"].set_visible(True)
    ax_d.legend(
        handles_d,
        [h.get_label() for h in handles_d],
        loc="lower right",
        fontsize=5.8,
        borderaxespad=0.3,
        title="open circle: adopted value",
        title_fontsize=5.6,
    )
    light_grid(ax_d, "y")
    spine_style(ax_d)
    panel_label(ax_d, "d", dx=-0.155)

    fig.tight_layout(pad=0.9, w_pad=2.4, h_pad=1.6)
    save(fig, "fig2_premium")

    vals = {}
    for name in ("western_us", "boreal_canada"):
        r = records[name]
        vals["premium_mean_over_credit_" + name] = r["mean_over_credit"]
        for eta in (0.5, 0.75, 0.9, 0.95):
            if eta in r["premium"]:
                vals["premium_{}_{:.0f}".format(name, 100 * eta)] = r["premium"][eta]
                vals["guarantee_{}_{:.0f}".format(name, 100 * eta)] = r["guarantee"][eta]
        vals["ceiling_no_fireproof_" + name] = r["ceiling_no_fireproof"]
    vals["diversifiable_share_western_us"] = solved("western_us")["moments"].diversifiable_share
    vals["diversifiable_share_boreal_canada"] = solved("boreal_canada")["moments"].diversifiable_share
    write_values(vals)
    for k in sorted(vals):
        print("  {} = {:.4g}".format(k, vals[k]))


if __name__ == "__main__":
    main()
