"""Figure 3: what a robust portfolio actually holds.

a, b  Composition against required confidence, for each region, as a share of
      delivered cooling. The deterministic portfolio is shown at the left as a
      separate bar, because it answers a different question and is not a point
      on the confidence axis. The shift from warm colours to cool is the
      result: as the guarantee tightens, cooling migrates out of what fire can
      reach and into what it cannot.

c     The mechanism behind that shift, as the optimality condition sees it.
      Each measure's *effective* cooling is its mean contribution less the
      marginal risk it adds to the portfolio. A measure whose effective
      cooling is driven to zero cannot enter at any price. Plotting effective
      cooling per unit cost against confidence shows the ordering changing
      hands, which is the content of the cost-effectiveness theorem.

d     Timing. The fire-aware deployment schedule against the fire-blind one.
      Exposure accumulates, so the schedule that ignores fire front-loads
      deployment and pays for it; the fire-aware schedule delays the exposed
      measures and brings forward the protected ones.
"""

from __future__ import annotations

import numpy as np

from common import (
    COLOURS,
    DOUBLE,
    MEASURE_ORDER,
    configure,
    halo,
    light_grid,
    measure_colours,
    panel_label,
    save,
    spine_style,
    write_values,
)
from scenarios import CONFIDENCES, climate, solved

from wrcp.regions import BOREAL_CANADA, WESTERN_US, build_stores
from wrcp.robust import kappa
from wrcp.schedule import FireSchedule


def _shares(data, eta_list):
    """Share of delivered cooling by measure, at each confidence."""
    fp = data["portfolio"]
    mean = data["moments"].mean
    out = []
    for eta in eta_list:
        res = data["results"][eta]
        if not res.feasible:
            out.append(np.full(fp.n, np.nan))
            continue
        cooling = res.alpha * mean
        total = float(np.sum(cooling))
        out.append(cooling / total if total > 0 else cooling)
    return np.array(out)


def _stack(ax, x, shares, names, width):
    bottom = np.zeros(len(x))
    colours = measure_colours(names)
    handles = []
    for i, name in enumerate(names):
        h = ax.bar(
            x,
            shares[:, i],
            bottom=bottom,
            width=width,
            color=colours[i],
            lw=0.25,
            edgecolor="white",
        )
        bottom = bottom + np.nan_to_num(shares[:, i])
        handles.append(h)
    return handles


def main():
    configure()
    import matplotlib.pyplot as plt

    fig, axes = plt.subplots(2, 2, figsize=(DOUBLE, 0.70 * DOUBLE))
    (ax_a, ax_b), (ax_c, ax_d) = axes
    etas = [e for e in CONFIDENCES]

    handles = None
    for ax, region, letter in (
        (ax_a, WESTERN_US, "a"),
        (ax_b, BOREAL_CANADA, "b"),
    ):
        data = solved(region.name)
        fp = data["portfolio"]
        names = fp.names
        shares = _shares(data, etas)

        det = data["deterministic"]
        det_cooling = det.alpha * data["moments"].deterministic
        det_shares = (det_cooling / det_cooling.sum()).reshape(1, -1)

        # The confidence levels are not evenly spaced, so they are placed on a
        # categorical axis: a linear one would crush the closely spaced
        # high-confidence cases, which are the ones the result turns on.
        x_det = np.array([-1.4])
        x = np.arange(len(etas), dtype=float)
        h = _stack(ax, x_det, det_shares, names, 0.78)
        _stack(ax, x, shares, names, 0.82)
        if handles is None:
            handles = h

        ax.set_xlim(-2.1, len(etas) - 0.4)
        ax.set_ylim(0, 1)
        ax.set_xticks([-1.4] + list(x))
        ax.set_xticklabels(
            ["det."] + ["{:g}".format(e) for e in etas], fontsize=5.8
        )
        ax.set_xlabel("required confidence $\\eta$")
        ax.set_ylabel("share of delivered cooling")
        ax.axvline(-0.72, color="#999999", lw=0.6)
        ax.text(
            0.99,
            1.02,
            region.label,
            transform=ax.transAxes,
            ha="right",
            va="bottom",
            fontsize=6.8,
            color=COLOURS["net"],
        )
        spine_style(ax)
        panel_label(ax, letter)

    labels = [
        s.label.split(",")[0] for s in solved("western_us")["portfolio"].stores
    ]
    fig.legend(
        [h[0] for h in handles],
        labels,
        loc="upper center",
        bbox_to_anchor=(0.5, 1.005),
        ncol=4,
        frameon=False,
        fontsize=6.0,
        columnspacing=1.1,
        handlelength=1.1,
    )

    # ---------------------------------------------------------------- panel c
    data = solved("western_us")
    fp = data["portfolio"]
    m = data["moments"]
    marg_cost = np.array(
        [s.marginal_cost(a) for s, a in zip(fp.stores, data["results"][0.9].alpha)]
    )
    for i, store in enumerate(fp.stores):
        eff = []
        for eta in etas:
            res = data["results"][eta]
            if not res.feasible:
                eff.append(np.nan)
                continue
            a = res.alpha
            var = float(a @ m.covariance @ a + fp.pathway_variance)
            risk = (m.covariance @ a) / np.sqrt(var) if var > 0 else np.zeros_like(a)
            eff.append(float(m.mean[i] - kappa(eta) * risk[i]))
        ax_c.plot(
            etas,
            np.array(eff) * 1e14,
            color=COLOURS[fp.names[i]],
            marker="o",
            ms=2.0,
            lw=1.0,
            path_effects=halo(1.4),
            label=labels[i],
        )
    ax_c.axhline(0.0, color=COLOURS["fire"], lw=0.7, ls=(0, (2, 1.6)))
    ax_c.set_xlabel("required confidence $\\eta$")
    ax_c.set_ylabel("effective cooling ($10^{-14}$ K yr kg$^{-1}$)")
    ax_c.set_xlim(0.48, 1.0)
    light_grid(ax_c, "y")
    spine_style(ax_c)
    panel_label(ax_c, "c")
    ax_c.set_ylim(bottom=min(-0.55, ax_c.get_ylim()[0]))
    ax_c.text(
        0.985,
        0.035,
        "below zero: cannot help at any price",
        fontsize=5.9,
        color=COLOURS["fire"],
        transform=ax_c.transAxes,
        ha="right",
        va="bottom",
    )
    ax_c.legend(
        loc="center left", fontsize=5.5, borderaxespad=0.3,
        handlelength=1.0, labelspacing=0.22,
    )

    # ---------------------------------------------------------------- panel d
    region = WESTERN_US
    times = np.array([0.0, 15.0, 30.0, 45.0])
    sched = FireSchedule(
        climate(),
        build_stores(region),
        region.pathway(),
        horizon=region.horizon,
        decision_times=times,
        n_block=8,
        weather_spread=region.weather_spread,
        n_weather=5,
    )
    aware = sched.solve(confidence=0.9, discount_rate=0.0)
    blind = FireSchedule(
        climate(),
        [
            type(s)(s.intervention, s.hazard.replace(lambda0=1e-10), s.label)
            for s in build_stores(region)
        ],
        region.pathway(),
        horizon=region.horizon,
        decision_times=times,
        n_block=8,
        weather_spread=0.0,
    ).solve(confidence=0.9, discount_rate=0.0)

    exposed = np.array([not s.hazard.is_inert for s in sched.stores])
    pos = np.arange(len(times), dtype=float)
    width = 0.38
    if aware.feasible:
        aw_exp = aware.deployment[exposed].sum(axis=0) / 1e12
        aw_pro = aware.deployment[~exposed].sum(axis=0) / 1e12
        ax_d.bar(pos - width / 2, aw_exp, width, color=COLOURS["exposed"],
                 label="fire-exposed", lw=0)
        ax_d.bar(pos - width / 2, aw_pro, width, bottom=aw_exp,
                 color=COLOURS["protected"], label="protected", lw=0)
    if blind.feasible:
        bl_exp = blind.deployment[exposed].sum(axis=0) / 1e12
        bl_pro = blind.deployment[~exposed].sum(axis=0) / 1e12
        ax_d.bar(pos + width / 2, bl_exp, width, color=COLOURS["exposed"],
                 lw=0.5, edgecolor="white", hatch="////", alpha=0.55)
        ax_d.bar(pos + width / 2, bl_pro, width, bottom=bl_exp,
                 color=COLOURS["protected"], lw=0.5, edgecolor="white",
                 hatch="////", alpha=0.55)
    ax_d.set_xlabel("deployment start year")
    ax_d.set_ylabel("deployment (Gt CO$_2$ at peak storage)")
    ax_d.set_xticks(pos)
    ax_d.set_xticklabels(["{:g}".format(t) for t in times])
    ax_d.set_ylim(top=float(ax_d.get_ylim()[1]) * 1.30)
    ax_d.legend(loc="upper right", fontsize=6.0, borderaxespad=0.3)
    ax_d.text(
        0.015, 0.965,
        "left bar: fire-aware      right bar (hatched): fire-blind",
        transform=ax_d.transAxes, fontsize=5.9, color=COLOURS["net"], va="top",
    )
    light_grid(ax_d, "y")
    spine_style(ax_d)
    panel_label(ax_d, "d")

    fig.tight_layout(pad=0.6, w_pad=1.6, h_pad=1.2, rect=(0, 0, 1, 0.945))
    save(fig, "fig3_portfolio")

    vals = {}
    for name in ("western_us", "boreal_canada"):
        d = solved(name)
        fp_n = d["portfolio"]
        prot = np.array([s.hazard.is_inert for s in fp_n.stores])
        for eta in (0.5, 0.9, 0.95):
            res = d["results"][eta]
            if not res.feasible:
                continue
            cooling = res.alpha * d["moments"].mean
            vals["protected_share_{}_{:.0f}".format(name, 100 * eta)] = float(
                cooling[prot].sum() / cooling.sum()
            )
        det = d["deterministic"]
        dc = det.alpha * d["moments"].deterministic
        vals["protected_share_{}_det".format(name)] = float(dc[prot].sum() / dc.sum())
    if aware.feasible and blind.feasible:
        vals["schedule_cost_fire_aware"] = aware.cost
        vals["schedule_cost_fire_blind"] = blind.cost
        com = aware.centre_of_mass()
        for lab, c in zip(labels, com):
            if np.isfinite(c):
                vals["schedule_com_" + lab.replace(" ", "_").lower()] = c
    write_values(vals)
    for k in sorted(vals):
        print("  {} = {:.4g}".format(k, vals[k]))


if __name__ == "__main__":
    main()
