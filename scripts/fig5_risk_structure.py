"""Figure 5: the structure of the risk that spreading projects leaves behind.

The main text reports that a bad fire year raises the hazard for every store in
a region at once, so a portfolio spread across many sites keeps a floor of
variance that no amount of spreading removes. That result carries a policy
conclusion, and until now it rested on a single bar panel. This figure gives it
its own treatment.

Panels:

    a, b  correlation of delivered cooling between measures, by region
    c     variance reduction against the number of independent sites, with the
          floor each region converges to
    d, e  the risk the optimal portfolio carries as the required confidence
          rises, split into the part spreading removes and the part it leaves
    f     coefficient of variation of each measure, by region

Every quantity is read from the cached scenario solutions, so the figure adds
no optimisation of its own.
"""

from __future__ import annotations

import numpy as np
from matplotlib import pyplot as plt
from matplotlib.colors import LinearSegmentedColormap

from common import (
    COLOURS,
    DOUBLE,
    configure,
    light_grid,
    measure_colours,
    panel_label,
    save,
    spine_style,
    write_values,
)
from scenarios import CONFIDENCES, solved

from wrcp.regions import REGIONS

# Sites to sweep in panel c. The curve is smooth, so a geometric grid resolves
# it and reaches the regime where the floor dominates.
SITES = np.unique(np.round(np.geomspace(1, 300, 40)).astype(int))


def short(label):
    """Measure labels short enough for a correlation matrix axis."""
    return {
        "Afforestation": "afforest.",
        "Improved forest management": "forest mgmt",
        "Soil carbon": "soil C",
        "Harvested wood products": "wood prod.",
        "Biochar": "biochar",
        "BECCS, geological storage": "BECCS",
        "Direct air capture": "direct air",
    }.get(label, label)


def exposed_block(data):
    """Indices, labels and colours of the measures fire can reach."""
    fp = data["portfolio"]
    idx = [i for i, s in enumerate(fp.stores) if not s.hazard.is_inert]
    labels = [short(fp.stores[i].label) for i in idx]
    colours = measure_colours([fp.names[i] for i in idx])
    return idx, labels, colours


def correlation_panel(ax, data, cmap):
    """Correlation of delivered cooling between the exposed measures."""
    idx, labels, _ = exposed_block(data)
    cov = np.asarray(data["moments"].covariance)[np.ix_(idx, idx)]
    sd = np.sqrt(np.clip(np.diag(cov), 0.0, None))
    corr = cov / np.outer(sd, sd)

    im = ax.imshow(corr, cmap=cmap, vmin=0.0, vmax=1.0, origin="upper")
    ax.set_xticks(range(len(labels)))
    ax.set_yticks(range(len(labels)))
    ax.set_xticklabels(labels, rotation=38, ha="right", fontsize=5.3)
    ax.set_yticklabels(labels, fontsize=5.3)
    ax.tick_params(length=0.0, pad=1.2)
    for i in range(len(labels)):
        for j in range(len(labels)):
            ax.text(
                j, i, "{:.2f}".format(corr[i, j]).lstrip("0"),
                ha="center", va="center", fontsize=4.7,
                color="white" if corr[i, j] > 0.62 else "#222222",
            )
    for spine in ax.spines.values():
        spine.set_visible(False)
    return im, corr


def reduction_curve(data):
    """Variance reduction from holding N independent copies of the portfolio.

    Spreading a fixed deployment over N sites averages the site-level part of
    the risk and leaves the common part untouched, so the variance of the
    average falls as ``idiosyncratic / N + common`` and the reduction factor
    approaches the inverse of the common share.
    """
    m = data["moments"]
    w = np.ones(m.covariance.shape[0])
    w = w / w.sum()
    idio = float(w @ np.asarray(m.idiosyncratic) @ w)
    common = float(w @ np.asarray(m.common) @ w)
    total = idio + common
    reduction = total / (idio / SITES + common)
    floor = total / common if common > 0 else np.inf
    return reduction, floor


def risk_panel(ax, data, region_key, show_ylabel):
    """How the optimal portfolio's risk splits as the guarantee tightens."""
    m = data["moments"]
    idio_m = np.asarray(m.idiosyncratic)
    common_m = np.asarray(m.common)

    etas, idio, common = [], [], []
    for eta in CONFIDENCES:
        res = data["results"][eta]
        if not getattr(res, "feasible", True):
            continue
        a = np.asarray(res.alpha, dtype=float)
        if not np.all(np.isfinite(a)):
            continue
        etas.append(eta)
        idio.append(float(a @ idio_m @ a))
        common.append(float(a @ common_m @ a))

    etas = np.array(etas)
    idio = np.array(idio) * 1e3
    common = np.array(common) * 1e3

    ax.stackplot(
        etas, idio, common,
        colors=[COLOURS["exposed"], COLOURS["fire"]],
        labels=["site timing and severity", "common fire-weather year"],
        edgecolor="white", linewidth=0.35,
    )
    ax.set_xlabel("required confidence $\\eta$")
    if show_ylabel:
        ax.set_ylabel("portfolio variance ($10^{-3}$ K$^2$ yr$^2$)")
    ax.set_xlim(etas[0], etas[-1])
    ax.set_ylim(bottom=0.0)
    ax.ticklabel_format(axis="y", style="plain")
    ax.text(
        0.0, 1.03, REGIONS[region_key].label, transform=ax.transAxes,
        fontsize=5.8, va="bottom", ha="left",
    )
    light_grid(ax, "y")
    spine_style(ax)
    return etas, idio, common


def main():
    configure()

    west = solved("western_us")
    boreal = solved("boreal_canada")

    warm = LinearSegmentedColormap.from_list(
        "corr_warm", ["#fdf3ec", COLOURS["exposed"], COLOURS["afforestation"]]
    )
    cool = LinearSegmentedColormap.from_list(
        "corr_cool", ["#eef3f7", COLOURS["boreal_canada"], COLOURS["daccs"]]
    )

    fig = plt.figure(figsize=(DOUBLE, 0.66 * DOUBLE))
    gs = fig.add_gridspec(
        2, 3, hspace=0.66, wspace=0.52, left=0.085, right=0.975,
        top=0.930, bottom=0.115,
    )
    ax_a = fig.add_subplot(gs[0, 0])
    ax_b = fig.add_subplot(gs[0, 1])
    ax_c = fig.add_subplot(gs[0, 2])
    ax_d = fig.add_subplot(gs[1, 0])
    ax_e = fig.add_subplot(gs[1, 1])
    ax_f = fig.add_subplot(gs[1, 2])

    # ------------------------------------------------------------ panels a, b
    im_a, corr_w = correlation_panel(ax_a, west, warm)
    im_b, corr_b = correlation_panel(ax_b, boreal, cool)
    for ax, label in ((ax_a, "western_us"), (ax_b, "boreal_canada")):
        ax.text(
            0.0, 1.16, REGIONS[label].label, transform=ax.transAxes,
            fontsize=5.8, va="bottom", ha="left",
        )
    # A single colour bar serves both matrices. The scale is shared, so a
    # second one would repeat it and crowd the axis to its right.
    cb = fig.colorbar(im_b, ax=ax_b, pad=0.03, fraction=0.045)
    cb.set_ticks([0.0, 0.5, 1.0])
    cb.ax.tick_params(labelsize=5.2)
    cb.set_label("correlation", fontsize=6.0)
    panel_label(ax_a, "a", dx=-0.38, dy=1.30)
    panel_label(ax_b, "b", dx=-0.38, dy=1.30)

    # --------------------------------------------------------------- panel c
    values = {}
    for key, data in (("western_us", west), ("boreal_canada", boreal)):
        reduction, floor = reduction_curve(data)
        colour = COLOURS[key]
        ax_c.plot(
            SITES, reduction, color=colour, lw=1.2,
            label=REGIONS[key].label,
        )
        ax_c.axhline(floor, color=colour, lw=0.6, ls=(0, (2.2, 1.6)))
        ax_c.text(
            SITES[-1], floor, "  {:.1f}$\\times$".format(floor),
            fontsize=5.2, color=colour, va="center", ha="left",
        )
        values["variance_floor_factor_" + key] = float(floor)
        values["variance_reduction_100_sites_" + key] = float(
            np.interp(100, SITES, reduction)
        )
    # Independent risk would fall without limit, which is the comparison the
    # floor is measured against.
    ax_c.plot(
        SITES, SITES.astype(float), color=COLOURS["deterministic"], lw=0.8,
        ls=(0, (1.4, 1.4)), label="independent sites",
    )
    ax_c.set_xscale("log")
    ax_c.set_yscale("log")
    ax_c.set_xlabel("number of sites")
    ax_c.set_ylabel("variance reduction")
    ax_c.set_xlim(1, SITES[-1] * 1.9)
    ax_c.set_ylim(0.9, 90.0)
    ax_c.legend(
        fontsize=5.3, loc="upper left", frameon=False,
        labelspacing=0.25, handlelength=1.3, borderaxespad=0.3,
    )
    light_grid(ax_c)
    spine_style(ax_c)
    panel_label(ax_c, "c", dy=1.10)

    # ------------------------------------------------------------ panels d, e
    risk_panel(ax_d, west, "western_us", True)
    etas, idio, common = risk_panel(ax_e, boreal, "boreal_canada", False)
    ax_e.legend(
        fontsize=5.1, loc="upper right", frameon=False,
        labelspacing=0.22, handlelength=1.1, borderaxespad=0.35,
    )
    panel_label(ax_d, "d", dy=1.20)
    panel_label(ax_e, "e", dy=1.20)

    # --------------------------------------------------------------- panel f
    idx_w, labels_w, _ = exposed_block(west)
    cv_w = np.asarray(west["moments"].coefficient_of_variation)[idx_w]
    idx_b, _, _ = exposed_block(boreal)
    cv_b = np.asarray(boreal["moments"].coefficient_of_variation)[idx_b]

    pos = np.arange(len(labels_w))
    width = 0.38
    ax_f.barh(
        pos + width / 2, cv_w, height=width, color=COLOURS["western_us"],
        label=REGIONS["western_us"].label, edgecolor="white", linewidth=0.3,
    )
    ax_f.barh(
        pos - width / 2, cv_b, height=width, color=COLOURS["boreal_canada"],
        label=REGIONS["boreal_canada"].label, edgecolor="white", linewidth=0.3,
    )
    ax_f.set_yticks(pos)
    ax_f.set_yticklabels(labels_w, fontsize=5.3)
    ax_f.invert_yaxis()
    ax_f.set_xlabel("coefficient of variation $\\sigma_i/\\mu_i$")
    ax_f.legend(
        fontsize=5.3, loc="lower right", frameon=False,
        labelspacing=0.25, handlelength=1.1, borderaxespad=0.35,
    )
    light_grid(ax_f, "x")
    spine_style(ax_f)
    panel_label(ax_f, "f", dx=-0.34, dy=1.10)

    # The largest off-diagonal correlation is what the text means by a bad year
    # taking many stores at once, so it is recorded with the other key values.
    for key, corr in (("western_us", corr_w), ("boreal_canada", corr_b)):
        off = corr[~np.eye(corr.shape[0], dtype=bool)]
        values["max_cross_correlation_" + key] = float(np.max(off))
    write_values(values)

    save(fig, "fig5_risk_structure")


if __name__ == "__main__":
    main()
