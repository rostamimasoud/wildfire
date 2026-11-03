"""Figure 3: the fire-adjusted credit as a surface.

The main-text figures report the two calibrated regimes. A reader who wants to
place a different forest, a different fire return interval or a different
severity mix needs the surrounding landscape, so this figure sweeps the two
parameters that set the fire penalty and the two that set the guarantee, and
draws the result as continuous surfaces.

Everything here is closed form. The mean and the variance of delivered cooling
come from the same expressions the main results use, evaluated on a grid of
parameters instead of at the two calibrated points, so no optimisation runs and
no simulation is involved.

Panels:

    a  mean loss of credit over fire return interval and mean storage time
    b  coefficient of variation over the same plane
    c  survival probability as a surface over time and intensification
    d  single-measure confidence ceiling over return interval and severity
"""

from __future__ import annotations

import numpy as np
from matplotlib import cm as mpl_cm
from matplotlib import pyplot as plt
from mpl_toolkits.mplot3d import Axes3D  # noqa: F401  (registers the projection)

from common import (
    COLOURS,
    DOUBLE,
    configure,
    light_grid,
    panel_label,
    save,
    spine_style,
)
from scenarios import climate

from cipo import ExponentialRelease
from wrcp.hazard import FireHazard, TwoPointSeverity, temperature_path
from wrcp.stochastic import FireExposedStore

HORIZON = 100.0
N_BLOCK = 10

# Grid resolution. The surfaces are smooth, so a modest grid resolves them and
# keeps the figure a few seconds of work.
NG = 26


def hazard_for(return_interval, stand_replacing=0.35, intensification=3.3):
    """A hazard with the stated return interval, severity mix and trend.

    ``beta_T`` is inverted from the intensification factor the same way the
    calibration does it, so a point on these surfaces is the same kind of
    object as a calibrated regime.
    """
    t_path = temperature_path(t0=1.0, t_peak=3.2, tau=70.0)
    dt = float(t_path(np.array([HORIZON]))[0] - t_path(np.array([0.0]))[0])
    beta_T = float(np.log(intensification) / dt)
    return FireHazard(
        lambda0=1.0 / float(return_interval),
        beta_T=beta_T,
        beta_D=0.0,
        severity=TwoPointSeverity(p=float(stand_replacing), partial=0.35),
        temperature=t_path,
        drought=None,
        region="sweep",
    )


def store_for(storage_time, hazard):
    """A first-order store of the stated mean storage time under a hazard."""
    return FireExposedStore(
        ExponentialRelease(float(storage_time)), hazard, label="sweep"
    )


def sweep_loss_and_spread(cmodel):
    """Mean loss and coefficient of variation over (return interval, storage)."""
    intervals = np.linspace(30.0, 200.0, NG)
    storage = np.linspace(20.0, 400.0, NG)
    loss = np.zeros((NG, NG))
    spread = np.zeros((NG, NG))
    for i, tau in enumerate(storage):
        for j, phi in enumerate(intervals):
            store = store_for(tau, hazard_for(phi))
            det = store.deterministic_cooling(cmodel, HORIZON)
            mu = store.mean_cooling(cmodel, HORIZON, N_BLOCK)
            var = store.variance_cooling(cmodel, HORIZON, N_BLOCK)
            loss[i, j] = 1.0 - mu / det if det > 0 else np.nan
            spread[i, j] = np.sqrt(max(var, 0.0)) / mu if mu > 0 else np.nan
    return intervals, storage, loss, spread


def sweep_survival():
    """Probability of escaping fire entirely, over time and intensification."""
    times = np.linspace(0.0, HORIZON, 60)
    factors = np.linspace(1.0, 5.0, 45)
    surface = np.zeros((factors.size, times.size))
    for i, factor in enumerate(factors):
        hz = hazard_for(70.0, intensification=factor)
        edges = np.linspace(0.0, HORIZON, N_BLOCK + 1)
        for k, t in enumerate(times):
            grid = np.unique(np.concatenate([edges[edges < t], [t]]))
            surface[i, k] = float(np.exp(-hz.cumulative(grid)[-1]))
    return times, factors, surface


def sweep_ceiling(cmodel):
    """Single-measure confidence ceiling over return interval and severity."""
    intervals = np.linspace(30.0, 200.0, NG)
    replacing = np.linspace(0.05, 0.95, NG)
    ceiling = np.zeros((NG, NG))
    for i, p in enumerate(replacing):
        for j, phi in enumerate(intervals):
            store = store_for(150.0, hazard_for(phi, stand_replacing=p))
            mu = store.mean_cooling(cmodel, HORIZON, N_BLOCK)
            var = store.variance_cooling(cmodel, HORIZON, N_BLOCK)
            sd = np.sqrt(max(var, 0.0))
            k = mu / sd if sd > 0 else np.inf
            ceiling[i, j] = k * k / (1.0 + k * k)
    return intervals, replacing, ceiling


def sweep_horizon(cmodel):
    """Mean loss against the accounting horizon, for four fire regimes."""
    horizons = np.linspace(20.0, 300.0, 15)
    regimes = (50.0, 70.0, 110.0, 150.0)
    curves = {}
    for phi in regimes:
        hz = hazard_for(phi)
        row = []
        for th in horizons:
            store = store_for(150.0, hz)
            det = store.deterministic_cooling(cmodel, float(th))
            mu = store.mean_cooling(cmodel, float(th), N_BLOCK)
            row.append(1.0 - mu / det if det > 0 else np.nan)
        curves[phi] = np.array(row)
    return horizons, curves


def sweep_signal_to_noise(cmodel):
    """Signal-to-noise ratio of one store, by return interval and severity."""
    intervals = np.linspace(30.0, 200.0, 14)
    severities = (0.20, 0.35, 0.50, 0.80)
    curves = {}
    for p in severities:
        row = []
        for phi in intervals:
            store = store_for(150.0, hazard_for(phi, stand_replacing=p))
            mu = store.mean_cooling(cmodel, HORIZON, N_BLOCK)
            sd = np.sqrt(max(store.variance_cooling(cmodel, HORIZON, N_BLOCK), 0.0))
            row.append(mu / sd if sd > 0 else np.nan)
        curves[p] = np.array(row)
    return intervals, curves


def main():
    configure()
    cmodel = climate()

    fig = plt.figure(figsize=(DOUBLE, 0.64 * DOUBLE))
    gs = fig.add_gridspec(
        2, 3, hspace=0.50, wspace=0.58, left=0.070, right=0.975,
        top=0.940, bottom=0.095,
    )
    ax_a = fig.add_subplot(gs[0, 0])
    ax_b = fig.add_subplot(gs[0, 1])
    ax_c = fig.add_subplot(gs[0, 2], projection="3d")
    ax_d = fig.add_subplot(gs[1, 0])
    ax_e = fig.add_subplot(gs[1, 1])
    ax_f = fig.add_subplot(gs[1, 2])

    # ------------------------------------------------------------- panels a,b
    intervals, storage, loss, spread = sweep_loss_and_spread(cmodel)
    X, Y = np.meshgrid(intervals, storage)

    cf_a = ax_a.contourf(
        X, Y, 100.0 * loss, levels=14, cmap="inferno_r", antialiased=True
    )
    cs_a = ax_a.contour(
        X, Y, 100.0 * loss, levels=cf_a.levels[::3], colors="white",
        linewidths=0.45, alpha=0.75,
    )
    ax_a.clabel(cs_a, inline=True, fontsize=5.0, fmt="%.0f")
    ax_a.set_xlabel("fire return interval (yr)")
    ax_a.set_ylabel("mean storage time (yr)")
    cb_a = fig.colorbar(cf_a, ax=ax_a, pad=0.02, fraction=0.046)
    cb_a.set_label("credit lost to fire (%)", fontsize=6.2)
    cb_a.ax.tick_params(labelsize=5.6)
    panel_label(ax_a, "a", dy=1.10)

    cf_b = ax_b.contourf(
        X, Y, spread, levels=14, cmap="viridis", antialiased=True
    )
    cs_b = ax_b.contour(
        X, Y, spread, levels=cf_b.levels[::3], colors="white",
        linewidths=0.45, alpha=0.75,
    )
    ax_b.clabel(cs_b, inline=True, fontsize=5.0, fmt="%.2f")
    ax_b.set_xlabel("fire return interval (yr)")
    ax_b.set_ylabel("mean storage time (yr)")
    cb_b = fig.colorbar(cf_b, ax=ax_b, pad=0.02, fraction=0.046)
    cb_b.set_label("spread $\\sigma/\\mu$", fontsize=6.2)
    cb_b.ax.tick_params(labelsize=5.6)
    panel_label(ax_b, "b", dy=1.10)

    # --------------------------------------------------------------- panel c
    times, factors, surface = sweep_survival()
    T, F = np.meshgrid(times, factors)
    ax_c.plot_surface(
        T, F, surface, cmap=mpl_cm.magma, linewidth=0.0, antialiased=True,
        rcount=45, ccount=60, alpha=0.96,
    )
    ax_c.contour(
        T, F, surface, zdir="z", offset=0.0, levels=8,
        cmap=mpl_cm.magma, linewidths=0.5,
    )
    ax_c.set_xlabel("time (yr)", labelpad=-3.0, fontsize=6.0)
    ax_c.set_ylabel("intensification", labelpad=-3.0, fontsize=6.0)
    # The z label sits clear of its tick labels, which a 3D axis does not do
    # by default at this size.
    ax_c.set_zlabel("escape probability", labelpad=-3.0, fontsize=6.0,
                    rotation=90)
    ax_c.set_zlim(0.0, 1.0)
    ax_c.set_zticks([0.0, 0.5, 1.0])
    ax_c.set_yticks([1, 2, 3, 4, 5])
    ax_c.set_xticks([0, 50, 100])
    ax_c.view_init(elev=22.0, azim=-126.0)
    ax_c.tick_params(labelsize=5.2, pad=-2.0)
    ax_c.set_box_aspect((1.25, 1.05, 0.80))
    # A 3D axis leaves a wide empty margin inside its cell, so the drawing is
    # grown into that margin instead of leaving the panel visually smaller
    # than its neighbours.
    pos = ax_c.get_position()
    ax_c.set_position([
        pos.x0 + 0.004,
        pos.y0 - 0.045,
        pos.width * 1.10,
        pos.height * 1.20,
    ])
    for pane in (ax_c.xaxis, ax_c.yaxis, ax_c.zaxis):
        pane.pane.set_alpha(0.06)
        pane.pane.set_edgecolor("none")
    # A 3D axis takes its in-plane text through text2D, so the shared panel
    # label helper does not apply here. The style matches it.
    ax_c.text2D(
        -0.02, 1.01, "c", transform=ax_c.transAxes,
        fontsize=8.5, fontweight="bold", va="top", ha="left",
    )

    # --------------------------------------------------------------- panel d
    intervals_d, replacing, ceiling = sweep_ceiling(cmodel)
    Xd, Yd = np.meshgrid(intervals_d, replacing)
    # The ceiling stays high over the whole plane, so the colour scale is set
    # to the range the data occupy. A scale running from zero would render the
    # structure as a single flat tone.
    lo = float(np.floor(np.nanmin(ceiling) * 50.0) / 50.0)
    cf_d = ax_d.contourf(
        Xd, Yd, ceiling, levels=np.linspace(lo, 1.0, 21), cmap="cividis",
        antialiased=True,
    )
    cs_d = ax_d.contour(
        Xd, Yd, ceiling, levels=[0.75, 0.85, 0.9, 0.95, 0.98],
        colors="white", linewidths=0.55,
    )
    ax_d.clabel(cs_d, inline=True, fontsize=5.0, fmt="%.2f")
    ax_d.set_xlabel("fire return interval (yr)")
    ax_d.set_ylabel("stand-replacing fraction")
    cb_d = fig.colorbar(cf_d, ax=ax_d, pad=0.02, fraction=0.046)
    cb_d.set_label("ceiling $\\eta_{\\max}$", fontsize=6.2)
    cb_d.ax.tick_params(labelsize=5.6)
    panel_label(ax_d, "d", dy=1.10)

    # --------------------------------------------------------------- panel e
    horizons, loss_curves = sweep_horizon(cmodel)
    shades = plt.get_cmap("inferno_r")(np.linspace(0.30, 0.82, 4))
    for colour, (phi, curve) in zip(shades, sorted(loss_curves.items())):
        ax_e.plot(
            horizons, 100.0 * curve, color=colour, lw=1.1,
            label="{:.0f} yr".format(phi),
        )
    ax_e.set_xlabel("accounting horizon (yr)")
    ax_e.set_ylabel("credit lost to fire (%)")
    ax_e.set_xlim(horizons[0], horizons[-1])
    ax_e.legend(
        title="return interval", fontsize=5.3, title_fontsize=5.5,
        loc="upper left", frameon=False, labelspacing=0.22,
        handlelength=1.1, borderaxespad=0.3,
    )
    light_grid(ax_e)
    spine_style(ax_e)
    panel_label(ax_e, "e", dy=1.10)

    # An embedded panel: the same quantity as a share of its century value,
    # which collapses the four regimes onto one curve and shows that the
    # horizon dependence is a property of the store and not of the regime.
    # The loss curves rise to the upper right, so the inset sits in the lower
    # right where the axes are empty, and it is given an opaque face so that
    # no host curve shows through it.
    ax_e.set_ylim(0.0, 100.0)
    ax_e_in = ax_e.inset_axes([0.555, 0.085, 0.41, 0.31], zorder=6)
    ax_e_in.set_facecolor("white")
    ax_e_in.patch.set_alpha(1.0)
    for colour, (phi, curve) in zip(shades, sorted(loss_curves.items())):
        ref = np.interp(HORIZON, horizons, curve)
        ax_e_in.plot(horizons, curve / ref, color=colour, lw=0.8)
    ax_e_in.axhline(1.0, color=COLOURS["fire"], lw=0.5, ls=(0, (2, 1.6)))
    ax_e_in.set_xlim(horizons[0], horizons[-1])
    ax_e_in.set_xticks([50, 150, 250])
    ax_e_in.set_ylim(0.25, 1.80)
    ax_e_in.set_yticks([0.5, 1.0, 1.5])
    ax_e_in.tick_params(labelsize=4.6, length=1.8, pad=1.0)
    # The label goes inside the inset, so it cannot collide with the curves of
    # the host panel above it.
    # The inset curves climb from lower left to upper right, so the label
    # occupies the empty upper-left corner inside the inset.
    ax_e_in.text(
        0.05, 0.95, "scaled to 100 yr", transform=ax_e_in.transAxes,
        fontsize=4.9, ha="left", va="top",
    )
    spine_style(ax_e_in)

    # --------------------------------------------------------------- panel f
    intervals_f, snr_curves = sweep_signal_to_noise(cmodel)
    tones = plt.get_cmap("viridis")(np.linspace(0.12, 0.84, 4))
    for colour, (p, curve) in zip(tones, sorted(snr_curves.items())):
        ax_f.plot(
            intervals_f, curve, color=colour, lw=1.1,
            label="{:.0f}%".format(100.0 * p),
        )
    for level, tag in ((3.0, "0.90"), (4.36, "0.95")):
        ax_f.axhline(level, color=COLOURS["fire"], lw=0.55, ls=(0, (2, 1.6)))
        ax_f.text(
            intervals_f[-1], level, "$\\eta$=" + tag + " ", fontsize=5.0,
            color=COLOURS["fire"], va="bottom", ha="right",
        )
    ax_f.set_xlabel("fire return interval (yr)")
    ax_f.set_ylabel("signal-to-noise $\\mu/\\sigma$")
    ax_f.set_xlim(intervals_f[0], intervals_f[-1] * 1.02)
    ax_f.legend(
        title="stand-replacing", fontsize=5.3, title_fontsize=5.5,
        loc="upper left", frameon=False, labelspacing=0.22,
        handlelength=1.1, borderaxespad=0.3,
    )
    light_grid(ax_f)
    spine_style(ax_f)
    panel_label(ax_f, "f", dy=1.10)

    for ax in (ax_a, ax_b, ax_d):
        light_grid(ax)
        spine_style(ax)

    save(fig, "fig3_landscape")


if __name__ == "__main__":
    main()
