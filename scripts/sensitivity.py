"""Sensitivity of the headline results to the choices the calibration leaves open.

Three inputs are not pinned by observation, and a reader is entitled to know how
far the conclusions move when each is varied over its plausible range.

``coupling``
    How severities at different measures co-vary within one shared fire year.
    The construction bounds this: independent severities give the smallest
    cross-measure covariance the model admits, and the comonotone coupling,
    which attains the Cauchy-Schwarz bound, gives the largest. The reported
    results sit at the midpoint. Note that the largest covariance gives the
    smallest common variance *share*, because it raises the total. This sweep
    reports all three.

``severity``
    The severity moments are held fixed while the arrival rate trends with
    climate. If fires also burn more severely as the century warms, the
    penalty grows. This sweep raises the stand-replacing fraction by a fifth
    and reports the effect, which bounds that omission.

``cost curvature``
    Deployment cost rises as ``c_i alpha^gamma``. The portfolio composition
    depends on how sharply marginal cost rises, so gamma is swept around the
    adopted value.

Each row is a complete re-solve, so the table is a statement about the
conclusions and not about one intermediate quantity.
"""

from __future__ import annotations

import copy

import numpy as np
import pandas as pd

from common import save_table, write_values
from scenarios import climate

from wrcp.hazard import TwoPointSeverity
from wrcp.regions import REGIONS, build_stores
from wrcp.robust import FirePortfolio

ETA = 0.9

# Lighter quadrature than the headline runs: this is a sweep over many
# portfolios, and the sensitivity is a difference between rows, which the
# coarser grid resolves.
FAST = dict(n_block=8, n_weather=5)


def build(region, sharing=0.5, severity_scale=1.0, gamma=None):
    """One portfolio under a stated set of the three open choices."""
    stores = build_stores(region)
    if severity_scale != 1.0:
        scaled = []
        for st in stores:
            if st.hazard.is_inert:
                scaled.append(st)
                continue
            sev = st.hazard.severity
            p = float(np.clip(getattr(sev, "p", 0.3) * severity_scale, 0.0, 1.0))
            hz = copy.copy(st.hazard)
            hz.severity = TwoPointSeverity(
                p=p,
                partial=getattr(sev, "partial", 0.35),
                full=getattr(sev, "full", 1.0),
            )
            st2 = copy.copy(st)
            st2.hazard = hz
            scaled.append(st2)
        stores = scaled
    if gamma is not None:
        import dataclasses
        for st in stores:
            cost = getattr(st.intervention, "cost_model", None)
            if cost is None or not hasattr(cost, "gamma"):
                continue
            # Both the cost model and the intervention are frozen dataclasses,
            # so the curvature is changed by rebuilding rather than assigning.
            new_cost = dataclasses.replace(cost, gamma=float(gamma))
            object.__setattr__(st.intervention, "cost_model", new_cost)
    return FirePortfolio(
        climate(), stores, region.pathway(), horizon=region.horizon,
        sharing=sharing, weather_spread=region.weather_spread, **FAST
    )


def summarise(fp):
    """Common variance share and cost multiple at the headline confidence."""
    m = fp.moments()
    n = m.mean.size
    w = np.ones(n) / n
    tot = float(w @ m.covariance @ w)
    common = float(w @ m.common @ w) / tot if tot > 0 else np.nan
    det = fp.deterministic_solution()
    res = fp.solve(ETA)
    ratio = float(res.cost / det.cost) if (res.feasible and det.cost > 0) else np.nan
    exposed = np.array([not s.hazard.is_inert for s in fp.stores])
    share = np.nan
    if res.feasible and np.all(np.isfinite(res.alpha)):
        delivered = m.mean * res.alpha
        if delivered.sum() > 0:
            share = float(delivered[exposed].sum() / delivered.sum())
    return common, ratio, share


def main():
    rows = []
    for name, region in REGIONS.items():
        for label, kw in (
            ("coupling: independent", dict(sharing=0.0)),
            ("coupling: midpoint (adopted)", dict(sharing=0.5)),
            ("coupling: comonotone", dict(sharing=1.0)),
            ("severity: stand-replacing +20%", dict(severity_scale=1.2)),
            ("cost curvature: gamma = 1.3", dict(gamma=1.3)),
            ("cost curvature: gamma = 1.6 (adopted)", dict(gamma=1.6)),
            ("cost curvature: gamma = 2.0", dict(gamma=2.0)),
        ):
            fp = build(region, **kw)
            common, ratio, share = summarise(fp)
            rows.append(
                {
                    "region": region.label,
                    "case": label,
                    "common_variance_share": common,
                    "cost_multiple_at_0.9": ratio,
                    "exposed_share_at_0.9": share,
                }
            )
            print("  {:22s} {:38s} common {:.3f}  cost {:.2f}x  exposed {:.3f}".format(
                region.label, label, common, ratio, share))

    df = pd.DataFrame(rows)
    save_table(df, "sensitivity")

    from make_tables import tex_escape, write_tex

    # write_tex passes cells through unescaped so that captions may carry
    # mathematics, so the free-text columns are escaped here. The severity
    # case label contains a per cent sign, which LaTeX would otherwise read
    # as a comment and swallow the rest of the row.
    pretty = pd.DataFrame(
        {
            "Region": [tex_escape(v) for v in df["region"]],
            "Case": [tex_escape(v) for v in df["case"]],
            "Common share": df["common_variance_share"],
            "Cost at $\\eta=0.9$": df["cost_multiple_at_0.9"],
            "Exposed share": df["exposed_share_at_0.9"],
        }
    )
    write_tex(
        pretty,
        "sensitivity",
        "\\textbf{Sensitivity of the conclusions to the three choices the "
        "calibration leaves open.} Each row is a complete re-solve. "
        "``Common share'' is the fraction of the variance of an equally "
        "weighted portfolio that spreading across sites leaves, ``Cost'' is "
        "the cost of the portfolio at nine-tenths confidence as a multiple of "
        "the fire-free cost, and ``Exposed share'' is the fraction of "
        "delivered cooling supplied by measures fire can reach. The coupling "
        "rows span the whole family the construction admits, from independent "
        "severities to the comonotone bound. The severity row raises the "
        "stand-replacing fraction by a fifth, which bounds the effect of "
        "holding severity fixed while the arrival rate trends. The curvature "
        "rows vary the exponent of the deployment cost curve.",
        "tab:sensitivity",
        float_format="%.3g",
    )

    vals = {}
    for name, region in REGIONS.items():
        sub = df[df.region == region.label]
        def pick(case, col):
            hit = sub[sub.case == case][col]
            return float(hit.iloc[0]) if len(hit) else np.nan
        vals["common_share_independent_" + name] = pick("coupling: independent",
                                                        "common_variance_share")
        vals["common_share_comonotone_" + name] = pick("coupling: comonotone",
                                                       "common_variance_share")
        vals["cost_multiple_severity_up_" + name] = pick(
            "severity: stand-replacing +20%", "cost_multiple_at_0.9")
        vals["cost_multiple_gamma_low_" + name] = pick(
            "cost curvature: gamma = 1.3", "cost_multiple_at_0.9")
        vals["cost_multiple_gamma_high_" + name] = pick(
            "cost curvature: gamma = 2.0", "cost_multiple_at_0.9")
    write_values(vals)
    print()
    print("  sensitivity table written")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
