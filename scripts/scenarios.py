"""Case-study driver: solves every scenario once and caches the results.

The figure scripts and the table scripts all need the same solved portfolios,
and each solve costs a few seconds of exact integration. This module solves
each scenario once, writes the results to disk, and hands back the same objects
to every caller, so a full rebuild of the paper solves nothing twice.
"""

from __future__ import annotations

import os
import pickle
from typing import Dict, List, Optional, Sequence

import numpy as np
import pandas as pd

from common import DATA_DIR

from cipo import AR6
from wrcp.regions import BOREAL_CANADA, REGIONS, WESTERN_US, build_stores
from wrcp.robust import FirePortfolio
from wrcp.spike import PeakPortfolio

CONFIDENCES = (0.5, 0.6, 0.7, 0.75, 0.8, 0.85, 0.9, 0.95, 0.975, 0.99)

#: Confidence levels quoted in the text and the tables.
HEADLINE = (0.5, 0.75, 0.9, 0.95)

CACHE = os.path.join(DATA_DIR, "scenarios.pkl")

_MEMO: Dict[str, object] = {}


def climate():
    if "climate" not in _MEMO:
        _MEMO["climate"] = AR6()
    return _MEMO["climate"]


def portfolio(region_name: str, **overrides) -> FirePortfolio:
    """The fire-exposed portfolio of one region, built once."""
    key = "fp:" + region_name + repr(sorted(overrides.items()))
    if key not in _MEMO:
        region = REGIONS[region_name]
        kwargs = dict(
            n_block=10,
            weather_spread=region.weather_spread,
            n_weather=7,
        )
        kwargs.update(overrides)
        _MEMO[key] = FirePortfolio(
            climate(),
            build_stores(region),
            region.pathway(),
            horizon=region.horizon,
            **kwargs,
        )
    return _MEMO[key]


def peak_portfolio(region_name: str, t_max: Optional[float] = None) -> PeakPortfolio:
    """The peak-constrained variant, with a target derived from the pathway.

    The target is set to the peak warming the pathway reaches with no
    intervention, scaled down by a stated factor: a policy that tolerated the
    unmitigated peak would not need a portfolio at all, and one that demanded
    an arbitrary absolute figure would confound the peak question with the
    scale of the case study. The factor is the only free choice and is stated
    with the results.
    """
    key = "peak:" + region_name + repr(t_max)
    if key not in _MEMO:
        region = REGIONS[region_name]
        base = portfolio(region_name)
        if t_max is None:
            t = np.linspace(0.0, region.horizon, 1201)
            unmitigated = float(
                np.max(base.pathway.temperature(climate(), t))
            )
            t_max = 0.55 * unmitigated
        _MEMO[key] = PeakPortfolio(
            climate(),
            build_stores(region),
            region.pathway(),
            region.horizon,
            n_block=8,
            weather_spread=region.weather_spread,
            n_weather=5,
            t_max=float(t_max),
            n_grid=121,
            n_fire=21,
        )
    return _MEMO[key]


def solved(region_name: str) -> Dict[str, object]:
    """Every quantity the figures and tables need, for one region."""
    key = "solved:" + region_name
    if key in _MEMO:
        return _MEMO[key]
    fp = portfolio(region_name)
    region = REGIONS[region_name]
    det = fp.deterministic_solution()
    results = {eta: fp.solve(eta) for eta in CONFIDENCES}
    out = {
        "region": region,
        "portfolio": fp,
        "moments": fp.moments(),
        "deterministic": det,
        "results": results,
        "premium": fp.premium(CONFIDENCES),
        "ceiling": fp.ceiling_report(),
        "required_cooling": fp.required_cooling(),
    }
    _MEMO[key] = out
    return out


def all_solved() -> Dict[str, Dict[str, object]]:
    return {name: solved(name) for name in REGIONS}


def composition_frame(region_name: str) -> pd.DataFrame:
    """Deployment of each measure at each confidence level, in tonnes."""
    data = solved(region_name)
    fp = data["portfolio"]
    rows = []
    det = data["deterministic"]
    rows.append(
        {
            "confidence": np.nan,
            "case": "deterministic",
            "cost": float(det.cost),
            **{n: float(a) for n, a in zip(fp.names, det.alpha)},
        }
    )
    for eta, res in data["results"].items():
        if not res.feasible:
            continue
        rows.append(
            {
                "confidence": float(eta),
                "case": "robust",
                "cost": float(res.cost),
                **{n: float(a) for n, a in zip(fp.names, res.alpha)},
            }
        )
    return pd.DataFrame(rows)


def exposed_share(region_name: str) -> pd.DataFrame:
    """Share of delivered cooling coming from fire-exposed measures."""
    data = solved(region_name)
    fp = data["portfolio"]
    m = data["moments"]
    exposed = np.array([not s.hazard.is_inert for s in fp.stores])
    rows = []
    for eta, res in data["results"].items():
        if not res.feasible:
            continue
        cooling = res.alpha * m.mean
        total = float(np.sum(cooling))
        rows.append(
            {
                "confidence": float(eta),
                "cost": res.cost,
                "exposed_cooling_share": float(np.sum(cooling[exposed]) / total)
                if total > 0
                else np.nan,
                "protected_cooling_share": float(
                    np.sum(cooling[~exposed]) / total
                )
                if total > 0
                else np.nan,
                "total_deployment": float(np.sum(res.alpha)),
            }
        )
    return pd.DataFrame(rows).sort_values("confidence")


if __name__ == "__main__":
    for name in REGIONS:
        data = solved(name)
        print("=" * 78)
        print(REGIONS[name].label)
        print("  required cooling  {:.5g} K yr".format(data["required_cooling"]))
        print("  deterministic cost {:.5g}".format(data["deterministic"].cost))
        print(
            "  ceiling: exposed-only eta_max = {:.4f}".format(
                data["ceiling"]["confidence_max_exposed_only"]
            )
        )
        print(exposed_share(name).to_string(index=False, float_format=lambda v: "%.4g" % v))
