"""Calibration of the fire hazard against the observational record.

Every number the hazard model needs is fixed here, from a published source, and
the inversion that turns a published statistic into a model parameter is
written out so it can be checked or replaced. No parameter in this module is
fitted to the results of this study.

What is calibrated from data, and what is not
---------------------------------------------
This distinction is important enough to state plainly, because the framework's
conclusions inherit it.

The **baseline event rate** :math:`\\lambda_{i,0}` comes from published fire
return intervals for the relevant forest type, which are themselves derived
from fire-scar and stand-age reconstructions. These are well constrained and
regionally specific.

The **severity distribution** comes from the published high-severity area
fractions of burn-severity classifications, which give the two moments the
robust constraint needs.

The **climate sensitivity of fire frequency** :math:`\\beta_T` is not observed
directly. It is obtained by inverting a published projection of how much fire
risk grows over the century, which is the only form in which the quantity is
actually reported. :func:`beta_from_intensification` performs that inversion,
so the assumption is a single stated number, the intensification factor, rather
than a coefficient with no observational counterpart.

The **year-to-year spread** is obtained from the largest anomaly in the
regional record and an assumed return period for it, again by inversion.
:func:`spread_from_extreme` performs it. This is the weakest link in the
calibration, and the sensitivity of every result to it is reported rather than
assumed away.

Gridded products
----------------
The functions below take published *aggregates*. Full gridded calibration
against the Global Fire Emissions Database, the Monitoring Trends in Burn
Severity archive and the Canadian National Fire Database requires those
products, which are not redistributed with this code.
:func:`load_fire_record` reads a table of annual regional fire emissions if one
is supplied and re-derives the aggregates from it, so that substituting the real
product changes the calibration and nothing else. When no file is given the
published aggregates recorded here are used, and
:attr:`FireRecord.source` says which of the two happened.
"""

from __future__ import annotations

import os
from dataclasses import dataclass, field
from typing import Dict, Optional, Sequence, Tuple

import numpy as np

__all__ = [
    "PUBLISHED",
    "FireRecord",
    "load_fire_record",
    "beta_from_intensification",
    "spread_from_extreme",
    "severity_from_class_fractions",
    "lambda_from_return_interval",
    "calibration_table",
]


#: Published aggregates used when no gridded product is supplied. Each entry
#: records the value, its units and the source, so the provenance of every
#: calibrated parameter is traceable from the code alone.
PUBLISHED: Dict[str, Dict[str, object]] = {
    "gfed5_global_carbon": {
        "value": 3.4,
        "units": "PgC yr^-1",
        "period": "2002-2022",
        "source": "Global Fire Emissions Database version 5",
        "note": "about 70 per cent above the previous estimate of 2.0, the "
        "increase attributed to improved detection of small fires",
    },
    "gfed5_previous_carbon": {
        "value": 2.0,
        "units": "PgC yr^-1",
        "source": "GFED4s, for comparison",
    },
    "gfed5_burned_area_trend": {
        "value": -1.2,
        "units": "per cent yr^-1",
        "period": "2002-2022",
        "source": "Global Fire Emissions Database version 5",
    },
    "gfed5_emission_trend": {
        "value": -0.3,
        "units": "per cent yr^-1",
        "period": "2002-2022",
        "source": "Global Fire Emissions Database version 5",
        "note": "not significant; fuel consumption rose as fire shifted into "
        "higher-biomass forest, offsetting the fall in burned area",
    },
    "global_co2_review": {
        "value": 8.72,
        "uncertainty": 0.67,
        "units": "PgCO2 yr^-1",
        "period": "2001-2020",
        "source": "review of global wildfire carbon dioxide emissions",
        "note": "83 per cent from tropical ecosystems",
    },
    "canada_2023_emission": {
        "value": 2.0,
        "units": "GtCO2",
        "source": "2023 Canadian fire season",
        "note": "close to six times the twenty-year average, and over four "
        "times Canada's annual fossil fuel emissions",
    },
    "canada_anomaly_ratio": {
        "value": 6.0,
        "units": "dimensionless",
        "source": "2023 Canadian fire season against the twenty-year mean",
    },
    "conus_reversal_risk_historical": {
        "value": 0.10,
        "units": "fraction of area",
        "source": "projected carbon reversal risk under historical fire models",
    },
    "conus_reversal_risk_updated": {
        "value": 0.33,
        "units": "fraction of area",
        "source": "projected carbon reversal risk to 2100 under updated models",
        "note": "the ratio of these two is the intensification target for the "
        "western United States case study",
    },
}


@dataclass
class FireRecord:
    """Aggregates of a regional fire record, however they were obtained."""

    region: str
    mean_emission: float
    coefficient_of_variation: float
    max_anomaly_ratio: float
    n_years: int
    source: str = "published aggregates"
    years: Optional[np.ndarray] = None
    emissions: Optional[np.ndarray] = None

    @property
    def from_gridded(self) -> bool:
        return self.emissions is not None

    def summary(self) -> Dict[str, object]:
        return {
            "region": self.region,
            "mean_emission": self.mean_emission,
            "coefficient_of_variation": self.coefficient_of_variation,
            "max_anomaly_ratio": self.max_anomaly_ratio,
            "n_years": self.n_years,
            "source": self.source,
            "from_gridded": self.from_gridded,
        }


def load_fire_record(
    region: str,
    path: Optional[str] = None,
    year_column: str = "year",
    value_column: str = "emission",
    fallback_cv: float = 0.45,
    fallback_anomaly: float = 4.0,
    fallback_mean: float = float("nan"),
    n_years: int = 21,
) -> FireRecord:
    """Aggregates of annual regional fire emissions.

    If ``path`` points to a readable table with a year column and an emission
    column, the mean, the coefficient of variation and the largest anomaly are
    computed from it. Otherwise the supplied fallbacks are used and the source
    is marked accordingly, so that a calibration run always states whether it
    saw data.
    """
    if path and os.path.exists(path):
        import pandas as pd

        df = pd.read_csv(path)
        y = np.asarray(df[year_column], dtype=float)
        e = np.asarray(df[value_column], dtype=float)
        mean = float(np.mean(e))
        return FireRecord(
            region=region,
            mean_emission=mean,
            coefficient_of_variation=float(np.std(e, ddof=1) / mean) if mean else np.nan,
            max_anomaly_ratio=float(np.max(e) / mean) if mean else np.nan,
            n_years=int(y.size),
            source="gridded record at {}".format(os.path.basename(path)),
            years=y,
            emissions=e,
        )
    return FireRecord(
        region=region,
        mean_emission=fallback_mean,
        coefficient_of_variation=fallback_cv,
        max_anomaly_ratio=fallback_anomaly,
        n_years=int(n_years),
        source="published aggregates (no gridded product supplied)",
    )


def lambda_from_return_interval(interval_years: float) -> float:
    """Baseline event rate from a fire return interval, in yr^-1."""
    if interval_years <= 0.0:
        raise ValueError("fire return interval must be positive")
    return float(1.0 / interval_years)


def beta_from_intensification(
    intensification: float,
    delta_T_start: float,
    delta_T_end: float,
    beta_D: float = 0.0,
    drought_start: float = 0.0,
    drought_end: float = 0.0,
) -> float:
    """Invert a projected intensification factor for ``beta_T``.

    The intensification is the ratio of the fire intensity at the end of the
    horizon to its present-day value. With the normalised intensity of
    :class:`wrcp.hazard.FireHazard` that ratio is
    :math:`\\exp(\\beta_T \\Delta(\\Delta T) + \\beta_D \\Delta D)`, so

    .. math::
        \\beta_T = \\frac{\\ln(\\mathrm{intensification})
                   - \\beta_D\\,[D(T_{\\mathrm H}) - D(0)]}
                  {\\Delta T(T_{\\mathrm H}) - \\Delta T(0)} .

    This is the only route by which the climate sensitivity of fire frequency
    enters, and it converts an unobservable coefficient into a published
    projection. The projected expansion of United States area at risk of
    carbon reversal, from a tenth to a third, gives an intensification near
    3.3 for the western United States case study.
    """
    d_temp = float(delta_T_end - delta_T_start)
    if abs(d_temp) < 1e-12:
        raise ValueError("temperature must change over the horizon")
    d_drought = float(drought_end - drought_start)
    return float(
        (np.log(float(intensification)) - beta_D * d_drought) / d_temp
    )


def spread_from_extreme(
    anomaly_ratio: float, return_period_years: float = 100.0
) -> float:
    """Log-normal spread implied by an observed extreme fire year.

    A year whose emissions are ``anomaly_ratio`` times the mean, treated as an
    event of the stated return period, fixes the spread of the fire-weather
    multiplier. The multiplier used in :mod:`wrcp.stochastic` is
    :math:`\\exp(\\sigma x - \\tfrac12\\sigma^2)` with :math:`x` standard
    normal, normalised so that its *mean* is one; a calibrated baseline rate
    must not be shifted by the act of adding variability around it. Writing
    :math:`z` for the upper standard-normal quantile at exceedance
    :math:`1/P`, matching the multiplier at that quantile to the observed ratio
    :math:`r` gives the quadratic

    .. math::
        \\tfrac12 \\sigma^2 - z\\sigma + \\ln r = 0 ,
        \\qquad
        \\sigma = z - \\sqrt{z^2 - 2\\ln r} ,

    the smaller root being the one that is continuous at :math:`r \\to 1`. The
    ratio attainable at a given quantile is bounded by
    :math:`e^{z^2/2}`, reached at :math:`\\sigma = z`; an observed ratio above
    that bound cannot be reproduced by any mean-one log-normal at that return
    period, and the bounding value is returned with the shortfall implicit.

    The 2023 Canadian season, close to six times the twenty-year mean, treated
    as a one-in-a-century event, gives a spread near 0.97; treated as
    one-in-fifty it gives 1.13. This is the least constrained number in the
    calibration, and the sensitivity of every result to it is reported rather
    than suppressed.
    """
    from scipy.stats import norm

    r = float(anomaly_ratio)
    z = float(norm.ppf(1.0 - 1.0 / float(return_period_years)))
    if r <= 1.0 or z <= 0.0:
        return 0.0
    disc = z * z - 2.0 * np.log(r)
    if disc <= 0.0:
        return float(z)
    return float(z - np.sqrt(disc))


def severity_from_class_fractions(
    high: float,
    moderate: float,
    low: float,
    loss_high: float = 1.0,
    loss_moderate: float = 0.45,
    loss_low: float = 0.12,
) -> Tuple[float, float]:
    """Severity moments from burn-severity class area fractions.

    Burn-severity products classify burned area rather than carbon loss, so a
    loss fraction must be attached to each class. The defaults take
    high-severity fire to kill the overstorey outright, moderate severity to
    remove about half the stock and low severity to remove about a tenth,
    which is the range reported for mixed-conifer systems. Returns
    :math:`(m_1, m_2)`.
    """
    frac = np.array([high, moderate, low], dtype=float)
    total = float(np.sum(frac))
    if total <= 0.0:
        raise ValueError("class fractions must sum to a positive number")
    frac = frac / total
    loss = np.array([loss_high, loss_moderate, loss_low], dtype=float)
    return float(frac @ loss), float(frac @ loss ** 2)


def calibration_table(regions: Sequence[object]):
    """Every calibrated parameter of every region, as one auditable table."""
    import pandas as pd

    rows = []
    for r in regions:
        rows.append(
            {
                "region": r.name,
                "fire_return_interval_yr": r.return_interval,
                "lambda0_per_yr": r.hazard_template().lambda0,
                "beta_T_per_K": r.beta_T,
                "beta_D": r.beta_D,
                "delta_T_start_K": r.delta_T_start,
                "delta_T_end_K": r.delta_T_end,
                "intensification_target": r.intensification,
                "severity_m1": r.severity.m1,
                "severity_m2": r.severity.m2,
                "weather_spread": r.weather_spread,
                "anomaly_ratio": r.anomaly_ratio,
                "anomaly_return_period_yr": r.anomaly_return_period,
                "record_source": r.record.source,
            }
        )
    return pd.DataFrame(rows)
