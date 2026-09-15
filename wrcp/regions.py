"""The two case studies: western United States forest, and boreal Canada.

Each region supplies a fire regime, a set of available measures with costs and
capacity limits, and an emission pathway to be offset. The two are chosen to
contrast, and the contrast is the point of the comparison.

Western United States
---------------------
A mixed-conifer system with a moderate baseline fire return interval, a
mixed-severity regime in which a substantial share of burned area is not
stand-replacing, and a strong projected intensification: the published
projection of the area at risk of carbon reversal expands from a tenth to a
third of the continental United States over the century, which is the
intensification target inverted for :math:`\\beta_T`. Interannual variability
is large but not extreme.

Boreal Canada
-------------
A longer baseline return interval, but a crown-fire regime in which almost all
burned area is stand-replacing, and interannual variability that is extreme:
the 2023 season released close to six times the twenty-year mean. The boreal
case therefore has a *lower* mean hazard and a *higher* variance than the
western United States, which separates the two mechanisms the framework prices.
If the fire premium were driven by the mean alone the western case would
dominate; if by the variance, the boreal case would.

Scale
-----
Emission magnitudes are round numbers of the order of the relevant regional
land sector. The framework is scale free in the deployment variables, so the
portfolio shares, the premium fractions and the confidence ceilings do not
depend on the absolute scale; only the reported absolute costs do. Cost
assumptions are central estimates of the assessed ranges and are listed in
:data:`UNIT_COSTS` with the sensitivity of the conclusions to them quantified
by the Sobol analysis.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Callable, Dict, List, Optional, Sequence, Tuple

import numpy as np

from cipo import (
    Afforestation,
    DelayedPulse,
    EmissionPathway,
    ExponentialRelease,
    LinearCost,
    PermanentRemoval,
    PowerCost,
)
from cipo.pathways import piecewise_linear

from .calibration import (
    FireRecord,
    beta_from_intensification,
    lambda_from_return_interval,
    load_fire_record,
    severity_from_class_fractions,
    spread_from_extreme,
)
from .hazard import (
    NO_FIRE,
    BetaSeverity,
    FireHazard,
    TwoPointSeverity,
    drought_path,
    temperature_path,
)
from .stochastic import FireExposedStore

__all__ = [
    "MT",
    "GT",
    "UNIT_COSTS",
    "Region",
    "WESTERN_US",
    "BOREAL_CANADA",
    "REGIONS",
    "build_stores",
    "western_us_pathway",
    "boreal_canada_pathway",
]

MT = 1.0e9  # kilograms per megatonne
GT = 1.0e12  # kilograms per gigatonne

#: Cost of one tonne of carbon dioxide at peak storage, United States dollars.
#: Central estimates of the assessed ranges; the exposure of the results to
#: each is quantified rather than assumed.
UNIT_COSTS: Dict[str, float] = {
    "afforestation": 30.0,
    "forest_management": 40.0,
    "soil_carbon": 45.0,
    "wood_products": 60.0,
    "biochar": 110.0,
    "beccs": 150.0,
    "daccs": 400.0,
}

#: Share of each measure's stock that fire can reach. Biochar is dominated by a
#: recalcitrant fraction that survives combustion and is mostly below ground,
#: so only a small part of it is exposed; harvested wood products are exposed
#: only through the structures that hold them. Geological storage is not
#: exposed at all.
EXPOSURE: Dict[str, float] = {
    "afforestation": 1.0,
    "forest_management": 1.0,
    "soil_carbon": 0.35,
    "wood_products": 0.10,
    "biochar": 0.08,
    "beccs": 0.0,
    "daccs": 0.0,
}


# ------------------------------------------------------------------- pathways
def western_us_pathway(horizon: float = 100.0) -> EmissionPathway:
    """Land-sector and agricultural emissions of the western United States.

    Carbon dioxide from land use, methane from livestock and nitrous oxide from
    managed soils, declining by about a fifth over seventy-five years under
    continued productivity growth, then held. Emissions continue beyond the
    policy horizon, which is the situation an inventory authority faces.
    """
    years = [0.0, 25.0, 50.0, 75.0, 200.0]
    co2 = [90.0 * MT, 84.0 * MT, 77.0 * MT, 72.0 * MT, 72.0 * MT]
    ch4 = [2.1 * MT, 1.95 * MT, 1.8 * MT, 1.7 * MT, 1.7 * MT]
    n2o = [0.08 * MT, 0.075 * MT, 0.07 * MT, 0.065 * MT, 0.065 * MT]
    return EmissionPathway(
        {
            "CO2": piecewise_linear(years, co2),
            "CH4": piecewise_linear(years, ch4),
            "N2O": piecewise_linear(years, n2o),
        },
        "Western United States land sector",
    )


def boreal_canada_pathway(horizon: float = 100.0) -> EmissionPathway:
    """Canada's residual emissions on a national net-zero trajectory.

    Carbon dioxide falls to a residual by mid-century and is then held, with
    methane and nitrous oxide falling to agricultural residuals on the same
    schedule. The retained carbon dioxide residual is what makes this the
    harder case for temporary storage, independently of fire.
    """
    years = [0.0, 25.0, 50.0, 75.0, 200.0]
    co2 = [260.0 * MT, 90.0 * MT, 30.0 * MT, 30.0 * MT, 30.0 * MT]
    ch4 = [2.0 * MT, 1.3 * MT, 1.05 * MT, 1.05 * MT, 1.05 * MT]
    n2o = [0.09 * MT, 0.07 * MT, 0.06 * MT, 0.06 * MT, 0.06 * MT]
    return EmissionPathway(
        {
            "CO2": piecewise_linear(years, co2),
            "CH4": piecewise_linear(years, ch4),
            "N2O": piecewise_linear(years, n2o),
        },
        "Boreal Canada net zero residual",
    )


# -------------------------------------------------------------------- regions
@dataclass
class Region:
    """A calibrated fire regime together with its measures and pathway."""

    name: str
    label: str
    return_interval: float
    delta_T_start: float
    delta_T_end: float
    intensification: float
    severity_classes: Tuple[float, float, float]
    anomaly_ratio: float
    anomaly_return_period: float
    drought_end: float
    caps: Dict[str, float]
    pathway_builder: Callable[[float], EmissionPathway]
    horizon: float = 100.0
    beta_D: float = 0.25
    record_path: Optional[str] = None
    cost_exponent: float = 1.6

    # ------------------------------------------------------------- calibration
    def __post_init__(self) -> None:
        m1, m2 = severity_from_class_fractions(*self.severity_classes)
        self._severity_moments = (m1, m2)
        self.record = load_fire_record(
            self.name,
            self.record_path,
            fallback_anomaly=self.anomaly_ratio,
        )
        self.anomaly_ratio = float(self.record.max_anomaly_ratio)

    @property
    def severity(self) -> TwoPointSeverity:
        """A severity law with the calibrated moments.

        The two-point family is used because it reproduces an arbitrary pair of
        moments on the unit interval exactly, and because the two points are
        interpretable: a stand-replacing fire and a partial burn. The robust
        constraint sees only the moments, so no generality is lost.
        """
        m1, m2 = self._severity_moments
        # Solve p * 1 + (1-p) * q = m1 and p + (1-p) q^2 = m2 for (p, q).
        # Eliminating gives a quadratic in q with the admissible root below.
        if m2 <= m1 * m1 + 1e-12:
            return TwoPointSeverity(p=0.0, partial=m1, full=1.0)
        # p = (m1 - q) / (1 - q), and substituting into the second moment
        # yields q^2 - q(1 + m1 - m2 / m1 * 0) ... solved numerically for
        # robustness against the degenerate cases.
        from scipy.optimize import brentq

        def resid(q: float) -> float:
            p = (m1 - q) / (1.0 - q)
            return p + (1.0 - p) * q * q - m2

        lo, hi = 1e-9, max(m1 - 1e-9, 2e-9)
        try:
            q = float(brentq(resid, lo, hi, xtol=1e-14))
        except ValueError:
            q = float(m1)
        p = float((m1 - q) / (1.0 - q))
        return TwoPointSeverity(p=float(np.clip(p, 0.0, 1.0)), partial=q, full=1.0)

    @property
    def beta_T(self) -> float:
        """Climate sensitivity of fire frequency, inverted from the target.

        The inversion uses the temperature and drought paths *as evaluated* at
        the ends of the horizon, not the nominal asymptotes. A saturating
        anomaly does not reach its asymptote within the horizon, so inverting
        against ``delta_T_end`` would leave the realised intensification short
        of the target by the saturation deficit, which for the paths used here
        is about fourteen per cent. Evaluating the path makes the calibration
        exact by construction, and
        :meth:`wrcp.hazard.FireHazard.intensification` checks it.
        """
        temp, drought = self.temperature(), self.drought()
        return beta_from_intensification(
            self.intensification,
            float(np.asarray(temp(0.0)).ravel()[0]),
            float(np.asarray(temp(self.horizon)).ravel()[0]),
            beta_D=self.beta_D,
            drought_start=float(np.asarray(drought(0.0)).ravel()[0]),
            drought_end=float(np.asarray(drought(self.horizon)).ravel()[0]),
        )

    @property
    def weather_spread(self) -> float:
        return spread_from_extreme(self.anomaly_ratio, self.anomaly_return_period)

    def temperature(self) -> Callable:
        return temperature_path(
            self.delta_T_start,
            self.delta_T_end,
            tau=0.55 * self.horizon,
        )

    def drought(self) -> Callable:
        return drought_path(0.0, self.drought_end, self.horizon)

    def hazard_template(self, exposure: float = 1.0) -> FireHazard:
        """The regional hazard, scaled by a measure's exposed fraction."""
        return FireHazard(
            lambda0=lambda_from_return_interval(self.return_interval),
            beta_T=self.beta_T,
            beta_D=self.beta_D,
            severity=self.severity,
            temperature=self.temperature(),
            drought=self.drought(),
            region=self.name,
            exposure=float(exposure),
            name="{}_hazard".format(self.name),
        )

    def pathway(self) -> EmissionPathway:
        return self.pathway_builder(self.horizon)

    def summary(self) -> Dict[str, object]:
        return {
            "region": self.name,
            "return_interval_yr": self.return_interval,
            "lambda0_per_yr": lambda_from_return_interval(self.return_interval),
            "beta_T_per_K": self.beta_T,
            "intensification": self.intensification,
            "severity_m1": self.severity.m1,
            "severity_m2": self.severity.m2,
            "weather_spread": self.weather_spread,
            "anomaly_ratio": self.anomaly_ratio,
        }


#: Mixed-conifer western United States. Return interval from fire-scar
#: reconstructions for the mixed-severity regime; severity classes from the
#: Monitoring Trends in Burn Severity archive; intensification from the
#: projected expansion of area at risk of carbon reversal, from a tenth to a
#: third of the continental United States.
WESTERN_US = Region(
    name="western_us",
    label="Western United States",
    return_interval=70.0,
    delta_T_start=1.4,
    delta_T_end=4.0,
    intensification=3.3,
    severity_classes=(0.32, 0.34, 0.34),
    anomaly_ratio=3.5,
    anomaly_return_period=100.0,
    drought_end=1.2,
    caps={
        # Land-limited measures are capped by the regional land base; the two
        # geologically stored measures are industrial rather than land limited,
        # so their ceilings reflect plausible build-out rather than acreage.
        "afforestation": 6.0 * GT,
        "forest_management": 5.0 * GT,
        "soil_carbon": 4.0 * GT,
        "wood_products": 2.5 * GT,
        "biochar": 3.0 * GT,
        "beccs": 5.0 * GT,
        "daccs": 5.0 * GT,
    },
    pathway_builder=western_us_pathway,
)

#: Boreal Canada. Longer baseline return interval, near-total stand
#: replacement when fire does arrive, and the largest interannual anomaly in
#: the record: the 2023 season at close to six times the twenty-year mean.
BOREAL_CANADA = Region(
    name="boreal_canada",
    label="Boreal Canada",
    return_interval=110.0,
    delta_T_start=1.9,
    delta_T_end=5.6,
    intensification=2.6,
    severity_classes=(0.82, 0.13, 0.05),
    anomaly_ratio=6.0,
    anomaly_return_period=100.0,
    drought_end=1.0,
    caps={
        "afforestation": 5.0 * GT,
        "forest_management": 7.0 * GT,
        "soil_carbon": 3.2 * GT,
        "wood_products": 3.2 * GT,
        "biochar": 2.4 * GT,
        "beccs": 5.0 * GT,
        "daccs": 5.0 * GT,
    },
    pathway_builder=boreal_canada_pathway,
)

REGIONS: Dict[str, Region] = {
    "western_us": WESTERN_US,
    "boreal_canada": BOREAL_CANADA,
}


# ------------------------------------------------------------------- measures
def _cost(key: str, region: Region, cap: float) -> PowerCost:
    """Convex cost rising with scale towards the capacity limit.

    Strict convexity is what makes an interior multi-measure optimum generic
    rather than exceptional, and so what makes the equalised marginal-cost
    condition of :meth:`wrcp.robust.FirePortfolio.kkt_report` testable. The
    reference scale is the capacity limit, so the quoted unit cost is the
    marginal cost at full deployment.
    """
    return PowerCost(
        unit_cost=UNIT_COSTS[key] / 1000.0,  # per kg of carbon dioxide
        gamma=region.cost_exponent,
        reference_scale=float(cap),
    )


def build_stores(region: Region) -> List[FireExposedStore]:
    """The measures available in a region, each coupled to its fire exposure.

    The families are those of :mod:`cipo.profiles`, so the deterministic
    behaviour is exactly that of the reference framework, and the only addition
    is the hazard each one faces. Afforestation and improved forest management
    hold carbon in standing biomass and are fully exposed; soil carbon is
    partly protected; wood products and biochar are largely protected;
    bioenergy with capture and direct air capture store geologically and are
    not exposed at all.
    """
    caps = region.caps
    out: List[FireExposedStore] = []

    out.append(
        FireExposedStore(
            Afforestation(
                tau_growth=28.0,
                tau_disturbance=140.0,
                cost_model=_cost("afforestation", region, caps["afforestation"]),
                max_scale=caps["afforestation"],
                name="afforestation",
                label="Afforestation",
            ),
            region.hazard_template(EXPOSURE["afforestation"]),
        )
    )
    out.append(
        FireExposedStore(
            Afforestation(
                tau_growth=45.0,
                tau_disturbance=220.0,
                cost_model=_cost(
                    "forest_management", region, caps["forest_management"]
                ),
                max_scale=caps["forest_management"],
                name="forest_management",
                label="Improved forest management",
            ),
            region.hazard_template(EXPOSURE["forest_management"]),
        )
    )
    out.append(
        FireExposedStore(
            ExponentialRelease(
                tau=45.0,
                cost_model=_cost("soil_carbon", region, caps["soil_carbon"]),
                max_scale=caps["soil_carbon"],
                name="soil_carbon",
                label="Soil carbon",
            ),
            region.hazard_template(EXPOSURE["soil_carbon"]),
        )
    )
    out.append(
        FireExposedStore(
            DelayedPulse(
                tau=55.0,
                cost_model=_cost("wood_products", region, caps["wood_products"]),
                max_scale=caps["wood_products"],
                name="wood_products",
                label="Harvested wood products",
            ),
            region.hazard_template(EXPOSURE["wood_products"]),
        )
    )
    out.append(
        FireExposedStore(
            ExponentialRelease(
                tau=350.0,
                cost_model=_cost("biochar", region, caps["biochar"]),
                max_scale=caps["biochar"],
                name="biochar",
                label="Biochar",
            ),
            region.hazard_template(EXPOSURE["biochar"]),
        )
    )
    out.append(
        FireExposedStore(
            PermanentRemoval(
                cost_model=_cost("beccs", region, caps["beccs"]),
                max_scale=caps["beccs"],
                name="beccs",
                label="BECCS, geological storage",
            ),
            NO_FIRE,
        )
    )
    out.append(
        FireExposedStore(
            PermanentRemoval(
                cost_model=_cost("daccs", region, caps["daccs"]),
                max_scale=caps["daccs"],
                name="daccs",
                label="Direct air capture",
            ),
            NO_FIRE,
        )
    )
    return out
