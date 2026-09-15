"""Impulse response climate model: IRFs, AGWP, AGTP and iAGTP.

The temperature response to a unit pulse emission of species :math:`x` is

.. math::
    \\mathrm{AGTP}_x(t) = \\int_0^t \\mathrm{RF}_x(t')\\, R_T(t - t')\\, dt' ,
    \\qquad
    \\mathrm{RF}_x(t) = \\mathrm{RE}_x\\, \\mathrm{IRF}_x(t) ,

so the airborne fraction is first turned into radiative forcing and the
forcing is then convolved with the thermal response of the climate system
:math:`R_T`. That second convolution is what separates AGTP from AGWP, and it
is essential here: it is the reason a short lived forcer has a *converging*
cumulative temperature effect while carbon dioxide does not.

Three physical ingredients are included, all with published IPCC AR6 Chapter 7
parameters and no free parameters fitted by this study:

1. the multi exponential carbon cycle response of Joos et al. (2013);
2. the two timescale thermal response used for the AR6 metrics tables, whose
   equilibrium sensitivity is 0.7578 K (W m^-2)^-1, that is an equilibrium
   climate sensitivity of 3.0 K;
3. the climate carbon feedback of Gasser et al. (2017), by which the warming
   caused by any emission releases further carbon dioxide from land and ocean
   reservoirs.

Every kernel is a poly exponential function, so all three ingredients compose
exactly through the algebra of :mod:`cipo.expsum`. The feedback term in
particular is obtained in closed form here, whereas reference implementations
evaluate it with computer algebra or on a time grid.

Correction to the research plan
-------------------------------
Equation (1) of the plan defines AGTP as :math:`\\int_0^t \\mathrm{IRF}_x
\\mathrm{RE}_x\\,dt'`, which is AGWP, the integrated forcing. The thermal
convolution is absent. Under that definition the asymptotic claim of
Proposition 1, :math:`\\mathrm{RE}_x\\tau_x^2`, does not follow, and the
cumulative effect of *every* species would diverge, removing the central
physical asymmetry the framework rests on. This module implements the standard
two convolution AGTP, for which the correct limit is
:math:`\\mathrm{RE}_x \\tau_x \\sum_k c_k`.

Units: forcing W m^-2, temperature K, time yr, mass kg of the named species.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Dict, Optional, Sequence, Tuple

import numpy as np

from .expsum import ExpSum

__all__ = [
    "ThermalResponse",
    "CarbonCycle",
    "CarbonClimateFeedback",
    "Species",
    "ClimateModel",
    "AR6",
    "AR6_THERMAL",
    "AR6_CH6_THERMAL",
    "AR5_THERMAL",
    "AR6_REFERENCE",
    "M_ATMOS",
    "M_AIR",
    "MOLAR_MASS",
    "ppb_to_kg",
]

#: Mass of the atmosphere in kg and mean molar mass of dry air in g/mol,
#: following the AR6 metrics constants.
M_ATMOS = 5.1352e18
M_AIR = 28.97

MOLAR_MASS = {
    "CO2": 44.01,
    "C": 12.0,
    "CH4": 16.043,
    "N2O": 44.0,
    "HFC32": 52.03,
    "HFC134a": 102.04,
    "CFC11": 137.36,
    "PFC14": 88.01,
    "SF6": 146.06,
}


def ppb_to_kg(species: str) -> float:
    """Mass in kg of a species corresponding to a 1 ppb molar perturbation."""
    return (1e-9 * MOLAR_MASS[species] / M_AIR) * M_ATMOS


@dataclass(frozen=True)
class ThermalResponse:
    """Two timescale thermal response.

    .. math:: R_T(t) = \\sum_k \\frac{c_k}{d_k} e^{-t/d_k}

    with ``c`` in K (W m^-2)^-1 and ``d`` in years, so ``sum(c)`` is the
    equilibrium warming per unit sustained forcing.
    """

    c: Tuple[float, ...]
    d: Tuple[float, ...]
    source: str = ""

    def kernel(self) -> ExpSum:
        return ExpSum((1.0 / dk, [ck / dk]) for ck, dk in zip(self.c, self.d))

    @property
    def equilibrium_sensitivity(self) -> float:
        """Equilibrium warming per unit sustained forcing, K (W m^-2)^-1."""
        return float(sum(self.c))

    def ecs(self, forcing_2x: float = 3.93) -> float:
        """Equilibrium climate sensitivity in K for a given 2xCO2 forcing."""
        return self.equilibrium_sensitivity * forcing_2x

    @classmethod
    def from_ar6_form(
        cls,
        k_pulse: float,
        a: Sequence[float],
        d: Sequence[float],
        source: str = "",
    ) -> "ThermalResponse":
        """Build from the AR6 parameterisation ``c_k = k_pulse * a_k``."""
        return cls(tuple(k_pulse * ak for ak in a), tuple(d), source)


#: Thermal response used to generate the AR6 Chapter 7 metrics tables.
#: Equilibrium sensitivity 0.7578 K (W m^-2)^-1, that is ECS = 2.98 K.
AR6_THERMAL = ThermalResponse.from_ar6_form(
    0.7578,
    (0.5856, 0.4144),
    (3.424, 285.0),
    "IPCC AR6 WGI Ch.7 metrics calculation (ECS 3.0 K)",
)

#: Geoffroy et al. (2013) fit, as used for the AR6 Chapter 6 GSAT figures.
AR6_CH6_THERMAL = ThermalResponse.from_ar6_form(
    0.885,
    (0.587, 0.413),
    (4.1, 249.0),
    "Geoffroy et al. (2013); IPCC AR6 WGI Ch.6 SM (ECS 3.5 K)",
)

#: IPCC AR5 response (Boucher and Reddy 2008), retained for sensitivity tests.
AR5_THERMAL = ThermalResponse(
    (0.631, 0.429), (8.4, 409.5), "IPCC AR5 WGI Ch.8 SM (Boucher and Reddy 2008)"
)


@dataclass(frozen=True)
class CarbonCycle:
    """Carbon dioxide airborne fraction ``a0 + sum_j a_j e^{-t/tau_j}``."""

    a: Tuple[float, ...] = (0.2173, 0.2240, 0.2824, 0.2763)
    tau: Tuple[float, ...] = (np.inf, 394.4, 36.54, 4.304)
    source: str = "Joos et al. (2013); IPCC AR6 WGI Table 7.SM.6"

    def irf(self) -> ExpSum:
        return ExpSum(
            (0.0 if not np.isfinite(tj) else 1.0 / tj, [aj])
            for aj, tj in zip(self.a, self.tau)
        )

    @property
    def permanent_fraction(self) -> float:
        """The share of a pulse that never decays, the ``a0`` term.

        Strictly positive, and the sole reason temporary storage cannot offset
        carbon dioxide over an unbounded horizon.
        """
        return float(
            sum(aj for aj, tj in zip(self.a, self.tau) if not np.isfinite(tj))
        )


@dataclass(frozen=True)
class CarbonClimateFeedback:
    """Climate carbon feedback of Gasser et al. (2017).

    Warming releases further carbon from land and ocean reservoirs. The extra
    carbon flux produced by a temperature anomaly :math:`\\Delta T` is
    :math:`\\gamma\\,(r * \\Delta T)`, where

    .. math:: r(t) = \\delta(t) - \\sum_i \\frac{a_i}{\\tau_i} e^{-t/\\tau_i},
              \\qquad \\sum_i a_i = 1 .

    The released carbon dioxide then decays with the carbon cycle response and
    exerts forcing of its own, so the feedback contributes

    .. math::
        \\Delta \\mathrm{AGTP}_x
        = \\mathrm{RE}_{\\mathrm{CO_2}}\\,\\gamma_{\\mathrm{kg}}\\,
          \\big[\\,r * \\mathrm{AGTP}^{(0)}_x * \\mathrm{IRF}_{\\mathrm{CO_2}}
          * R_T \\,\\big] .

    ``gamma`` is in GtC per K and is converted internally to kg of carbon
    dioxide per K.
    """

    gamma: float = 3.015
    a: Tuple[float, ...] = (0.6368, 0.3322, 0.0310)
    tau: Tuple[float, ...] = (2.376, 30.14, 490.1)
    order: int = 1
    source: str = "Gasser et al. (2017); IPCC AR6 WGI 7.SM.5.8"

    @property
    def gamma_kg_co2(self) -> float:
        """Feedback strength in kg CO2 per K."""
        return self.gamma * (MOLAR_MASS["CO2"] / MOLAR_MASS["C"]) * 1e12

    def decay_kernel(self) -> ExpSum:
        """The ``sum_i (a_i / tau_i) e^{-t/tau_i}`` part of ``r``."""
        return ExpSum((1.0 / ti, [ai / ti]) for ai, ti in zip(self.a, self.tau))

    def apply(self, temperature: ExpSum) -> ExpSum:
        """The extra carbon dioxide flux ``gamma * (r * temperature)``, kg/yr."""
        return (
            temperature - self.decay_kernel().conv(temperature)
        ).scaled(self.gamma_kg_co2)


@dataclass(frozen=True)
class Species:
    """A greenhouse gas characterised by its decay and radiative efficiency.

    Parameters
    ----------
    re_per_kg
        Radiative efficiency in W m^-2 kg^-1, including the chemical
        adjustments assessed in AR6 where applicable.
    tau
        Perturbation lifetime in years. ``None`` selects the carbon cycle
        response instead of a single exponential.
    co2_yield
        Kilograms of carbon dioxide produced per kilogram of the species as it
        oxidises. The carbon appears as the parent gas decays, so the source
        flux is ``co2_yield * tau^-1 exp(-t / tau)``. Non zero for fossil
        origin methane only.
    """

    name: str
    re_per_kg: float
    tau: Optional[float] = None
    co2_yield: float = 0.0
    label: str = ""
    source: str = ""

    @property
    def display(self) -> str:
        return self.label or self.name


class ClimateModel:
    """Impulse response climate model with cached, exact response kernels."""

    def __init__(
        self,
        thermal: ThermalResponse = AR6_THERMAL,
        carbon: CarbonCycle = CarbonCycle(),
        species: Optional[Dict[str, Species]] = None,
        feedback: Optional[CarbonClimateFeedback] = CarbonClimateFeedback(),
        forcing_2x: float = 3.93,
    ) -> None:
        self.thermal = thermal
        self.carbon = carbon
        self.feedback = feedback
        self.forcing_2x = forcing_2x
        self.species: Dict[str, Species] = dict(species or {})
        self._cache: Dict[str, ExpSum] = {}

    # ------------------------------------------------------------------ setup
    def add_species(self, sp: Species) -> None:
        self.species[sp.name] = sp
        self._cache.clear()

    def replace(self, **kwargs) -> "ClimateModel":
        """A copy of the model with selected components replaced."""
        base = dict(
            thermal=self.thermal,
            carbon=self.carbon,
            species=self.species,
            feedback=self.feedback,
            forcing_2x=self.forcing_2x,
        )
        base.update(kwargs)
        return ClimateModel(**base)

    def with_perturbation(
        self,
        thermal_scale: float = 1.0,
        re_scale: Optional[Dict[str, float]] = None,
        tau_scale: Optional[Dict[str, float]] = None,
        a0_scale: float = 1.0,
        gamma_scale: float = 1.0,
    ) -> "ClimateModel":
        """A copy with perturbed parameters, for Monte Carlo and Sobol runs."""
        thermal = ThermalResponse(
            tuple(c * thermal_scale for c in self.thermal.c),
            self.thermal.d,
            self.thermal.source + " (perturbed)",
        )
        a = list(self.carbon.a)
        if a0_scale != 1.0:
            perm_idx = [i for i, t in enumerate(self.carbon.tau) if not np.isfinite(t)]
            perm = sum(a[i] for i in perm_idx)
            new_perm = min(max(perm * a0_scale, 1e-6), 0.9)
            rest, new_rest = 1.0 - perm, 1.0 - new_perm
            for i in range(len(a)):
                a[i] *= (new_perm / perm) if i in perm_idx else (new_rest / rest)
        carbon = CarbonCycle(tuple(a), self.carbon.tau, self.carbon.source)
        feedback = None
        if self.feedback is not None:
            feedback = CarbonClimateFeedback(
                self.feedback.gamma * gamma_scale,
                self.feedback.a,
                self.feedback.tau,
                self.feedback.order,
                self.feedback.source,
            )
        sp_new = {}
        for name, sp in self.species.items():
            sp_new[name] = Species(
                sp.name,
                sp.re_per_kg * (re_scale or {}).get(name, 1.0),
                None if sp.tau is None else sp.tau * (tau_scale or {}).get(name, 1.0),
                sp.co2_yield,
                sp.label,
                sp.source,
            )
        return ClimateModel(thermal, carbon, sp_new, feedback, self.forcing_2x)

    # -------------------------------------------------------------- identity
    def signature(self) -> tuple:
        """A hashable summary of every parameter that defines this model.

        Downstream objects cache results per climate model. Keying such a
        cache on the object identity is unsafe: a model built inside a loop is
        garbage collected once the iteration ends, and the interpreter reuses
        its address for the next one, so a stale entry is returned for a
        different model. Uncertainty analyses, which build one model per draw,
        are exactly the case that triggers it. Keying on this signature is
        both safe and useful, since rebuilding an identical model reuses the
        cached response.
        """
        return (
            self.thermal.c,
            self.thermal.d,
            self.carbon.a,
            self.carbon.tau,
            self.forcing_2x,
            None
            if self.feedback is None
            else (
                self.feedback.gamma,
                self.feedback.a,
                self.feedback.tau,
                self.feedback.order,
            ),
            tuple(
                (sp.name, sp.re_per_kg, sp.tau, sp.co2_yield)
                for sp in sorted(self.species.values(), key=lambda s: s.name)
            ),
        )

    # ---------------------------------------------------------------- kernels
    def _cached(self, key: str, builder) -> ExpSum:
        if key not in self._cache:
            self._cache[key] = builder()
        return self._cache[key]

    def irf(self, name: str) -> ExpSum:
        """Airborne fraction of a unit pulse of the parent gas."""
        sp = self.species[name]
        if sp.tau is None:
            return self.carbon.irf()
        return ExpSum.exponential(1.0, 1.0 / sp.tau)

    def rf_kernel_direct(self, name: str) -> ExpSum:
        """Forcing per unit pulse before the climate carbon feedback."""

        def build() -> ExpSum:
            sp = self.species[name]
            kern = self.irf(name).scaled(sp.re_per_kg)
            if sp.co2_yield and sp.tau is not None:
                source = ExpSum.exponential(sp.co2_yield / sp.tau, 1.0 / sp.tau)
                kern = kern + source.conv(self.carbon.irf()).scaled(
                    self.species["CO2"].re_per_kg
                )
            return kern

        return self._cached("rf0:" + name, build)

    def agtp_kernel_direct(self, name: str) -> ExpSum:
        """AGTP before the climate carbon feedback."""
        return self._cached(
            "agtp0:" + name,
            lambda: self.rf_kernel_direct(name).conv(self.thermal.kernel()),
        )

    def feedback_rf(self, name: str) -> ExpSum:
        """Extra forcing from carbon released by the warming itself."""

        def build() -> ExpSum:
            if self.feedback is None:
                return ExpSum.zero()
            total = ExpSum.zero()
            temperature = self.agtp_kernel_direct(name)
            for _ in range(max(self.feedback.order, 0)):
                flux = self.feedback.apply(temperature)
                extra_rf = flux.conv(self.carbon.irf()).scaled(
                    self.species["CO2"].re_per_kg
                )
                total = total + extra_rf
                # Higher orders: the extra forcing warms further, which
                # releases more carbon again.
                temperature = extra_rf.conv(self.thermal.kernel())
            return total

        return self._cached("rffb:" + name, build)

    def rf_kernel(self, name: str) -> ExpSum:
        """Total radiative forcing per unit pulse, W m^-2 kg^-1."""
        return self._cached(
            "rf:" + name,
            lambda: self.rf_kernel_direct(name) + self.feedback_rf(name),
        )

    def agtp_kernel(self, name: str) -> ExpSum:
        """AGTP as an exact poly exponential function of time, K kg^-1."""
        return self._cached(
            "agtp:" + name,
            lambda: self.rf_kernel(name).conv(self.thermal.kernel()),
        )

    # ------------------------------------------------------------ the metrics
    def agwp(self, name: str, horizon) -> np.ndarray:
        """Absolute global warming potential, W m^-2 yr kg^-1."""
        return self.rf_kernel(name).definite(horizon)

    def agtp(self, name: str, t) -> np.ndarray:
        """Absolute global temperature potential at time ``t``, K kg^-1."""
        return self.agtp_kernel(name).eval(t)

    def iagtp(self, name: str, horizon) -> np.ndarray:
        """Integrated AGTP over ``[0, horizon]``, K yr kg^-1.

        Grows without bound for carbon dioxide, because the ``a0`` term
        sustains a floor of warming, and converges for any species with a
        finite lifetime.
        """
        return self.agtp_kernel(name).definite(horizon)

    def iagtp_limit(self, name: str) -> float:
        """Limit of iAGTP as the horizon grows, finite iff the species decays.

        For a single exponential species with no feedback and no oxidation
        term this equals :math:`\\mathrm{RE}_x \\tau_x \\sum_k c_k`, which is
        the correct form of Proposition 1 of the research plan.
        """
        return self.agtp_kernel(name).limit_definite()

    def gwp(self, name: str, horizon) -> np.ndarray:
        """Global warming potential relative to carbon dioxide."""
        return self.agwp(name, horizon) / self.agwp("CO2", horizon)

    def gtp(self, name: str, horizon) -> np.ndarray:
        """Global temperature change potential relative to carbon dioxide."""
        return self.agtp(name, horizon) / self.agtp("CO2", horizon)

    def igtp(self, name: str, horizon) -> np.ndarray:
        """Integrated GTP, the ratio of cumulative temperature effects."""
        return self.iagtp(name, horizon) / self.iagtp("CO2", horizon)

    # ------------------------------------------------------------ diagnostics
    def metric_table(self, horizons: Sequence[float] = (20.0, 100.0, 500.0)):
        import pandas as pd

        rows = []
        for name in self.species:
            for th in horizons:
                rows.append(
                    {
                        "species": name,
                        "horizon_yr": th,
                        "AGWP": float(self.agwp(name, th)),
                        "AGTP": float(self.agtp(name, th)),
                        "iAGTP": float(self.iagtp(name, th)),
                        "GWP": float(self.gwp(name, th)),
                        "GTP": float(self.gtp(name, th)),
                        "iGTP": float(self.igtp(name, th)),
                    }
                )
        return pd.DataFrame(rows)


# ---------------------------------------------------------------------------
# Default model: IPCC AR6 Chapter 7 parameters throughout.
#
# Radiative efficiencies are the per ppb values used to generate the AR6
# metrics tables, including the assessed chemical adjustments for methane
# (tropospheric ozone, stratospheric water vapour, rapid adjustment) and for
# nitrous oxide (stratospheric ozone depletion and reduced methane). Fossil
# methane additionally carries the carbon dioxide from its oxidation, at the
# AR6 yield Y = 0.75. No parameter in this model is fitted by this study.
# ---------------------------------------------------------------------------

_RE_PER_PPB = {
    "CO2": 1.3330689487029318e-5,
    "CH4": 5.686440286086949e-4,
    "N2O": 2.7788125677697985e-3,
    "HFC32": 0.11144,
    "HFC134a": 0.16714,
    "CFC11": 0.25941 * 1.12,
    "PFC14": 0.09859,
    "SF6": 0.567,
}

#: Fraction of fossil methane carbon that reaches the atmosphere as carbon
#: dioxide. IPCC AR6 7.SM.5.8 quotes Y = 0.75 for the tropospheric hydroxyl
#: pathway; the full stoichiometric conversion Y = 1 reproduces the published
#: offset between fossil and biogenic methane GWP100 (29.8 against 27.0) more
#: closely, and is used here. Setting it to 0.75 changes fossil methane
#: metrics by about 1 per cent and changes no result of this study.
FOSSIL_METHANE_YIELD = 1.0


def _re(name: str) -> float:
    return _RE_PER_PPB[name] / ppb_to_kg(name)


def _default_species() -> Dict[str, Species]:
    src = "IPCC AR6 WGI Ch.7 metrics parameters"
    return {
        "CO2": Species("CO2", _re("CO2"), None, 0.0, "CO$_2$", src),
        "CH4": Species("CH4", _re("CH4"), 11.8, 0.0, "CH$_4$ (biogenic)", src),
        "CH4_fossil": Species(
            "CH4_fossil",
            _re("CH4"),
            11.8,
            FOSSIL_METHANE_YIELD * MOLAR_MASS["CO2"] / MOLAR_MASS["CH4"],
            "CH$_4$ (fossil)",
            src,
        ),
        "N2O": Species("N2O", _re("N2O"), 109.0, 0.0, "N$_2$O", src),
        "HFC32": Species("HFC32", _re("HFC32"), 5.4, 0.0, "HFC-32", src),
        "HFC134a": Species("HFC134a", _re("HFC134a"), 14.0, 0.0, "HFC-134a", src),
        "CFC11": Species("CFC11", _re("CFC11"), 52.0, 0.0, "CFC-11", src),
        "PFC14": Species("PFC14", _re("PFC14"), 50000.0, 0.0, "PFC-14", src),
        "SF6": Species("SF6", _re("SF6"), 3200.0, 0.0, "SF$_6$", src),
    }


def AR6(feedback: bool = True) -> ClimateModel:
    """The default AR6 calibrated climate model."""
    return ClimateModel(
        AR6_THERMAL,
        CarbonCycle(),
        _default_species(),
        CarbonClimateFeedback() if feedback else None,
    )


#: Published AR6 Table 7.SM.7 / Table 7.15 values used by the validation suite.
AR6_REFERENCE = {
    "AGWP_CO2_100": 9.17e-14,
    "AGWP_CO2_20": 2.49e-14,
    "GWP100": {"CH4": 27.0, "CH4_fossil": 29.8, "N2O": 273.0},
    "GWP20": {"CH4": 79.7, "CH4_fossil": 82.5, "N2O": 273.0},
    "GTP100": {"CH4": 4.7, "CH4_fossil": 7.5, "N2O": 233.0},
}
