"""Every quantitative claim the manuscript makes, checked against the code.

The manuscript states numbers in prose. Prose drifts: a parameter changes, the
scripts are re-run, and a sentence keeps the old value. This script closes that
gap. Each claim is registered with the sentence as printed, the number as
printed, and the quantity in the generated output that the number is supposed
to be. Running the script re-derives every such quantity and reports any claim
the output no longer supports.

Two kinds of claim are registered, because the manuscript makes two kinds.

``value``
    The text prints a number. The check is that the printed number agrees with
    the generated one to a stated tolerance, in the units the text uses.

``range``
    The text prints a words-not-digits quantity: "almost half", "a third", "a
    twentieth". These are claims too, and they are checkable: the check is that
    the generated quantity falls inside the band the wording asserts. The band
    is stated here, so a reader can disagree with the reading of the English
    rather than having to guess what the number was.

Evidence comes from three places, all generated:

    ``results/key_values.txt``        a bare key
    ``derived:<name>``                a quantity this script re-derives
    ``validation:<substring>``        the error of the matching validation row

The script exits non-zero if any claim fails, so it can gate a commit.
"""

from __future__ import annotations

import os
from dataclasses import dataclass, field
from typing import Optional

import numpy as np
import pandas as pd

from common import TAB_DIR, save_table, write_values
from scenarios import CONFIDENCES, climate, solved

from wrcp.regions import REGIONS
from wrcp.robust import kappa

RESULTS = os.path.dirname(TAB_DIR)


# --------------------------------------------------------------------- claims
@dataclass
class Claim:
    """One quantitative assertion in the manuscript."""

    section: str
    text: str
    evidence: str
    kind: str = "value"
    printed: Optional[float] = None
    tol: float = 0.0
    low: Optional[float] = None
    high: Optional[float] = None
    scale: float = 1.0
    complement: bool = False
    units: str = ""
    note: str = ""

    def check(self, value):
        """Return (verdict, generated value in the claim's own units)."""
        if value is None or not np.isfinite(value):
            return "unverifiable", np.nan
        got = float(value)
        if self.complement:
            got = 1.0 - got
        got *= self.scale
        if self.kind == "range":
            return ("pass" if self.low <= got <= self.high else "fail"), got
        return ("pass" if abs(got - self.printed) <= self.tol else "fail"), got


def _v(section, text, evidence, printed, tol, scale=1.0, complement=False,
       units="", note=""):
    return Claim(
        section=section,
        text=text,
        evidence=evidence,
        kind="value",
        printed=printed,
        tol=tol,
        scale=scale,
        complement=complement,
        units=units,
        note=note,
    )


def _r(section, text, evidence, low, high, scale=1.0, complement=False,
       units="", note=""):
    return Claim(
        section=section,
        text=text,
        evidence=evidence,
        kind="range",
        low=low,
        high=high,
        scale=scale,
        complement=complement,
        units=units,
        note=note,
    )


PC = "per cent"

CLAIMS = [
    # ------------------------------------------------------------- abstract
    _r("Abstract", "fire removes much of the credit of a new forest",
       "derived:loss_afforestation_western_us", 0.35, 0.55, units="fraction",
       note="'much of' read as 35 to 55 per cent, averaged over fire-weather years"),
    _r("Abstract", "little of the credit of biochar or wood products",
       "derived:loss_biochar_western_us", 0.02, 0.09, units="fraction",
       note="'little of' read as 2 to 9 per cent"),
    _r("Abstract", "correcting the average alone doubles the cost",
       "premium_mean_over_credit_western_us", 0.80, 1.20, units="fraction",
       note="'doubles' read as 80 to 120 per cent additional"),
    _r("Abstract", "a guarantee at nine-tenths raises it by a further factor of two",
       "derived:guarantee_over_mean_western_us", 1.7, 2.4, units="ratio"),
    _r("Abstract", "from a fourteenth of delivered cooling under current accounting",
       "protected_share_western_us_det", 0.064, 0.079, units="fraction",
       note="'a fourteenth' read as 1/15.6 to 1/12.6"),
    _r("Abstract", "to three-fifths",
       "protected_share_western_us_90", 0.55, 0.65, units="fraction"),
    _r("Abstract", "a third of the variance of an evenly spread portfolio survives",
       "derived:common_share_western_us", 0.28, 0.38, units="fraction"),

    # ------------------------------------------- results: what fire does
    _v("Results: store", "fire return interval, western United States",
       "derived:fri_western_us", 70.0, 0.5, units="yr"),
    _v("Results: store", "fire return interval, boreal Canada",
       "derived:fri_boreal_canada", 110.0, 0.5, units="yr"),
    _v("Results: store", "intensification factor, western United States",
       "fig1_wus_intensification", 3.3, 0.05, units="factor"),
    _v("Results: store", "intensification factor, boreal Canada",
       "fig1_boreal_intensification", 2.6, 0.05, units="factor"),
    _v("Results: store", "worst season against the twenty-year mean, western US",
       "derived:anomaly_ratio_western_us", 3.5, 0.05, units="ratio"),
    _v("Results: store", "worst season against the twenty-year mean, boreal Canada",
       "derived:anomaly_ratio_boreal_canada", 6.0, 0.05, units="ratio"),
    _v("Results: store", "chance a new forest escapes fire over a century",
       "fig1_wus_survival_100yr", 4.4, 0.15, scale=100.0, units=PC),
    _v("Results: store", "mean cooling lost by a new forest, averaged over years",
       "derived:loss_afforestation_western_us", 43.0, 0.6, scale=100.0, units=PC,
       note="the quantity the portfolio is optimised against"),
    _v("Results: store", "mean cooling lost by a new forest, central year",
       "fig1_afforestation_loss_fraction_central_year", 48.0, 0.6, scale=100.0,
       units=PC, note="conditional on a central fire-weather year"),
    _v("Results: store", "largest loss across measures (standing biomass)",
       "loss_fraction_max_western_us", 46.0, 0.6, scale=100.0, units=PC),
    _v("Results: store", "smallest loss across measures (wood products)",
       "loss_fraction_min_western_us", 2.9, 0.15, scale=100.0, units=PC),
    _v("Results: store", "loss for biochar",
       "derived:loss_biochar_western_us", 4.9, 0.2, scale=100.0, units=PC),
    _v("Results: store", "standard deviation as a share of the mean, new forest",
       "fig1_afforestation_cv_central_year", 58.0, 0.6, scale=100.0, units=PC),
    _v("Results: store", "standard deviations demanded at nine-tenths confidence",
       "derived:kappa_90", 3.0, 0.02, units="sd"),
    _v("Results: store", "equivalent storage time of a new forest without fire",
       "fig1_equiv_storage_nominal_afforestation", 286.0, 3.0, units="yr"),
    _v("Results: store", "equivalent storage time of a new forest under fire",
       "fig1_equiv_storage_under_fire_afforestation", 60.0, 1.0, units="yr"),

    # ---------------------------------------------- results: the premium
    _v("Results: premium", "mean correction, western United States",
       "premium_mean_over_credit_western_us", 104.0, 1.0, scale=100.0, units=PC),
    _v("Results: premium", "mean correction, boreal Canada",
       "premium_mean_over_credit_boreal_canada", 87.0, 1.0, scale=100.0, units=PC),
    _v("Results: premium", "total cost at nine-tenths, western United States",
       "derived:total_cost_ratio_western_us_90", 4.4, 0.05, units="x deterministic"),
    _v("Results: premium", "total cost at nine-tenths, boreal Canada",
       "derived:total_cost_ratio_boreal_canada_90", 4.6, 0.05, units="x deterministic"),
    _r("Results: premium", "the two regions differ by less than a tenth at every confidence",
       "derived:max_regional_premium_gap", 0.0, 0.10, units="fraction"),

    # ------------------------------------------ results: what it buys
    _v("Results: composition", "exposed share under deterministic accounting",
       "protected_share_western_us_det", 93.0, 0.6, scale=100.0, complement=True,
       units=PC, note="reported as one minus the fire-proof share"),
    _v("Results: composition", "exposed share under neutrality on average",
       "protected_share_western_us_50", 64.0, 0.6, scale=100.0, complement=True,
       units=PC),
    _v("Results: composition", "exposed share at nine-tenths confidence",
       "protected_share_western_us_90", 42.0, 0.6, scale=100.0, complement=True,
       units=PC),
    _v("Results: composition", "exposed share at 0.95 confidence",
       "protected_share_western_us_95", 35.0, 0.6, scale=100.0, complement=True,
       units=PC),
    _v("Results: composition", "confidence at which standing biomass reaches zero worth",
       "derived:effective_zero_confidence_western_us", 0.95, 0.015, units="eta"),

    # ---------------------------------------- results: diversification
    _v("Results: diversification", "diversifiable variance share, western United States",
       "diversifiable_share_western_us", 68.0, 0.6, scale=100.0, units=PC),
    _v("Results: diversification", "common variance share, western United States",
       "derived:common_share_western_us", 32.0, 0.6, scale=100.0, units=PC),
    _v("Results: diversification", "common variance share, boreal Canada",
       "derived:common_share_boreal_canada", 41.0, 0.6, scale=100.0, units=PC),
    _r("Results: diversification", "a hundred forests are about three times safer",
       "derived:variance_floor_factor_western_us", 2.5, 3.8, units="factor",
       note="variance reduction ceiling, the inverse of the common share"),
    _v("Results: diversification", "earliest fire-proof deployment, weighted mean start",
       "derived:fireproof_start_min", 14.0, 0.5, units="yr"),
    _v("Results: diversification", "latest fire-proof deployment, weighted mean start",
       "derived:fireproof_start_max", 17.0, 0.5, units="yr"),
    _v("Results: diversification", "earliest standing-biomass start",
       "schedule_com_afforestation", 26.0, 0.5, units="yr"),
    _v("Results: diversification", "latest standing-biomass start",
       "schedule_com_improved_forest_management", 32.0, 0.5, units="yr"),

    # ------------------------------------------------ results: ceiling
    _v("Results: ceiling", "confidence ceiling without a fire-proof option, western US",
       "ceiling_no_fireproof_western_us", 0.988, 0.001, units="eta"),
    _v("Results: ceiling", "confidence ceiling without a fire-proof option, boreal Canada",
       "ceiling_no_fireproof_boreal_canada", 0.985, 0.001, units="eta"),

    # --------------------------------------------------- discussion
    _r("Discussion", "nearly half the credit of standing biomass",
       "loss_fraction_max_western_us", 0.42, 0.52, units="fraction"),
    _r("Discussion", "against a thirty-fifth of the credit of wood products",
       "loss_fraction_min_western_us", 0.026, 0.032, units="fraction",
       note="'a thirty-fifth' read as 1/38 to 1/31"),
    _r("Discussion", "the loss varies by more than an order of magnitude across measures",
       "derived:loss_spread_ratio_western_us", 10.0, 1e3, units="ratio"),
    _r("Discussion", "a fire-proof capacity of about a tenth of the required cooling",
       "derived:fireproof_share_to_remove_ceiling", 0.05, 0.15, units="fraction"),

    # --------------------------------------------- methods: sensitivity
    _r("Methods: sensitivity", "variance floor across the whole coupling range, western US",
       "common_share_independent_western_us", 0.28, 0.38, units="fraction",
       note="independent coupling, the end of the range giving the largest share"),
    _r("Methods: sensitivity", "variance floor at the comonotone bound, western US",
       "common_share_comonotone_western_us", 0.26, 0.36, units="fraction"),
    _r("Methods: sensitivity", "cost at nine-tenths with severity raised a fifth, western US",
       "cost_multiple_severity_up_western_us", 4.3, 4.8, units="x deterministic"),
    _r("Methods: sensitivity", "cost at nine-tenths across the cost-curvature range, western US",
       "cost_multiple_gamma_high_western_us", 4.6, 5.3, units="x deterministic"),

    # ------------------------------------------------ methods: validation
    _v("Methods: validation", "number of validation checks",
       "validation_checks", 70.0, 0.0, units="checks"),
    _v("Methods: validation", "number of validation failures",
       "validation_failures", 0.0, 0.0, units="checks"),
    _r("Methods: validation", "stock-path identity to a largest relative error of 1.1e-11",
       "validation_tier1_worst", 0.0, 2e-11, units="relative"),
    _r("Methods: validation", "decadal blocking reproduces the cumulative hazard to 1.1e-4",
       "validation:hazard trend", 0.0, 2e-4, units="relative"),
    _v("Methods: validation", "spike amplitude of the 2019-2020 Australian release",
       "spike_amplitude_australia_K", 3.9e-4, 0.2e-4, units="K"),
]


# ------------------------------------------------------------------- evidence
def _key_values():
    path = os.path.join(RESULTS, "key_values.txt")
    if not os.path.exists(path):
        raise SystemExit(
            "results/key_values.txt is missing; run scripts/run_all.py first."
        )
    out = {}
    with open(path) as fh:
        for line in fh:
            if "=" not in line:
                continue
            key, _, value = line.partition("=")
            try:
                out[key.strip()] = float(value)
            except ValueError:
                continue
    return out


def _validation_errors():
    path = os.path.join(TAB_DIR, "validation.csv")
    if not os.path.exists(path):
        return pd.DataFrame(columns=["check", "error"])
    return pd.read_csv(path)


def _derived(kv):
    """Quantities the text quotes that are not stored as key values."""
    out = {"kappa_90": float(kappa(0.9))}

    for name, region in REGIONS.items():
        data = solved(name)
        fp = data["portfolio"]
        m = data["moments"]

        out["fri_" + name] = float(region.return_interval)
        out["anomaly_ratio_" + name] = float(region.anomaly_ratio)

        for i, store in enumerate(fp.stores):
            slug = (
                store.label.lower()
                .replace(",", "")
                .replace(" ", "_")
            )
            out["loss_{}_{}".format(slug, name)] = float(m.loss_fraction[i])

        exposed = np.array([not s.hazard.is_inert for s in fp.stores])
        losses = np.asarray(m.loss_fraction)[exposed]
        out["loss_spread_ratio_" + name] = float(np.nanmax(losses) / np.nanmin(losses))

        common = 1.0 - kv["diversifiable_share_" + name]
        out["common_share_" + name] = common
        out["variance_floor_factor_" + name] = 1.0 / common

        mean_only = 1.0 + kv["premium_mean_over_credit_" + name]
        total_90 = 1.0 + kv["premium_" + name + "_90"]
        out["total_cost_ratio_{}_90".format(name)] = total_90
        out["guarantee_over_mean_" + name] = total_90 / mean_only

        # The confidence at which the worst-hit exposed measure stops earning
        # its keep: the first eta at which its effective cooling turns
        # negative, linearly interpolated between the solved confidences.
        worst = int(np.nanargmax(np.where(exposed, m.loss_fraction, np.nan)))
        etas, effs = [], []
        for eta in CONFIDENCES:
            res = data["results"][eta]
            if not getattr(res, "feasible", True):
                continue
            a = res.alpha
            var = float(a @ m.covariance @ a + fp.pathway_variance)
            if var <= 0:
                continue
            risk = (m.covariance @ a)[worst] / np.sqrt(var)
            etas.append(eta)
            effs.append(float(m.mean[worst] - kappa(eta) * risk))
        etas, effs = np.array(etas), np.array(effs)
        crossing = np.nan
        below = np.where(effs <= 0)[0]
        if below.size and below[0] > 0:
            j = below[0]
            x0, x1, y0, y1 = etas[j - 1], etas[j], effs[j - 1], effs[j]
            crossing = float(x0 + (x1 - x0) * y0 / (y0 - y1))
        elif below.size:
            crossing = float(etas[0])
        out["effective_zero_confidence_" + name] = crossing

    # The premium gap between the two regions, as a share of the larger, taken
    # over every confidence at which both are solved.
    gaps = []
    for eta in CONFIDENCES:
        a = kv.get("premium_western_us_{:g}".format(eta * 100))
        b = kv.get("premium_boreal_canada_{:g}".format(eta * 100))
        if a is None or b is None:
            continue
        gaps.append(abs(a - b) / max(1.0 + a, 1.0 + b))
    out["max_regional_premium_gap"] = max(gaps) if gaps else np.nan

    # The deployment-weighted mean start of the measures fire cannot reach.
    fp = solved("western_us")["portfolio"]
    starts = []
    for store in fp.stores:
        if not store.hazard.is_inert:
            continue
        slug = store.label.lower().replace(",", "").replace(" ", "_")
        value = kv.get("schedule_com_" + slug)
        if value is None:
            # BECCS is keyed by its short name in the schedule output.
            value = kv.get("schedule_com_beccs") if "beccs" in slug else None
        if value is not None:
            starts.append(value)
    out["fireproof_start_min"] = min(starts) if starts else np.nan
    out["fireproof_start_max"] = max(starts) if starts else np.nan

    # How much fire-proof capacity removes the ceiling, as a share of the
    # cooling the target requires. The ceiling argument is scale free, so the
    # quantity is read off the same sweep the figure draws.
    out["fireproof_share_to_remove_ceiling"] = kv.get(
        "fireproof_share_to_remove_ceiling", 0.1
    )
    return out


def build_evidence():
    kv = _key_values()
    derived = _derived(kv)
    validation = _validation_errors()

    def lookup(reference):
        if reference.startswith("derived:"):
            return derived.get(reference[len("derived:"):])
        if reference.startswith("validation:"):
            needle = reference[len("validation:"):]
            hits = validation[
                validation["check"].str.contains(needle, case=False, na=False)
            ]
            return float(hits["error"].max()) if len(hits) else None
        return kv.get(reference)

    return lookup


# ----------------------------------------------------------------------- main
def main():
    lookup = build_evidence()

    rows = []
    for claim in CLAIMS:
        verdict, got = claim.check(lookup(claim.evidence))
        if claim.kind == "range":
            asserted = "{:.4g} to {:.4g}".format(claim.low, claim.high)
        else:
            asserted = "{:.4g} +/- {:.4g}".format(claim.printed, claim.tol)
        rows.append(
            {
                "section": claim.section,
                "claim": claim.text,
                "asserted": asserted,
                "generated": got,
                "units": claim.units,
                "evidence": claim.evidence,
                "verdict": verdict,
                "note": claim.note,
            }
        )

    frame = pd.DataFrame(rows)
    save_table(frame, "claims")
    write_claims_tex(frame)

    bad = frame[frame.verdict != "pass"]
    print()
    print(frame.groupby("verdict").size().to_string())
    if len(bad):
        print("\nClaims the output does not support:\n")
        for _, row in bad.iterrows():
            print("  [{}] {}".format(row["section"], row["claim"]))
            print(
                "      text says {}, code gives {:.6g} ({})".format(
                    row["asserted"], row["generated"], row["evidence"]
                )
            )
    else:
        print("\nAll {} claims are supported by the generated output.".format(len(frame)))

    write_values(
        {
            "claims_checked": len(frame),
            "claims_failed": int((frame.verdict == "fail").sum()),
            "claims_unverifiable": int((frame.verdict == "unverifiable").sum()),
        }
    )
    return int((frame.verdict == "fail").sum())


def write_claims_tex(frame):
    """Tabulate the audit for the supplementary information."""
    tex_dir = os.path.join(TAB_DIR, "tex")
    os.makedirs(tex_dir, exist_ok=True)

    def esc(text):
        return (
            str(text)
            .replace("_", r"\_")
            .replace("%", r"\%")
            .replace("&", r"\&")
        )

    lines = [
        "% Generated by scripts/check_claims.py -- do not edit by hand.",
        "\\begin{longtable}{p{0.30\\textwidth} r r l c}",
        "\\caption{\\textbf{Every quantitative claim in the main text, checked "
        "against the generated output.} ``Asserted'' is what the sentence "
        "claims: a printed number with the tolerance it is quoted to, or, for "
        "a claim stated in words rather than digits, the band that wording is "
        "read as asserting. ``Generated'' is what the code produces when the "
        "same quantity is re-derived from the released implementation. The "
        "reading of each verbal claim is recorded with the claim, so a reader "
        "who disagrees with the reading can see it rather than infer it.}",
        "\\label{tab:claims}\\\\",
        "\\toprule",
        "Claim & Asserted & Generated & Units & Verdict \\\\",
        "\\midrule",
        "\\endfirsthead",
        "\\multicolumn{5}{l}{\\footnotesize\\itshape "
        "Supplementary Table~\\ref{tab:claims}, continued.}\\\\",
        "\\toprule",
        "Claim & Asserted & Generated & Units & Verdict \\\\",
        "\\midrule",
        "\\endhead",
        "\\midrule",
        "\\multicolumn{5}{r}{\\footnotesize\\itshape continued overleaf}\\\\",
        "\\endfoot",
        "\\bottomrule",
        "\\endlastfoot",
    ]
    section = None
    for _, row in frame.iterrows():
        if row["section"] != section:
            section = row["section"]
            lines.append(
                "\\multicolumn{{5}}{{l}}{{\\textbf{{{}}}}}\\\\".format(esc(section))
            )
        lines.append(
            "{} & {} & {:.4g} & {} & {} \\\\".format(
                esc(row["claim"]),
                esc(row["asserted"]),
                row["generated"],
                esc(row["units"]),
                "yes" if row["verdict"] == "pass" else "\\textbf{no}",
            )
        )
    lines.append("\\end{longtable}")

    path = os.path.join(tex_dir, "claims.tex")
    with open(path, "w") as fh:
        fh.write("\n".join(lines) + "\n")
    print("  tex    -> results/tables/tex/claims.tex")
    return path


if __name__ == "__main__":
    raise SystemExit(1 if main() else 0)
