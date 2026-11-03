# wrcp

Fire-exposed carbon dioxide removal portfolios, in closed form.

Temporary carbon storage is usually credited as though its storage time were
known. `wrcp` drops that assumption. It treats wildfire as a marked Poisson
process acting on the standing carbon stock, and returns the mean and the
covariance of the cooling a store actually delivers, with no time grid and no
discretisation error.

## What it computes

- The fire-adjusted credit of a measure, its mean and its variance, in closed
  form.
- The least-cost portfolio that offsets an emission pathway while meeting the
  climate target with a stated confidence, posed as a second-order cone
  programme.
- The effective cooling of each measure, its mean contribution less the
  marginal risk it adds, which decides whether the measure enters at all.
- The confidence ceiling of a portfolio, and the condition under which it
  binds.
- The deployment schedule, since exposure accumulates from the date a store is
  established.

## Install

```
pip install -r requirements.txt
pip install -e .
```

Python 3.8 or later. It depends on NumPy, SciPy, pandas and Matplotlib.

## Reproduce everything

```
cd scripts
python3 run_all.py
```

This runs the validation suite, regenerates every table and figure, re-derives
the sensitivity sweep and re-checks every quantitative claim against the
generated output. Figures land in `figures/`, tables and key values in
`results/`. A full run takes minutes on one core.

Individual stages can be run on their own; each script in `scripts/` is
independent except through a cached scenario solution.

## Layout

```
wrcp/       the fire-exposed model: hazard, stochastic moments, robust
            portfolio, scheduling, calibration, regions
cipo/       the deterministic climate and portfolio model it builds on,
            vendored unchanged
scripts/    figure, table, validation, sensitivity and audit drivers
tests/      pytest suite
```

## Validation

```
python3 -m pytest tests -q          # 383 tests
cd scripts && python3 validation.py # 70 checks, seven tiers
```

The validation tiers compare the closed-form algebra against an independent
analytical route, the climate model against assessed emission metrics, the fire
moments against direct simulation of the fire process, the optimality
conditions against their analytical form, the reduction to the deterministic
case as the hazard vanishes, the internal consistency of the calibrated fire
regimes, and the sensitivity of every conclusion to the inputs that observation
does not pin down.

`scripts/check_claims.py` registers every quantitative claim in the
accompanying manuscript against the quantity in the generated output that
produces it, and fails if any claim is no longer supported.

## Licence

MIT.
