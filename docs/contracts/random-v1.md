# Random sampling for the Python model

Status: accepted after independent review, 26 September 2026.

> The per-draw key/hash design is superseded by one `np.random.default_rng(seed)`
> per run (this contract) and by decision 0004 item 14 (28 Sep 2026), which layers
> `scipy.stats` distributions and a statsmodels AR(1) price process on the same
> generator. This contract's one-generator-per-run rule stays current; read it
> together with item 14 for the sampling libraries in use.

This replaces the earlier per-draw key protocol. That protocol is preserved in
Git history, not in the active model. No Python caller used it to sample a value.

## One generator per full run

Create a NumPy generator when an explicit full Run starts, then pass it to
the functions that draw uncertain inputs. Do not create a generator in charts,
ordinary Streamlit reruns, or per-EV/per-interval helpers.

```python
import numpy as np

rng = np.random.default_rng(seed)
population = sample_population(rng, assumptions)
planning_inputs = sample_planning_inputs(rng, population, assumptions)
evaluation_inputs = sample_evaluation_inputs(rng, population, assumptions)
```

These function names illustrate the call flow; the simulation runner and
samplers do not exist yet. Generate a missing seed only after a valid Run
click and store it with the configuration and model version. The same seed,
inputs, code and dependency versions reproduce the same run. Do not promise
the same draws after changing draw order, distribution calls or NumPy version.

## Sampling boundaries

- Sample persistent population traits once and reuse them in planning and
  evaluation.
- Planning and evaluation future samples are distinct. The planner receives
  only planning inputs and information available at decision time; it never
  receives future evaluation values.
- Sample exogenous inputs once per evaluation world. Pass the same weather,
  price, travel, connection, fault and service inputs to both normal and
  selected physical paths. Do not draw them again for either path.
- Arrival SoC is derived from battery stock-flow, never sampled.
- Aggregate additive quantities within each world before calculating
  quantiles across worlds. Never add percentiles.

Use NumPy distribution methods directly with parameters recorded in the
source and assumptions data. Keep observed data, fitted parameters and
illustrative assumptions distinct. A generic distribution registry is not
required for this prototype.

One sequential generator means that adding an earlier draw can change later
samples. This is an accepted prototype limitation, not a failure of
fixed-seed repeatability. If a concrete later requirement needs evaluation
samples to stay fixed when planning draw counts change, use NumPy's
`SeedSequence.spawn` for a small number of run-level streams, with a focused
test. Per-draw key encoding, hashes, namespaces, manifests, runtime profiles
and golden-bit fixtures are not current requirements.

## Tests when the runner exists

Use small fixed-seed scenarios to test repeatability, distinct planning and
evaluation inputs, population reuse and exact normal/selected exogenous
pairing. Test distribution parameter validation and physical stock-flow in
their own functions. Do not test NumPy's internals in isolation.
