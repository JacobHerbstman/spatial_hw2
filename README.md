# spatial_hw2

Minimal submission-oriented HW2 repo for a simplified 4-sector CDP-style dynamic model.

## Canonical Objective

The submission-facing exercise is the AI/cognitive counterfactual from `ps2.pdf`:

- `cognitive_immediate`
- `cognitive_anticipated`

The repo keeps the validated baseline plus identity and toy counterfactuals as regression checks, but the only canonical submission outputs live in `tasks/cognitive_counterfactual/output/`.

## Build Surface

Run the full submission build from the project root:

```bash
make
```

This does three things:

1. builds the baseline reference path and validation,
2. reruns the identity and toy reference regressions,
3. builds the canonical AI reference outputs and then compiles the paper.

Useful root targets:

```bash
make tasks
make reference
make fast
make diagnostic
make clean
```

- `make tasks` runs the regression pipeline without compiling the paper.
- `make reference` runs only the canonical AI reference pipeline.
- `make fast` runs only the canonical AI debug pipeline.
- `make diagnostic` runs the MATLAB-faithful cognitive diagnostic task.

## Structure

The repo follows the `spatial_hw1` convention:

- root `Makefile` orchestrates the project,
- each active task is built from its `code/Makefile`,
- dormant task-root wrapper `Makefile`s are removed.

Active task roles:

- `tasks/cdp4_baseline_solver`: static baseline solve.
- `tasks/cdp4_baseline_dynamics`: baseline transition path.
- `tasks/cdp4_baseline_validate`: baseline checks against `Hvectnoshock.mat`.
- `tasks/cdp4_counterfactual_dynamics`: shared counterfactual solver used for identity, toy, CSV, and MAT shocks.
- `tasks/cdp4_counterfactual_validate`: generic counterfactual validation and reporting.
- `tasks/cognitive_intensity_data`: builds the cognitive-intensity matrix used to scale the AI shocks.
- `tasks/cognitive_counterfactual`: canonical submission pipeline and collected outputs.
- `tasks/cdp4_cognitive_minimal`: diagnostic-only MATLAB-faithful cognitive check.

## Reference vs Fast

Final reported results should come from `reference` mode:

- `WARM_START_STATIC=0`
- `USE_ANDERSON=0`
- `HVECT_RELAX=0.5`
- no threaded static or dynamic kernels
- baseline reference path used as the initial anchor

`fast` mode is debug-only and is not the submission target.

The canonical AI task searches over a bounded shock ladder:

- `0.01`
- `0.005`
- `0.0025`
- `0.001`

It keeps the largest common delta that converges for both timing scenarios and records the chosen value, solver settings, and shock checksum in per-scenario manifests.

## Paper

The paper build lives in `paper/` and reads only from `tasks/cognitive_counterfactual/output/`.

Expected canonical outputs include:

- baseline validation table,
- immediate vs anticipated AI results table,
- dynamic adjustment figures,
- cross-state real-wage map,
- scenario manifests.

## Local Reference Data

These folders are local-only references and should remain untracked:

- `CDP_codes_four_sectors/`
- `ecta200013-sup-0002-dataandprograms/`
