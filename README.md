# spatial_hw2

Spatial Economics HW2 project.

Current implemented scope:
- 4-sector baseline-only CDP Julia machinery (no counterfactual yet)
- Task-based pipeline with `code/input/output` per task

Pipeline:
1. `tasks/setup_environment`
2. `tasks/cdp4_baseline_solver`
3. `tasks/cdp4_baseline_dynamics`
4. `tasks/cdp4_baseline_validate`

Run from project root:
```bash
make
```

Run quick smoke pipeline:
```bash
make smoke_pipeline
```

Run replication profiles:
```bash
make baseline_dynamics_reference validate_reference
make baseline_dynamics_fast validate_fast
```

Runtime knobs:
- `MAX_ITER_STATIC` and `MAX_ITER_DYNAMIC` to cap solver iterations.
- `USE_THREADS=1` to enable optional threaded kernels (off by default).
- `THREADS_DYNAMIC=1` and/or `THREADS_STATIC=1` to select threaded sections when `USE_THREADS=1`.
- `PROFILE=reference|fast` (`fast` default).
- `WARM_START_STATIC=0|1` (defaults to on in `fast`, off in `reference`).
- `RECORD_TRACE=0|1` to control outer-iteration trace CSV writing.
- `CONFIG_TAG=<label>` to tag benchmark CSV outputs.

Benchmark outputs:
- `tasks/cdp4_baseline_solver/output/benchmark_solver_4sector.csv`
- `tasks/cdp4_baseline_dynamics/output/benchmark_4sector.csv`
- `tasks/cdp4_baseline_validate/output/benchmark_validate_4sector.csv`

Trace/parity outputs:
- `tasks/cdp4_baseline_dynamics/output/outer_trace_4sector.csv`
- `tasks/cdp4_baseline_validate/output/parity_by_time_4sector.csv`

Expected behavior bands:
- Smoke profile (`MAX_ITER_DYNAMIC=5`) is for regression safety checks only and is not expected to fully converge.
- Replication profiles are expected to converge (`final_ymax <= 1e-3`) and match MATLAB `Hvectnoshock.mat` at converged-path tolerance (`ynew_max_rel_error <= 1e-3`).

Replication folders are local-only and intentionally untracked:
- `CDP_codes_four_sectors/`
- `ecta200013-sup-0002-dataandprograms/`
