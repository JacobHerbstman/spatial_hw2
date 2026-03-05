# spatial_hw2

Spatial Economics HW2 project.

Current implemented scope:
- 4-sector baseline CDP Julia machinery
- 4-sector CDP-style counterfactual machinery (identity, toy, CSV/MAT shock inputs)
- Task-based pipeline with `code/input/output` per task

Pipeline:
1. `tasks/setup_environment`
2. `tasks/cdp4_baseline_solver`
3. `tasks/cdp4_baseline_dynamics`
4. `tasks/cdp4_baseline_validate`
5. `tasks/cdp4_counterfactual_dynamics`
6. `tasks/cdp4_counterfactual_validate`

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

Run counterfactual profiles:
```bash
make counterfactual_identity_reference counterfactual_validate_identity_reference
make counterfactual_identity_fast counterfactual_validate_identity_fast
make counterfactual_toy_smoke
make counterfactual_toy_reference counterfactual_validate_toy_reference
make counterfactual_toy_fast counterfactual_validate_toy_fast
```

Generate counterfactual PDF plots and LaTeX tables:
```bash
make counterfactual_report_identity_fast
make counterfactual_report_identity_reference
make counterfactual_report_toy_fast
make counterfactual_report_toy_reference
make counterfactual_impact_report_toy_smoke_fast
```

Runtime knobs:
- `MAX_ITER_STATIC` and `MAX_ITER_DYNAMIC` to cap solver iterations.
- `USE_THREADS=1` to enable optional threaded kernels (off by default).
- `THREADS_DYNAMIC=1` and/or `THREADS_STATIC=1` to select threaded sections when `USE_THREADS=1`.
- `PROFILE=reference|fast` (`fast` default).
- `WARM_START_STATIC=0|1` (defaults to on in `fast`, off in `reference`).
- `RECORD_TRACE=0|1` to control outer-iteration trace CSV writing.
- `CONFIG_TAG=<label>` to tag benchmark CSV outputs.
- `SHOCK_INPUT_MODE=identity|toy|csv|mat` for counterfactual runs.
- `LAMBDA_CSV`, `KAPPA_CSV`, `SHOCK_MAT` for external counterfactual shock paths.
- `BASELINE_ANCHOR_FILE` to override the default baseline anchor path.

Benchmark outputs:
- `tasks/cdp4_baseline_solver/output/benchmark_solver_4sector.csv`
- `tasks/cdp4_baseline_dynamics/output/benchmark_4sector.csv`
- `tasks/cdp4_baseline_validate/output/benchmark_validate_4sector.csv`
- `tasks/cdp4_counterfactual_dynamics/output/benchmark_counterfactual_4sector.csv`
- `tasks/cdp4_counterfactual_validate/output/benchmark_validate_counterfactual_4sector.csv`

Trace/parity outputs:
- `tasks/cdp4_baseline_dynamics/output/outer_trace_4sector.csv`
- `tasks/cdp4_baseline_validate/output/parity_by_time_4sector.csv`
- `tasks/cdp4_counterfactual_dynamics/output/outer_trace_counterfactual_4sector.csv`
- `tasks/cdp4_counterfactual_validate/output/parity_by_time_counterfactual_4sector.csv`

Report artifacts (PDF + TeX):
- `tasks/cdp4_counterfactual_validate/output/dynamics_counterfactual_<profile>_<shock>.pdf`
- `tasks/cdp4_counterfactual_validate/output/dynamics_outer_counterfactual_<profile>_<shock>.pdf`
- `tasks/cdp4_counterfactual_validate/output/dynamics_parity_counterfactual_<profile>_<shock>.pdf`
- `tasks/cdp4_counterfactual_validate/output/table_counterfactual_summary_<profile>_<shock>.tex`
- `tasks/cdp4_counterfactual_validate/output/table_counterfactual_validation_<profile>_<shock>.tex`
- `tasks/cdp4_counterfactual_validate/output/table_counterfactual_parity_selected_t_<profile>_<shock>.tex`
- `tasks/cdp4_counterfactual_validate/output/key_econ_impacts_<profile>_<shock>.pdf`
- `tasks/cdp4_counterfactual_validate/output/key_econ_impacts_early_<profile>_<shock>.pdf`
- `tasks/cdp4_counterfactual_validate/output/table_key_econ_selected_t_<profile>_<shock>.tex`
- `tasks/cdp4_counterfactual_validate/output/table_key_econ_window_means_<profile>_<shock>.tex`

Expected behavior bands:
- Smoke profile (`MAX_ITER_DYNAMIC=5`) is for regression safety checks only and is not expected to fully converge.
- Replication profiles are expected to converge (`final_ymax <= 1e-3`) and match MATLAB `Hvectnoshock.mat` at converged-path tolerance (`ynew_max_rel_error <= 1e-3`).
- Counterfactual identity profiles are expected to converge at `final_ymax <= 1e-3` and match the chosen baseline anchor path at `ynew_max_rel_error <= 1e-3`.
- Counterfactual toy profiles are expected to converge and show non-zero deviation from identity, with higher early real wages in the shocked cell.

Replication folders are local-only and intentionally untracked:
- `CDP_codes_four_sectors/`
- `ecta200013-sup-0002-dataandprograms/`
