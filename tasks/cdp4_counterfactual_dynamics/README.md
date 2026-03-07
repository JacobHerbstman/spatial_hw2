# cdp4_counterfactual_dynamics

Runs CDP-style 4-sector counterfactual dynamics using a baseline anchor path and shock inputs.

Outputs:
- `counterfactual_4sector_path.jld2`
- `counterfactual_4sector_path_<profile>.jld2`
- `counterfactual_4sector_path_<profile>_<shock>.jld2`
- `summary_counterfactual_4sector*.csv`
- `outer_trace_counterfactual_4sector*.csv`
- `benchmark_counterfactual_4sector*.csv`

Runtime controls:
- `PROFILE=fast|reference`
- `MAX_ITER_STATIC`, `MAX_ITER_DYNAMIC`
- `USE_THREADS`, `THREADS_DYNAMIC`, `THREADS_STATIC`
- `WARM_START_STATIC=0|1` (optional override)
- `RECORD_TRACE=0|1`
- `TIME_HORIZON` (default `200`)
- `BASELINE_ANCHOR_FILE` (default `../input/baseline_4sector_path_reference.jld2`)
- `SHOCK_INPUT_MODE=identity|toy|csv|mat`
- `LAMBDA_CSV`, `KAPPA_CSV` (CSV mode)
- `SHOCK_MAT` (MAT mode)
- `SHOCK_NAME`, `CONFIG_TAG`

Make targets:
- `make -C code smoke`
- `make -C code toy_smoke`
- `make -C code identity_reference`
- `make -C code identity_fast`
- `make -C code toy_reference`
- `make -C code toy_fast`
