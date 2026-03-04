# cdp4_baseline_dynamics

Runs the full 4-sector baseline dynamic recursion (no counterfactual).

Outputs:
- `baseline_4sector_path.jld2`
- `baseline_4sector_path_fast.jld2`
- `baseline_4sector_path_reference.jld2`
- `summary_4sector.csv`
- `summary_4sector_fast.csv`
- `summary_4sector_reference.csv`
- `outer_trace_4sector.csv`
- `outer_trace_4sector_fast.csv`
- `outer_trace_4sector_reference.csv`

Optional runtime controls:
- `MAX_ITER_STATIC` (default `2000`)
- `MAX_ITER_DYNAMIC` (default `1000`)
- `PROFILE=fast|reference` (`fast` default)
- `WARM_START_STATIC=0|1` (`fast` defaults to warm starts, `reference` defaults off)
- `RECORD_TRACE=0|1` (default `1` in task runners)

Make targets:
- `make -C code smoke`
- `make -C code replication_reference`
- `make -C code replication_fast`
