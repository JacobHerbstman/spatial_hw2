using Pkg
Pkg.activate(normpath(joinpath(@__DIR__, "../../_lib/cdp_julia")))

using CDPJulia
using JLD2
using CSV
using DataFrames
using Statistics

function _env_bool(name::String, default::Bool)
    raw = lowercase(get(ENV, name, default ? "1" : "0"))
    raw in ("1", "true", "yes")
end

function _env_profile()
    raw = lowercase(get(ENV, "PROFILE", "fast"))
    if raw == "reference"
        return :reference
    end
    if raw == "fast"
        return :fast
    end
    error("Unsupported PROFILE=$(raw). Use fast or reference.")
end

profile = _env_profile()
mode_label_default = profile == :reference ? "replication_reference" : "replication_fast"
mode_label = get(ENV, "VALIDATION_MODE", mode_label_default)
baseline_path_file = get(ENV, "BASELINE_OUTPUT_FILE", "../input/baseline_4sector_path.jld2")
rel_tol = parse(Float64, get(ENV, "REL_TOL", "1e-6"))

jld = load(baseline_path_file)
path = jld["path"]
matlab_ynew = load_matlab_ynew("../input/Hvectnoshock.mat")

max_iter_dynamic = parse(Int, get(ENV, "MAX_ITER_DYNAMIC", "1000"))
max_iter_static = parse(Int, get(ENV, "MAX_ITER_STATIC", "2000"))
use_threads = _env_bool("USE_THREADS", false)
threads_dynamic = _env_bool("THREADS_DYNAMIC", false)
threads_static = _env_bool("THREADS_STATIC", false)
config_tag = get(ENV, "CONFIG_TAG", "default")
record_trace = _env_bool("RECORD_TRACE", true)

report = DataFrame()
parity_time = DataFrame()
delta = 0.0
path_rerun = path

validate_stats = @timed begin
    report = validate_baseline_4sector(path, matlab_ynew; rel_tol = rel_tol, mode_label = mode_label)
    parity_time = parity_by_time(path, matlab_ynew; mode_label = mode_label)

    # Deterministic rerun check
    base = load_base_state_4sector("../input/Base_year_four_sectors.mat")
    params = default_model_params(
        base;
        tol_dynamic = 1e-3,
        max_iter_dynamic = max_iter_dynamic,
        max_iter_static = max_iter_static,
        use_threads = use_threads,
        threads_dynamic = threads_dynamic,
        threads_static = threads_static,
        profile = profile,
        record_trace = record_trace,
    )
    rerun_trace_path = "../output/outer_trace_4sector_validate_rerun.csv"
    path_rerun = run_baseline_4sector(
        base,
        params;
        time_horizon = 200,
        profile_override = profile,
        trace_path = rerun_trace_path,
    )
    delta = deterministic_delta(path, path_rerun)
end

det_row = DataFrame(
    mode = [mode_label],
    check = ["deterministic_rerun_max_abs_delta"],
    value = [delta],
    threshold = [1e-10],
    status = [delta <= 1e-10 ? "PASS" : "FAIL"],
)

full_report = vcat(report, det_row)
CSV.write("../output/validation_report_4sector.csv", full_report)
CSV.write("../output/validation_report_4sector_$(String(profile)).csv", full_report)
CSV.write("../output/parity_by_time_4sector.csv", parity_time)
CSV.write("../output/parity_by_time_4sector_$(String(profile)).csv", parity_time)

bench = DataFrame(
    stage = ["validate"],
    wall_seconds = [validate_stats.time],
    alloc_bytes = [Float64(validate_stats.bytes)],
    gc_seconds = [validate_stats.gctime],
    iterations_static = [mean(path_rerun.static_iterations)],
    iterations_dynamic = [Float64(path_rerun.iterations)],
    converged = [path_rerun.converged ? 1.0 : 0.0],
    final_ymax = [path_rerun.final_ymax],
    profile = [String(profile)],
    validation_mode = [mode_label],
    config_tag = [config_tag],
)
CSV.write("../output/benchmark_validate_4sector.csv", bench)
CSV.write("../output/benchmark_validate_4sector_$(String(profile)).csv", bench)
println("Wrote validation report to ../output/validation_report_4sector.csv")
