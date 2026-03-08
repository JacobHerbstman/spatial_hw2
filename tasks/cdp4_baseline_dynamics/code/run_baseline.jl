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

function _env_optional_bool(name::String)
    raw = strip(lowercase(get(ENV, name, "")))
    if isempty(raw)
        return nothing
    end
    raw in ("1", "true", "yes")
end

base = load_base_state_4sector("../input/Base_year_four_sectors.mat")
max_iter_dynamic = parse(Int, get(ENV, "MAX_ITER_DYNAMIC", "1000"))
max_iter_static = parse(Int, get(ENV, "MAX_ITER_STATIC", "2000"))
tol_dynamic = parse(Float64, get(ENV, "TOL_DYNAMIC", "1e-5"))
use_threads = _env_bool("USE_THREADS", false)
threads_dynamic = _env_bool("THREADS_DYNAMIC", false)
threads_static = _env_bool("THREADS_STATIC", false)
profile = _env_profile()
warm_start_static = _env_optional_bool("WARM_START_STATIC")
use_anderson = _env_optional_bool("USE_ANDERSON")
confirm_fixed_point = _env_optional_bool("CONFIRM_FIXED_POINT")
hvect_relax = parse(Float64, get(ENV, "HVECT_RELAX", "0.5"))
record_trace = _env_bool("RECORD_TRACE", true)
config_tag = get(ENV, "CONFIG_TAG", "default")
output_tag = get(ENV, "OUTPUT_TAG", String(profile))
confirm_fixed_point = isnothing(confirm_fixed_point) ? (profile != :reference) : confirm_fixed_point
params = default_model_params(
    base;
    tol_dynamic = tol_dynamic,
    max_iter_dynamic = max_iter_dynamic,
    max_iter_static = max_iter_static,
    use_threads = use_threads,
    threads_dynamic = threads_dynamic,
    threads_static = threads_static,
    profile = profile,
    warm_start_static = warm_start_static,
    use_anderson = use_anderson,
    hvect_relax = hvect_relax,
    record_trace = record_trace,
)

trace_path = "../output/outer_trace_4sector.csv"
run_stats = @timed run_baseline_4sector(
    base,
    params;
    time_horizon = 200,
    profile_override = profile,
    trace_path = trace_path,
    confirm_fixed_point = confirm_fixed_point,
)
path = run_stats.value
profile_tag = String(profile)

@save "../output/baseline_4sector_path.jld2" path
@save "../output/baseline_4sector_path_$(output_tag).jld2" path

summary = DataFrame(
    metric = [
        "profile",
        "iterations",
        "converged",
        "final_ymax",
        "max_abs_Ynew",
        "min_Ldyn",
        "max_static_residual",
        "mean_static_iterations",
        "max_static_iterations",
    ],
    value = [
        profile_tag,
        Float64(path.iterations),
        path.converged ? 1.0 : 0.0,
        path.final_ymax,
        maximum(abs.(path.Ynew)),
        minimum(path.Ldyn),
        maximum(path.static_residuals),
        mean(path.static_iterations),
        maximum(path.static_iterations),
    ],
)

CSV.write("../output/summary_4sector.csv", summary)
CSV.write("../output/summary_4sector_$(output_tag).csv", summary)

bench = DataFrame(
    stage = ["baseline_dynamics"],
    wall_seconds = [run_stats.time],
    alloc_bytes = [Float64(run_stats.bytes)],
    gc_seconds = [run_stats.gctime],
    iterations_static = [mean(path.static_iterations)],
    iterations_dynamic = [Float64(path.iterations)],
    converged = [path.converged ? 1.0 : 0.0],
    final_ymax = [path.final_ymax],
    profile = [profile_tag],
    config_tag = [config_tag],
)
CSV.write("../output/benchmark_4sector.csv", bench)
CSV.write("../output/benchmark_4sector_$(output_tag).csv", bench)
if isfile(trace_path)
    cp(trace_path, "../output/outer_trace_4sector_$(output_tag).csv"; force = true)
end
println("Wrote baseline dynamic outputs to ../output")
