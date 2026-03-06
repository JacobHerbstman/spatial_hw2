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

function _slug(x::AbstractString)
    y = replace(lowercase(strip(String(x))), r"[^a-z0-9]+" => "_")
    isempty(y) ? "default" : y
end

function _load_shocks(base::BaseState4, time_horizon::Int)
    mode = lowercase(get(ENV, "SHOCK_INPUT_MODE", "identity"))
    lambda_csv = get(ENV, "LAMBDA_CSV", "")
    kappa_csv = get(ENV, "KAPPA_CSV", "")
    mat_path = get(ENV, "SHOCK_MAT", "")

    if mode == "identity"
        return load_counterfactual_shocks_4sector(base; time_horizon = time_horizon)
    end
    if mode == "csv"
        lambda_path = isempty(lambda_csv) ? nothing : lambda_csv
        kappa_path = isempty(kappa_csv) ? nothing : kappa_csv
        return load_counterfactual_shocks_4sector(
            base;
            time_horizon = time_horizon,
            lambda_csv = lambda_path,
            kappa_csv = kappa_path,
        )
    end
    if mode == "mat"
        if isempty(mat_path)
            error("SHOCK_MAT is required when SHOCK_INPUT_MODE=mat")
        end
        return load_counterfactual_shocks_4sector(base; time_horizon = time_horizon, mat_path = mat_path)
    end
    if mode == "toy"
        toy_sector = parse(Int, get(ENV, "TOY_SECTOR", "1"))
        toy_region = parse(Int, get(ENV, "TOY_REGION", "1"))
        toy_periods = parse(Int, get(ENV, "TOY_PERIODS", "20"))
        toy_lambda = parse(Float64, get(ENV, "TOY_LAMBDA_MULTIPLIER", "1.05"))
        return toy_counterfactual_shocks_4sector(
            base;
            time_horizon = time_horizon,
            sector = toy_sector,
            region = toy_region,
            periods = toy_periods,
            lambda_multiplier = toy_lambda,
        )
    end

    error("Unsupported SHOCK_INPUT_MODE=$(mode). Use identity, csv, mat, or toy.")
end

base = load_base_state_4sector("../input/Base_year_four_sectors.mat")
time_horizon = parse(Int, get(ENV, "TIME_HORIZON", "200"))
max_iter_dynamic = parse(Int, get(ENV, "MAX_ITER_DYNAMIC", "1000"))
max_iter_static = parse(Int, get(ENV, "MAX_ITER_STATIC", "2000"))
tol_dynamic = parse(Float64, get(ENV, "TOL_DYNAMIC", "1e-5"))
use_threads = _env_bool("USE_THREADS", false)
threads_dynamic = _env_bool("THREADS_DYNAMIC", false)
threads_static = _env_bool("THREADS_STATIC", false)
profile = _env_profile()
warm_start_static = _env_optional_bool("WARM_START_STATIC")
record_trace = _env_bool("RECORD_TRACE", true)
config_tag = get(ENV, "CONFIG_TAG", "default")
shock_name = get(ENV, "SHOCK_NAME", get(ENV, "SHOCK_INPUT_MODE", "identity"))

baseline_anchor_file = get(ENV, "BASELINE_ANCHOR_FILE", "../input/baseline_4sector_path_reference.jld2")
baseline_anchor_y = load_baseline_anchor_y(baseline_anchor_file)

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
    record_trace = record_trace,
)

shocks = _load_shocks(base, time_horizon)

trace_path = "../output/outer_trace_counterfactual_4sector.csv"
run_stats = @timed run_counterfactual_4sector(
    base,
    params;
    baseline_anchor_y = baseline_anchor_y,
    shocks = shocks,
    time_horizon = time_horizon,
    profile_override = profile,
    trace_path = trace_path,
    shock_name = shock_name,
)
path = run_stats.value

profile_tag = String(profile)
shock_tag = _slug(shock_name)

@save "../output/counterfactual_4sector_path.jld2" path
@save "../output/counterfactual_4sector_path_$(profile_tag).jld2" path
@save "../output/counterfactual_4sector_path_$(profile_tag)_$(shock_tag).jld2" path

summary = DataFrame(
    metric = [
        "profile",
        "shock_name",
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
        String(shock_name),
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
CSV.write("../output/summary_counterfactual_4sector.csv", summary)
CSV.write("../output/summary_counterfactual_4sector_$(profile_tag).csv", summary)
CSV.write("../output/summary_counterfactual_4sector_$(profile_tag)_$(shock_tag).csv", summary)

bench = DataFrame(
    stage = ["counterfactual_dynamics"],
    wall_seconds = [run_stats.time],
    alloc_bytes = [Float64(run_stats.bytes)],
    gc_seconds = [run_stats.gctime],
    iterations_static = [mean(path.static_iterations)],
    iterations_dynamic = [Float64(path.iterations)],
    converged = [path.converged ? 1.0 : 0.0],
    final_ymax = [path.final_ymax],
    profile = [profile_tag],
    shock_name = [String(shock_name)],
    config_tag = [config_tag],
)
CSV.write("../output/benchmark_counterfactual_4sector.csv", bench)
CSV.write("../output/benchmark_counterfactual_4sector_$(profile_tag).csv", bench)
CSV.write("../output/benchmark_counterfactual_4sector_$(profile_tag)_$(shock_tag).csv", bench)

if isfile(trace_path)
    cp(trace_path, "../output/outer_trace_counterfactual_4sector_$(profile_tag).csv"; force = true)
    cp(trace_path, "../output/outer_trace_counterfactual_4sector_$(profile_tag)_$(shock_tag).csv"; force = true)
end

println("Wrote counterfactual dynamic outputs to ../output")
