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

function _load_saved_path(path_file::AbstractString)
    jld = load(path_file)
    if !haskey(jld, "path")
        error("Could not find key `path` in $(path_file)")
    end
    jld["path"]
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

function _status_row(mode_label::String, check::String, value::Float64, threshold::Float64, pass::Bool)
    DataFrame(
        mode = [mode_label],
        check = [check],
        value = [value],
        threshold = [threshold],
        status = [pass ? "PASS" : "FAIL"],
    )
end

profile = _env_profile()
time_horizon = parse(Int, get(ENV, "TIME_HORIZON", "200"))
validation_kind = lowercase(get(ENV, "VALIDATION_KIND", "identity"))
shock_name = get(ENV, "SHOCK_NAME", get(ENV, "SHOCK_INPUT_MODE", "identity"))
mode_label_default = validation_kind == "identity" ? (profile == :reference ? "identity_reference" : "identity_fast") :
    (profile == :reference ? "toy_reference" : "toy_fast")
mode_label = get(ENV, "VALIDATION_MODE", mode_label_default)
rel_tol = parse(Float64, get(ENV, "REL_TOL", "1e-3"))
fast_ref_tol = parse(Float64, get(ENV, "FAST_REF_TOL", "1e-4"))
config_tag = get(ENV, "CONFIG_TAG", "default")

counterfactual_output_file = get(ENV, "COUNTERFACTUAL_OUTPUT_FILE", "../input/counterfactual_4sector_path.jld2")
baseline_anchor_file = get(ENV, "BASELINE_ANCHOR_FILE", "../input/baseline_4sector_path_reference.jld2")
identity_output_file = get(ENV, "IDENTITY_OUTPUT_FILE", "")
reference_output_file = get(ENV, "REFERENCE_OUTPUT_FILE", "")

path = _load_saved_path(counterfactual_output_file)
baseline_anchor_y = load_baseline_anchor_y(baseline_anchor_file)

report = DataFrame()
parity_time = DataFrame()
delta = NaN
fast_ref_delta = NaN
path_rerun = path

max_iter_dynamic = parse(Int, get(ENV, "MAX_ITER_DYNAMIC", "1000"))
max_iter_static = parse(Int, get(ENV, "MAX_ITER_STATIC", "2000"))
use_threads = _env_bool("USE_THREADS", false)
threads_dynamic = _env_bool("THREADS_DYNAMIC", false)
threads_static = _env_bool("THREADS_STATIC", false)
record_trace = _env_bool("RECORD_TRACE", true)
warm_start_static = _env_optional_bool("WARM_START_STATIC")

validate_stats = @timed begin
    if validation_kind == "identity"
        report = validate_counterfactual_identity_4sector(path, baseline_anchor_y; rel_tol = rel_tol, mode_label = mode_label)
        parity_time = parity_by_time_counterfactual(path, baseline_anchor_y; mode_label = mode_label)
    elseif validation_kind == "toy"
        report = validate_counterfactual_core_4sector(path; dyn_tol = rel_tol, mode_label = mode_label)

        if !isempty(identity_output_file) && isfile(identity_output_file)
            identity_path = _load_saved_path(identity_output_file)
            y_delta = maximum(abs.(path.Ynew .- identity_path.Ynew))
            report = vcat(report, _status_row(mode_label, "toy_ynew_nonzero_vs_identity", y_delta, 1e-8, y_delta > 1e-8))

            t_hi = min(21, size(path.realwages, 3))
            toy_rw = mean(path.realwages[1, 1, 2:t_hi])
            id_rw = mean(identity_path.realwages[1, 1, 2:t_hi])
            rw_gap = toy_rw - id_rw
            report = vcat(report, _status_row(mode_label, "toy_rw_early_shocked_cell_gap", rw_gap, 0.0, rw_gap > 0.0))

            parity_time = parity_by_time_counterfactual(path, identity_path.Ynew; mode_label = mode_label)
        else
            parity_time = parity_by_time_counterfactual(path, baseline_anchor_y; mode_label = mode_label)
        end
    else
        error("Unsupported VALIDATION_KIND=$(validation_kind). Use identity or toy.")
    end

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
        warm_start_static = warm_start_static,
        record_trace = record_trace,
    )
    rerun_trace_path = "../output/outer_trace_counterfactual_4sector_validate_rerun.csv"
    rerun_shocks = _load_shocks(base, time_horizon)
    path_rerun = run_counterfactual_4sector(
        base,
        params;
        baseline_anchor_y = baseline_anchor_y,
        shocks = rerun_shocks,
        time_horizon = time_horizon,
        profile_override = profile,
        trace_path = rerun_trace_path,
        shock_name = shock_name,
    )
    delta = deterministic_delta_counterfactual(path, path_rerun)

    det_row = _status_row(mode_label, "deterministic_rerun_max_abs_delta", delta, 1e-10, delta <= 1e-10)
    report = vcat(report, det_row)

    if !isempty(reference_output_file) && isfile(reference_output_file)
        reference_path = _load_saved_path(reference_output_file)
        fast_ref_delta = fast_reference_delta_counterfactual(path, reference_path)
        report = vcat(
            report,
            _status_row(mode_label, "fast_vs_reference_max_abs_delta", fast_ref_delta, fast_ref_tol, fast_ref_delta <= fast_ref_tol),
        )
    end
end

shock_tag = _slug(shock_name)
profile_tag = String(profile)

CSV.write("../output/validation_counterfactual_4sector.csv", report)
CSV.write("../output/validation_counterfactual_4sector_$(profile_tag).csv", report)
CSV.write("../output/validation_counterfactual_4sector_$(profile_tag)_$(shock_tag).csv", report)

CSV.write("../output/parity_by_time_counterfactual_4sector.csv", parity_time)
CSV.write("../output/parity_by_time_counterfactual_4sector_$(profile_tag).csv", parity_time)
CSV.write("../output/parity_by_time_counterfactual_4sector_$(profile_tag)_$(shock_tag).csv", parity_time)

bench = DataFrame(
    stage = ["counterfactual_validate"],
    wall_seconds = [validate_stats.time],
    alloc_bytes = [Float64(validate_stats.bytes)],
    gc_seconds = [validate_stats.gctime],
    iterations_static = [mean(path_rerun.static_iterations)],
    iterations_dynamic = [Float64(path_rerun.iterations)],
    converged = [path_rerun.converged ? 1.0 : 0.0],
    final_ymax = [path_rerun.final_ymax],
    profile = [profile_tag],
    shock_name = [String(shock_name)],
    validation_mode = [mode_label],
    deterministic_delta = [delta],
    fast_reference_delta = [fast_ref_delta],
    config_tag = [config_tag],
)
CSV.write("../output/benchmark_validate_counterfactual_4sector.csv", bench)
CSV.write("../output/benchmark_validate_counterfactual_4sector_$(profile_tag).csv", bench)
CSV.write("../output/benchmark_validate_counterfactual_4sector_$(profile_tag)_$(shock_tag).csv", bench)

println("Wrote counterfactual validation outputs to ../output")
