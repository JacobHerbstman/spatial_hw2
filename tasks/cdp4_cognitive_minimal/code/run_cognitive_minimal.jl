using Pkg
Pkg.activate(normpath(joinpath(@__DIR__, "../../_lib/cdp_julia")))

using CDPJulia
using JLD2
using CSV
using DataFrames
using Statistics

const STATE_ORDER = [
    "AL", "AK", "AZ", "AR", "CA", "CO", "CT", "DE", "FL", "GA",
    "HI", "ID", "IL", "IN", "IA", "KS", "KY", "LA", "ME", "MD",
    "MA", "MI", "MN", "MS", "MO", "MT", "NE", "NV", "NH", "NJ",
    "NM", "NY", "NC", "ND", "OH", "OK", "OR", "PA", "RI", "SC",
    "SD", "TN", "TX", "UT", "VT", "VA", "WA", "WV", "WI", "WY",
]

const SECTOR_NAMES = Dict(
    1 => "Manufacturing",
    2 => "Construction",
    3 => "Wholesale/Retail",
    4 => "Services",
)

function _env_bool(name::String, default::Bool)
    raw = lowercase(strip(get(ENV, name, default ? "1" : "0")))
    raw in ("1", "true", "yes")
end

function _env_profile()
    raw = lowercase(strip(get(ENV, "PROFILE", "reference")))
    if raw == "reference"
        return :reference
    elseif raw == "fast"
        return :fast
    end
    error("Unsupported PROFILE=$(raw). Use reference or fast.")
end

function _slug(x::AbstractString)
    y = replace(lowercase(strip(String(x))), r"[^a-z0-9]+" => "_")
    isempty(y) ? "default" : y
end

function _write_ynew_csv(path::AbstractString, Ynew::Matrix{Float64})
    pairs = Pair{Symbol, Any}[(:row_idx => collect(1:size(Ynew, 1)))]
    for t in 1:size(Ynew, 2)
        push!(pairs, Symbol("t_" * lpad(string(t), 3, '0')) => Ynew[:, t])
    end
    df = DataFrame(pairs)
    CSV.write(path, df)
end

function _selected_state_sector_df(base::BaseState4, result, run_name::String)
    selected_times = [t for t in (2, 10, 20, 50, 100, 199) if t <= size(result.Ldyn, 3)]
    rows = NamedTuple[]
    for t in selected_times
        for state_idx in 1:base.dims.R
            for sector_idx in 1:base.dims.J
                emp0 = base.L0[sector_idx, state_idx]
                push!(rows, (
                    run_name = run_name,
                    t = t,
                    state_idx = state_idx,
                    state_abbr = STATE_ORDER[state_idx],
                    sector_idx = sector_idx,
                    sector_name = SECTOR_NAMES[sector_idx],
                    employment = result.Ldyn[sector_idx, state_idx, t],
                    employment_hat_vs_t0 = emp0 == 0.0 ? NaN : result.Ldyn[sector_idx, state_idx, t] / emp0,
                    real_wage = result.realwages[sector_idx, state_idx, t],
                ))
            end
        end
    end
    DataFrame(rows)
end

function _selected_state_aggregate_df(base::BaseState4, result, run_name::String)
    selected_times = [t for t in (2, 10, 20, 50, 100, 199) if t <= size(result.Ldyn, 3)]
    rows = NamedTuple[]
    base_totals = vec(sum(base.L0; dims = 1))
    for t in selected_times
        for state_idx in 1:base.dims.R
            emp_total = sum(result.Ldyn[:, state_idx, t])
            emp0 = base_totals[state_idx]
            push!(rows, (
                run_name = run_name,
                t = t,
                state_idx = state_idx,
                state_abbr = STATE_ORDER[state_idx],
                employment_total = emp_total,
                employment_total_hat_vs_t0 = emp0 == 0.0 ? NaN : emp_total / emp0,
            ))
        end
    end
    DataFrame(rows)
end

base_year_mat = get(ENV, "BASE_YEAR_MAT", "../input/Base_year_four_sectors.mat")
hno_shock_mat = get(ENV, "HNO_SHOCK_MAT", "../input/Hvectnoshock.mat")
lambda_mat = get(ENV, "LAMBDA_MAT", "")
run_name = get(ENV, "RUN_NAME", "identity")
time_horizon = parse(Int, get(ENV, "TIME_HORIZON", "200"))
tol_dynamic = parse(Float64, get(ENV, "TOL_DYNAMIC", "1e-3"))
max_iter_dynamic = parse(Int, get(ENV, "MAX_ITER_DYNAMIC", "1000"))
max_iter_static = parse(Int, get(ENV, "MAX_ITER_STATIC", "2000"))
use_threads = _env_bool("USE_THREADS", false)
threads_dynamic = _env_bool("THREADS_DYNAMIC", false)
threads_static = _env_bool("THREADS_STATIC", false)
record_trace = _env_bool("RECORD_TRACE", true)
profile = _env_profile()

base = load_base_state_4sector(base_year_mat)
Hvect_anchor_full = load_baseline_anchor_y(hno_shock_mat)
if size(Hvect_anchor_full, 2) < time_horizon
    error("Hvect anchor has only $(size(Hvect_anchor_full, 2)) columns, smaller than TIME_HORIZON=$(time_horizon).")
end
Hvect_anchor = Hvect_anchor_full[:, 1:time_horizon]
shocks = if isempty(lambda_mat)
    load_counterfactual_shocks_4sector(base; time_horizon = time_horizon)
else
    load_counterfactual_shocks_4sector(base; time_horizon = time_horizon, mat_path = lambda_mat)
end

params = default_model_params(
    base;
    tol_dynamic = tol_dynamic,
    max_iter_dynamic = max_iter_dynamic,
    max_iter_static = max_iter_static,
    use_threads = use_threads,
    threads_dynamic = threads_dynamic,
    threads_static = threads_static,
    profile = profile,
    warm_start_static = false,
    use_anderson = false,
    hvect_relax = 0.5,
    record_trace = record_trace,
)

run_slug = _slug(run_name)
trace_path = "../output/outer_trace_" * run_slug * ".csv"
run_stats = @timed run_counterfactual_4sector(
    base,
    params;
    baseline_anchor_y = Hvect_anchor,
    shocks = shocks,
    time_horizon = time_horizon,
    profile_override = profile,
    trace_path = trace_path,
    shock_name = run_name,
)
path = run_stats.value
active_static = path.static_iterations[2:(time_horizon - 1)]
active_resid = path.static_residuals[2:(time_horizon - 1)]

@save "../output/counterfactual_path_$(run_slug).jld2" path

summary = DataFrame(
    metric = [
        "run_name",
        "iterations",
        "converged",
        "final_ymax",
        "max_abs_Ynew",
        "min_Ldyn",
        "max_static_residual",
        "mean_static_iterations",
        "max_static_iterations",
        "profile",
    ],
    value = [
        run_name,
        Float64(path.iterations),
        path.converged ? 1.0 : 0.0,
        path.final_ymax,
        maximum(abs.(path.Ynew)),
        minimum(path.Ldyn),
        maximum(active_resid),
        mean(active_static),
        maximum(active_static),
        String(profile),
    ],
)
CSV.write("../output/summary_$(run_slug).csv", summary)

bench = DataFrame(
    stage = ["cognitive_minimal_counterfactual"],
    wall_seconds = [run_stats.time],
    alloc_bytes = [Float64(run_stats.bytes)],
    gc_seconds = [run_stats.gctime],
    iterations_dynamic = [Float64(path.iterations)],
    mean_static_iterations = [mean(active_static)],
    converged = [path.converged ? 1.0 : 0.0],
    final_ymax = [path.final_ymax],
)
CSV.write("../output/benchmark_$(run_slug).csv", bench)

_write_ynew_csv("../output/ynew_matrix_$(run_slug).csv", path.Ynew)
CSV.write("../output/selected_state_sector_$(run_slug).csv", _selected_state_sector_df(base, path, run_name))
CSV.write("../output/selected_state_aggregate_$(run_slug).csv", _selected_state_aggregate_df(base, path, run_name))

println("Wrote minimal cognitive outputs for $(run_name) to ../output")
