using Pkg
Pkg.activate(normpath(joinpath(@__DIR__, "../../_lib/cdp_julia")))

using CDPJulia
using MAT
using JLD2
using CSV
using DataFrames
using LinearAlgebra
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

function _fill_col_weights!(w::Vector{Float64}, hvect::Matrix{Float64}, t::Int, beta::Float64)
    @inbounds for i in eachindex(w)
        w[i] = hvect[i, t] ^ beta
    end
    w
end

function _scale_cols_normalize_rows!(dest::AbstractMatrix{Float64}, src::AbstractMatrix{Float64},
                                     col_weights::Vector{Float64}, row_sums::Vector{Float64})
    nrows, ncols = size(dest)
    @inbounds for i in 1:nrows
        rs = 0.0
        for j in 1:ncols
            v = src[i, j] * col_weights[j]
            dest[i, j] = v
            rs += v
        end
        row_sums[i] = rs
        if rs == 0.0
            for j in 1:ncols
                dest[i, j] = 0.0
            end
        else
            inv_rs = 1.0 / rs
            for j in 1:ncols
                dest[i, j] *= inv_rs
            end
        end
    end
    dest
end

function _safe_ratio(a::Float64, b::Float64)
    b == 0.0 ? 1.0 : a / b
end

function _write_outer_trace(trace_path::AbstractString,
                            outer_ymax::Vector{Float64},
                            outer_mean_static_iterations::Vector{Float64},
                            outer_max_static_iterations::Vector{Int},
                            outer_max_static_residual::Vector{Float64})
    df = DataFrame(
        outer_iter = collect(1:length(outer_ymax)),
        Ymax = outer_ymax,
        mean_static_iterations = outer_mean_static_iterations,
        max_static_iterations = outer_max_static_iterations,
        max_static_residual = outer_max_static_residual,
    )
    CSV.write(trace_path, df)
    nothing
end

function _load_lambdas(path::Union{Nothing, String}, J::Int, N::Int, time_horizon::Int)
    lambdas = ones(J, N, time_horizon)
    if isnothing(path) || isempty(path)
        return lambdas
    end
    raw = matread(path)
    if !haskey(raw, "lambdas")
        error("Expected key `lambdas` in $(path).")
    end
    arr = Array{Float64}(raw["lambdas"])
    if size(arr) != size(lambdas)
        error("lambdas in $(path) has size $(size(arr)); expected $(size(lambdas)).")
    end
    lambdas .= arr
    lambdas
end

function _vec_us_realwage_pow!(dest::Vector{Float64}, realwages::Array{Float64, 3}, t::Int,
                               nu::Float64, J::Int, R::Int)
    idx = 1
    @inbounds for r in 1:R, j in 1:J
        dest[idx] = realwages[j, r, t] ^ (1.0 / nu)
        idx += 1
    end
    dest
end

function run_counterfactual_minimal(base::BaseState4, Hvect_anchor::Matrix{Float64},
                                    lambdas::Array{Float64, 3}, params::ModelParams;
                                    trace_path::AbstractString)
    J, N, R = base.dims.J, base.dims.N, base.dims.R
    RJ = J * R
    time_horizon = size(Hvect_anchor, 2)

    if size(Hvect_anchor, 1) != RJ
        error("Hvect anchor has size $(size(Hvect_anchor)); expected ($(RJ), T).")
    end
    if size(lambdas) != (J, N, time_horizon)
        error("lambdas has size $(size(lambdas)); expected ($(J), $(N), $(time_horizon)).")
    end

    Hvect = copy(Hvect_anchor)
    Hvectnoshock = copy(Hvect_anchor)

    row_sums = zeros(RJ)
    hpow = zeros(RJ)
    hpow_noshock_t1 = zeros(RJ)
    ratio_t1 = zeros(RJ)
    mu00 = zeros(RJ, RJ)
    mu0_tilde = zeros(RJ, RJ)
    mu_path = zeros(RJ, RJ, time_horizon)
    special_num1 = zeros(RJ, RJ)

    lvec = zeros(RJ)
    lnext = zeros(RJ)
    ytmp = zeros(RJ)
    ynew = zeros(RJ, time_horizon)
    checky = zeros(time_horizon)

    Ldyn = zeros(J, R, time_horizon)
    realwages = ones(J, N, time_horizon)
    static_iterations = zeros(Int, time_horizon)
    static_residuals = zeros(time_horizon)

    ljn_hat = ones(J, N)
    kappa_hat = ones(J * N, N)
    lambda_hat_t = ones(J, N)
    snp = zeros(N)
    temp_ws = TempEqWorkspace(base)
    om_init = ones(J, N)
    VARjn0 = copy(base.VARjn00)
    VALjn0 = copy(base.VALjn00)
    Din0 = copy(base.Din00)
    Sn0 = copy(base.Sn00)

    outer_ymax = Float64[]
    outer_mean_static_iterations = Float64[]
    outer_max_static_iterations = Int[]
    outer_max_static_residual = Float64[]

    _fill_col_weights!(hpow_noshock_t1, Hvectnoshock, 2, params.beta)
    _scale_cols_normalize_rows!(mu00, base.mu0, hpow_noshock_t1, row_sums)

    Ynew_last = copy(Hvect)
    converged = false
    final_ymax = Inf
    iter = 1

    while (iter <= params.max_iter_dynamic)
        println("Minimal outer iteration $(iter)")

        fill!(realwages, 1.0)
        fill!(static_iterations, 0)
        fill!(static_residuals, 0.0)

        _fill_col_weights!(hpow, Hvect, 2, params.beta)
        @inbounds for j in 1:RJ
            denom = hpow_noshock_t1[j]
            ratio_t1[j] = denom == 0.0 ? 1.0 : hpow[j] / denom
        end
        @inbounds for i in 1:RJ, j in 1:RJ
            mu0_tilde[i, j] = mu00[i, j] * ratio_t1[j]
        end
        @views mu_path[:, :, 1] .= mu0_tilde

        for t in 1:(time_horizon - 2)
            _fill_col_weights!(hpow, Hvect, t + 2, params.beta)
            _scale_cols_normalize_rows!(view(mu_path, :, :, t + 1), view(mu_path, :, :, t), hpow, row_sums)
        end

        @views Ldyn[:, :, 1] .= base.L0
        lvec .= vec(base.L0)
        mul!(lnext, transpose(mu00), lvec)
        @views Ldyn[:, :, 2] .= reshape(lnext, J, R)

        for t in 2:(time_horizon - 1)
            lvec .= vec(view(Ldyn, :, :, t))
            mul!(lnext, transpose(view(mu_path, :, :, t)), lvec)
            @views Ldyn[:, :, t + 1] .= reshape(lnext, J, R)
        end
        @views Ldyn[:, :, time_horizon] .= 0.0

        VARjn0 .= base.VARjn00
        VALjn0 .= base.VALjn00
        Din0 .= base.Din00
        Sn0 .= base.Sn00

        for t in 1:(time_horizon - 2)
            if t % 25 == 0
                println("  Solving temporary equilibrium at t=$(t)")
            end
            fill!(snp, 0.0)
            fill!(ljn_hat, 1.0)
            @inbounds for r in 1:R, j in 1:J
                ljn_hat[j, r] = _safe_ratio(Ldyn[j, r, t + 1], Ldyn[j, r, t])
            end

            fill!(om_init, 1.0)
            lambda_hat_t .= view(lambdas, :, :, t)
            temp = solve_temporary_equilibrium_inplace!(
                base,
                ljn_hat,
                VARjn0,
                VALjn0,
                Din0,
                snp;
                kappa_hat = kappa_hat,
                lambda_hat = lambda_hat_t,
                params = params,
                om_init = om_init,
                workspace = temp_ws,
                reset_price_guess = true,
            )

            VARjn0 .= temp.VARjn
            VALjn0 .= temp.VALjn
            Din0 .= temp.Din
            Sn0 .= temp.Sn

            @inbounds for n in 1:N, j in 1:J
                realwages[j, n, t + 1] = temp.wf[j, n] / temp.Pidx[n]
            end
            static_iterations[t + 1] = temp.iterations
            static_residuals[t + 1] = temp.residual
        end

        fill!(ynew, 0.0)
        @views ynew[:, time_horizon] .= 1.0
        @views Hvect[:, time_horizon] .= 1.0

        t1 = 2
        _fill_col_weights!(hpow, Hvect, t1, params.beta)
        @inbounds for i in 1:RJ, j in 1:RJ
            mu0_tilde[i, j] = mu00[i, j] * ratio_t1[j]
        end
        @inbounds for i in 1:RJ
            den_row = 0.0
            for j in 1:RJ
                num1 = base.mu0[i, j] * hpow[j]
                special_num1[i, j] = num1
                d = mu0_tilde[i, j]
                denv = d == 0.0 ? 0.0 : (base.mu0[i, j] * mu00[i, j] / d) * hpow[j]
                if !isfinite(denv)
                    denv = 0.0
                end
                den_row += denv
            end
            row_sums[i] = den_row
        end
        @inbounds for i in 1:RJ
            inv_den = row_sums[i] == 0.0 ? 0.0 : 1.0 / row_sums[i]
            for j in 1:RJ
                mu0_tilde[i, j] = special_num1[i, j] * inv_den
            end
        end

        _fill_col_weights!(hpow, Hvect, t1 + 1, params.beta)
        _vec_us_realwage_pow!(lvec, realwages, t1, params.nu, J, R)
        @inbounds for i in 1:RJ
            acc = 0.0
            for j in 1:RJ
                acc += mu0_tilde[i, j] * hpow[j]
            end
            ynew[i, t1] = acc * lvec[i]
        end

        for t in (t1 + 1):(time_horizon - 1)
            _vec_us_realwage_pow!(lvec, realwages, t, params.nu, J, R)
            _fill_col_weights!(hpow, Hvect, t + 1, params.beta)
            mul!(ytmp, view(mu_path, :, :, t - 1), hpow)
            @inbounds for i in 1:RJ
                ynew[i, t] = lvec[i] * ytmp[i]
            end
        end

        fill!(checky, 0.0)
        @inbounds for t in 1:time_horizon
            m = 0.0
            for i in 1:RJ
                dev = abs(ynew[i, t] - Hvect[i, t])
                if dev > m
                    m = dev
                end
            end
            checky[t] = m
        end

        Ymax = maximum(checky)
        Ynew_last .= ynew
        final_ymax = Ymax

        t_rng = 2:(time_horizon - 1)
        push!(outer_ymax, Ymax)
        push!(outer_mean_static_iterations, mean(static_iterations[t_rng]))
        push!(outer_max_static_iterations, maximum(static_iterations[t_rng]))
        push!(outer_max_static_residual, maximum(static_residuals[t_rng]))
        _write_outer_trace(trace_path, outer_ymax, outer_mean_static_iterations, outer_max_static_iterations, outer_max_static_residual)

        if Ymax <= params.tol_dynamic
            converged = true
            break
        end

        @inbounds for idx in eachindex(Hvect)
            Hvect[idx] = 0.5 * ynew[idx] + 0.5 * Hvect[idx]
        end
        iter += 1
    end

    actual_iters = converged ? iter : min(iter, params.max_iter_dynamic)
    result = (
        Ynew = copy(Ynew_last),
        mu_path = copy(mu_path),
        Ldyn = copy(Ldyn),
        realwages = copy(realwages),
        static_iterations = copy(static_iterations),
        static_residuals = copy(static_residuals),
        iterations = actual_iters,
        converged = converged,
        final_ymax = final_ymax,
        outer_ymax = copy(outer_ymax),
        outer_mean_static_iterations = copy(outer_mean_static_iterations),
        outer_max_static_iterations = copy(outer_max_static_iterations),
        outer_max_static_residual = copy(outer_max_static_residual),
        mu00 = copy(mu00),
    )
    return result
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
lambdas = _load_lambdas(isempty(lambda_mat) ? nothing : lambda_mat, base.dims.J, base.dims.N, time_horizon)

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
run_stats = @timed run_counterfactual_minimal(base, Hvect_anchor, lambdas, params; trace_path = trace_path)
result = run_stats.value
active_static = result.static_iterations[2:(time_horizon - 1)]
active_resid = result.static_residuals[2:(time_horizon - 1)]

@save "../output/counterfactual_path_$(run_slug).jld2" result

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
        Float64(result.iterations),
        result.converged ? 1.0 : 0.0,
        result.final_ymax,
        maximum(abs.(result.Ynew)),
        minimum(result.Ldyn),
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
    iterations_dynamic = [Float64(result.iterations)],
    mean_static_iterations = [mean(active_static)],
    converged = [result.converged ? 1.0 : 0.0],
    final_ymax = [result.final_ymax],
)
CSV.write("../output/benchmark_$(run_slug).csv", bench)

_write_ynew_csv("../output/ynew_matrix_$(run_slug).csv", result.Ynew)
CSV.write("../output/selected_state_sector_$(run_slug).csv", _selected_state_sector_df(base, result, run_name))
CSV.write("../output/selected_state_aggregate_$(run_slug).csv", _selected_state_aggregate_df(base, result, run_name))

println("Wrote minimal cognitive outputs for $(run_name) to ../output")
