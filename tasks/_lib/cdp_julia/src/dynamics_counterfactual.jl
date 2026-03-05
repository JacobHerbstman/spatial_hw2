function _check_counterfactual_shapes(base::BaseState4, baseline_anchor_y::Matrix{Float64},
                                      shocks::CounterfactualShocks4, time_horizon::Int)
    J, N, R = base.dims.J, base.dims.N, base.dims.R
    RJ = R * J
    if size(baseline_anchor_y) != (RJ, time_horizon)
        error("baseline_anchor_y has size $(size(baseline_anchor_y)); expected ($(RJ), $(time_horizon)).")
    end
    shock_periods = time_horizon - 2
    if size(shocks.kappa_hat) != (J * N, N, shock_periods)
        error("kappa_hat has size $(size(shocks.kappa_hat)); expected ($(J * N), $(N), $(shock_periods)).")
    end
    if size(shocks.lambda_hat) != (J, N, shock_periods)
        error("lambda_hat has size $(size(shocks.lambda_hat)); expected ($(J), $(N), $(shock_periods)).")
    end
end

function _scale_cols!(dest::AbstractMatrix{Float64}, src::AbstractMatrix{Float64}, col_scale::Vector{Float64};
                      threaded::Bool = false)
    nrows, ncols = size(dest)
    if threaded && Threads.nthreads() > 1
        Threads.@threads for i in 1:nrows
            @inbounds for j in 1:ncols
                dest[i, j] = src[i, j] * col_scale[j]
            end
        end
    else
        @inbounds for i in 1:nrows, j in 1:ncols
            dest[i, j] = src[i, j] * col_scale[j]
        end
    end
    dest
end

function _fill_rw_pow!(dest::Vector{Float64}, realwages::Array{Float64,3}, t::Int, nu::Float64,
                       J::Int, R::Int)
    idx = 1
    @inbounds for r in 1:R, j in 1:J
        dest[idx] = realwages[j, r, t] ^ (1.0 / nu)
        idx += 1
    end
    dest
end

function _counterfactual_t1_update!(ws::CounterfactualWorkspace4, base::BaseState4,
                                    Hvect::Matrix{Float64}, params::ModelParams,
                                    t1::Int)
    RJ = length(ws.col_weights)

    _fill_col_weights!(ws.col_weights, Hvect, t1, params.beta)
    @inbounds for j in 1:RJ
        denom = ws.hpow_noshock_t1[j]
        ws.ratio_t1[j] = denom == 0.0 ? 1.0 : ws.col_weights[j] / denom
    end
    _scale_cols!(ws.mu0_tilde, ws.mu00, ws.ratio_t1)

    @inbounds for i in 1:RJ
        den_row = 0.0
        for j in 1:RJ
            hpow = ws.col_weights[j]
            num1 = base.mu0[i, j] * hpow
            ws.special_num1[i, j] = num1

            d = ws.mu0_tilde[i, j]
            denv = d == 0.0 ? 0.0 : (base.mu0[i, j] * ws.mu00[i, j] / d) * hpow
            if !isfinite(denv)
                denv = 0.0
            end
            den_row += denv
        end
        ws.row_sums[i] = den_row
    end

    @inbounds for i in 1:RJ
        den_row = ws.row_sums[i]
        inv_den = den_row == 0.0 ? 0.0 : 1.0 / den_row
        for j in 1:RJ
            ws.mu0_tilde[i, j] = ws.special_num1[i, j] * inv_den
        end
    end

    _fill_col_weights!(ws.hpow, Hvect, t1 + 1, params.beta)
    @inbounds for i in 1:RJ
        acc = 0.0
        for j in 1:RJ
            acc += ws.mu0_tilde[i, j] * ws.hpow[j]
        end
        ws.ynew[i, t1] = acc * ws.lvec[i]
    end

    ws
end

function run_counterfactual_4sector(base::BaseState4, params::ModelParams;
                                    baseline_anchor_y::Matrix{Float64},
                                    shocks::CounterfactualShocks4,
                                    time_horizon::Int = 200,
                                    y_init::Union{Nothing, Matrix{Float64}} = nothing,
                                    workspace::Union{Nothing, CounterfactualWorkspace4} = nothing,
                                    profile_override::Union{Nothing, Symbol} = nothing,
                                    trace_path::Union{Nothing, AbstractString} = nothing,
                                    shock_name::AbstractString = "identity")
    J, N, R = base.dims.J, base.dims.N, base.dims.R
    RJ = R * J

    _check_counterfactual_shapes(base, baseline_anchor_y, shocks, time_horizon)

    profile = _run_profile(params, profile_override)
    dynamic_threaded = (profile == :reference) ? false : (params.use_threads && params.threads_dynamic)
    warm_start_static = (profile == :reference) ? false : params.warm_start_static
    reset_price_guess = (profile == :reference)
    hvect_relax = 0.5

    Hvect = isnothing(y_init) ? ones(RJ, time_horizon) : copy(y_init)
    if size(Hvect) != (RJ, time_horizon)
        error("y_init has size $(size(Hvect)); expected ($(RJ), $(time_horizon)).")
    end

    Ldyn = zeros(J, R, time_horizon)
    realwages = ones(J, N, time_horizon)
    static_residuals = zeros(time_horizon)
    static_iterations = zeros(Int, time_horizon)

    ws = if isnothing(workspace) || size(workspace.ynew, 2) != time_horizon ||
            length(workspace.hpow) != RJ || size(workspace.mu_path, 3) != time_horizon ||
            size(workspace.om_guesses, 3) != time_horizon
        CounterfactualWorkspace4(base; time_horizon = time_horizon)
    else
        workspace
    end

    empty!(ws.outer_ymax)
    empty!(ws.outer_mean_static_iterations)
    empty!(ws.outer_max_static_iterations)
    empty!(ws.outer_max_static_residual)
    fill!(ws.om_prev, 1.0)
    fill!(ws.om_guesses, 1.0)
    fill!(ws.om_init, 1.0)

    _fill_col_weights!(ws.col_weights, baseline_anchor_y, 2, params.beta)
    ws.hpow_noshock_t1 .= ws.col_weights
    _scale_cols_normalize_rows!(
        ws.mu00,
        base.mu0,
        ws.col_weights,
        ws.row_sums;
        threaded = dynamic_threaded,
    )

    Ynew_last = copy(Hvect)
    converged = false
    final_ymax = Inf

    iter = 1
    while iter <= params.max_iter_dynamic
        println("Counterfactual outer iteration $(iter)")

        fill!(static_residuals, 0.0)
        fill!(static_iterations, 0)

        _fill_col_weights!(ws.col_weights, Hvect, 2, params.beta)
        @inbounds for j in 1:RJ
            denom = ws.hpow_noshock_t1[j]
            ws.ratio_t1[j] = denom == 0.0 ? 1.0 : ws.col_weights[j] / denom
        end
        _scale_cols!(ws.mu0_tilde, ws.mu00, ws.ratio_t1; threaded = dynamic_threaded)
        @views ws.mu_path[:, :, 1] .= ws.mu0_tilde

        for t in 1:(time_horizon - 2)
            _fill_col_weights!(ws.col_weights, Hvect, t + 2, params.beta)
            _scale_cols_normalize_rows!(
                view(ws.mu_path, :, :, t + 1),
                view(ws.mu_path, :, :, t),
                ws.col_weights,
                ws.row_sums;
                threaded = dynamic_threaded,
            )
        end

        Ldyn[:, :, 1] .= reshape(base.L0, J, R)
        ws.lvec .= reshape(base.L0, RJ)
        mul!(ws.lnext, transpose(ws.mu00), ws.lvec)
        Ldyn[:, :, 2] .= reshape(ws.lnext, J, R)

        for t in 2:(time_horizon - 1)
            ws.lvec .= reshape(view(Ldyn, :, :, t), RJ)
            mul!(ws.lnext, transpose(view(ws.mu_path, :, :, t)), ws.lvec)
            Ldyn[:, :, t + 1] .= reshape(ws.lnext, J, R)
        end
        Ldyn[:, :, time_horizon] .= 0.0

        ws.VALjn0 .= base.VALjn00
        ws.VARjn0 .= base.VARjn00
        ws.Sn0 .= base.Sn00
        ws.Din0 .= base.Din00

        for t in 1:(time_horizon - 2)
            if t % 25 == 0
                println("  Solving temporary equilibrium at t=$(t)")
            end
            fill!(ws.snp, 0.0)
            fill!(ws.ljn_hat, 1.0)
            @inbounds for r in 1:R, j in 1:J
                ws.ljn_hat[j, r] = _safe_ratio(Ldyn[j, r, t + 1], Ldyn[j, r, t])
            end

            ws.kappa_hat .= view(shocks.kappa_hat, :, :, t)
            ws.lambda_hat .= view(shocks.lambda_hat, :, :, t)

            if warm_start_static
                if t == 1
                    @views ws.om_init .= ws.om_guesses[:, :, t]
                else
                    ws.om_init .= ws.om_prev
                end
            else
                fill!(ws.om_init, 1.0)
            end

            temp = solve_temporary_equilibrium_inplace!(
                base,
                ws.ljn_hat,
                ws.VARjn0,
                ws.VALjn0,
                ws.Din0,
                ws.snp;
                kappa_hat = ws.kappa_hat,
                lambda_hat = ws.lambda_hat,
                params = params,
                om_init = ws.om_init,
                workspace = ws.temp_ws,
                reset_price_guess = reset_price_guess,
            )

            ws.VALjn0 .= temp.VALjn
            ws.VARjn0 .= temp.VARjn
            ws.Sn0 .= temp.Sn
            ws.Din0 .= temp.Din
            if warm_start_static
                ws.om_prev .= temp.om
                @views ws.om_guesses[:, :, t] .= temp.om
            end

            @inbounds for n in 1:N, j in 1:J
                realwages[j, n, t + 1] = temp.wf[j, n] / temp.Pidx[n]
            end
            static_residuals[t + 1] = temp.residual
            static_iterations[t + 1] = temp.iterations
        end

        fill!(ws.ynew, 0.0)
        ws.ynew[:, time_horizon] .= 1.0
        Hvect[:, time_horizon] .= 1.0

        t1 = 2
        _fill_rw_pow!(ws.lvec, realwages, t1, params.nu, J, R)
        _counterfactual_t1_update!(ws, base, Hvect, params, t1)

        for t in (t1 + 1):(time_horizon - 1)
            _fill_rw_pow!(ws.lvec, realwages, t, params.nu, J, R)
            _fill_col_weights!(ws.hpow, Hvect, t + 1, params.beta)
            mul!(ws.ytmp, view(ws.mu_path, :, :, t - 1), ws.hpow)
            @inbounds for i in 1:RJ
                ws.ynew[i, t] = ws.lvec[i] * ws.ytmp[i]
            end
        end

        fill!(ws.checky, 0.0)
        @inbounds for t in 1:time_horizon
            m = 0.0
            for i in 1:RJ
                dev = abs(ws.ynew[i, t] - Hvect[i, t])
                if dev > m
                    m = dev
                end
            end
            ws.checky[t] = m
        end
        Ymax = maximum(ws.checky)

        Ynew_last .= ws.ynew
        final_ymax = Ymax

        t_rng = 2:(time_horizon - 1)
        push!(ws.outer_ymax, Ymax)
        push!(ws.outer_mean_static_iterations, mean(static_iterations[t_rng]))
        push!(ws.outer_max_static_iterations, maximum(static_iterations[t_rng]))
        push!(ws.outer_max_static_residual, maximum(static_residuals[t_rng]))

        @inbounds for t in 1:time_horizon, i in 1:RJ
            Hvect[i, t] = hvect_relax * ws.ynew[i, t] + (1.0 - hvect_relax) * Hvect[i, t]
        end

        if Ymax <= params.tol_dynamic
            converged = true
            break
        end

        iter += 1
    end

    actual_iters = converged ? iter : min(iter - 1, params.max_iter_dynamic)

    if !isnothing(trace_path) && params.record_trace
        trace_df = DataFrame(
            outer_iter = collect(1:length(ws.outer_ymax)),
            Ymax = ws.outer_ymax,
            mean_static_iterations = ws.outer_mean_static_iterations,
            max_static_iterations = ws.outer_max_static_iterations,
            max_static_residual = ws.outer_max_static_residual,
        )
        CSV.write(trace_path, trace_df)
    end

    CounterfactualPath4(
        Ynew_last,
        copy(ws.mu_path),
        Ldyn,
        realwages,
        static_iterations,
        actual_iters,
        converged,
        final_ymax,
        static_residuals,
        copy(ws.outer_ymax),
        copy(ws.outer_mean_static_iterations),
        copy(ws.outer_max_static_iterations),
        copy(ws.outer_max_static_residual),
        profile,
        copy(ws.mu00),
        String(shock_name),
    )
end
