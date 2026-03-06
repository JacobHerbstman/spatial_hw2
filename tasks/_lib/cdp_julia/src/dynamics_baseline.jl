_safe_ratio(a::Float64, b::Float64) = b == 0.0 ? 1.0 : a / b

function _fill_col_weights!(w::Vector{Float64}, hvect::Matrix{Float64}, t::Int, beta::Float64)
    @inbounds for i in eachindex(w)
        w[i] = hvect[i, t] ^ beta
    end
    w
end

function _scale_cols_normalize_rows!(dest::AbstractMatrix{Float64}, src::AbstractMatrix{Float64},
                                     col_weights::Vector{Float64}, row_sums::Vector{Float64};
                                     threaded::Bool = false)
    nrows, ncols = size(dest)
    if threaded && Threads.nthreads() > 1
        Threads.@threads for i in 1:nrows
            rs = 0.0
            @inbounds for j in 1:ncols
                v = src[i, j] * col_weights[j]
                dest[i, j] = v
                rs += v
            end
            row_sums[i] = rs
            inv_rs = 1.0 / rs
            @inbounds for j in 1:ncols
                dest[i, j] *= inv_rs
            end
        end
    else
        @inbounds for i in 1:nrows
            rs = 0.0
            for j in 1:ncols
                v = src[i, j] * col_weights[j]
                dest[i, j] = v
                rs += v
            end
            row_sums[i] = rs
            inv_rs = 1.0 / rs
            for j in 1:ncols
                dest[i, j] *= inv_rs
            end
        end
    end
    dest
end

function _run_profile(params::ModelParams, profile_override::Union{Nothing, Symbol})
    if isnothing(profile_override)
        return _check_profile(params.profile)
    end
    _check_profile(profile_override)
end

function _anderson_reset!(acc::AndersonAccelerator)
    acc.k = 0
    acc
end

function _anderson_simple_step!(acc::AndersonAccelerator, x_vec::AbstractVector{Float64})
    @inbounds for i in 1:acc.n
        x_vec[i] = x_vec[i] + acc.beta * acc.f_curr[i]
    end
    x_vec
end

function _relax_hvect!(hvect::Matrix{Float64}, ynew::Matrix{Float64}, relax::Float64)
    keep = 1.0 - relax
    @inbounds for i in eachindex(hvect)
        hvect[i] = relax * ynew[i] + keep * hvect[i]
    end
    hvect
end

function _anderson_update!(acc::AndersonAccelerator, x::Matrix{Float64}, g::Matrix{Float64})
    if length(x) != acc.n || length(g) != acc.n
        error("Anderson state size mismatch: expected n=$(acc.n), got $(length(x)) and $(length(g)).")
    end

    x_vec = vec(x)
    g_vec = vec(g)

    @inbounds for i in 1:acc.n
        acc.f_curr[i] = g_vec[i] - x_vec[i]
    end

    acc.k += 1
    col = mod1(acc.k, acc.m)
    @views acc.x_hist[:, col] .= x_vec
    @views acc.f_hist[:, col] .= acc.f_curr

    if acc.k <= acc.m
        _anderson_simple_step!(acc, x_vec)
        return false
    end

    k_eff = min(acc.k, acc.m)
    start_iter = acc.k - k_eff + 1
    @inbounds for j in 1:k_eff
        acc.active_cols[j] = mod1(start_iter + j - 1, acc.m)
    end

    p = k_eff + 1
    K = acc.kkt
    rhs = acc.rhs
    sol = acc.sol

    @inbounds begin
        for i in 1:p, j in 1:p
            K[i, j] = 0.0
        end
        for i in 1:k_eff
            ci = acc.active_cols[i]
            for j in i:k_eff
                cj = acc.active_cols[j]
                s = 0.0
                for t in 1:acc.n
                    s += acc.f_hist[t, ci] * acc.f_hist[t, cj]
                end
                if i == j
                    s += acc.ridge
                end
                K[i, j] = s
                K[j, i] = s
            end
            K[i, p] = 1.0
            K[p, i] = 1.0
            rhs[i] = 0.0
        end
        K[p, p] = 0.0
        rhs[p] = 1.0
    end

    solve_ok = true
    try
        sol[1:p] .= K[1:p, 1:p] \ rhs[1:p]
    catch
        solve_ok = false
    end

    if !solve_ok
        _anderson_simple_step!(acc, x_vec)
        return false
    end

    @inbounds for i in 1:acc.n
        v = 0.0
        for j in 1:k_eff
            cj = acc.active_cols[j]
            αj = sol[j]
            v += αj * (acc.x_hist[i, cj] + acc.beta * acc.f_hist[i, cj])
        end
        acc.x_next[i] = v
    end

    if !all(isfinite, acc.x_next)
        _anderson_simple_step!(acc, x_vec)
        return false
    end

    x_vec .= acc.x_next
    true
end

function run_baseline_4sector(base::BaseState4, params::ModelParams; time_horizon::Int = 200,
                              y_init::Union{Nothing, Matrix{Float64}} = nothing,
                              workspace::Union{Nothing, BaselineWorkspace4} = nothing,
                              profile_override::Union{Nothing, Symbol} = nothing,
                              trace_path::Union{Nothing, AbstractString} = nothing)
    J, N, R = base.dims.J, base.dims.N, base.dims.R
    RJ = R * J
    profile = _run_profile(params, profile_override)
    dynamic_threaded = (profile == :reference) ? false : (params.use_threads && params.threads_dynamic)
    warm_start_static = false
    reset_price_guess = true
    hvect_relax = 0.5
    use_anderson = false

    Hvect = isnothing(y_init) ? ones(RJ, time_horizon) : copy(y_init)
    if size(Hvect) != (RJ, time_horizon)
        error("y_init has size $(size(Hvect)); expected ($(RJ), $(time_horizon)).")
    end
    mu_path = zeros(RJ, RJ, time_horizon)
    Ldyn = zeros(J, R, time_horizon)
    realwages = ones(J, N, time_horizon)
    static_residuals = zeros(time_horizon)
    static_iterations = zeros(Int, time_horizon)

    ws = if isnothing(workspace) || size(workspace.ynew, 2) != time_horizon || length(workspace.hpow) != RJ ||
            size(workspace.om_guesses, 3) != time_horizon
        BaselineWorkspace4(base; time_horizon = time_horizon)
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
    acc = AndersonAccelerator(RJ * time_horizon; m = 5, beta = hvect_relax, ridge = 1e-10)
    _anderson_reset!(acc)

    Ynew_last = copy(Hvect)
    converged = false
    final_ymax = Inf

    iter = 1
    while (iter <= params.max_iter_dynamic)
        println("Baseline outer iteration $(iter)")

        fill!(static_residuals, 0.0)
        fill!(static_iterations, 0)

        _fill_col_weights!(ws.col_weights, Hvect, 2, params.beta)
        _scale_cols_normalize_rows!(
            view(mu_path, :, :, 1),
            base.mu0,
            ws.col_weights,
            ws.row_sums;
            threaded = dynamic_threaded,
        )

        for t in 1:(time_horizon - 2)
            _fill_col_weights!(ws.col_weights, Hvect, t + 2, params.beta)
            _scale_cols_normalize_rows!(
                view(mu_path, :, :, t + 1),
                view(mu_path, :, :, t),
                ws.col_weights,
                ws.row_sums;
                threaded = dynamic_threaded,
            )
        end

        Ldyn[:, :, 1] .= reshape(base.L0, J, R)
        ws.lvec .= reshape(base.L0, RJ)
        mul!(ws.lnext, transpose(view(mu_path, :, :, 1)), ws.lvec)
        Ldyn[:, :, 2] .= reshape(ws.lnext, J, R)

        for t in 2:(time_horizon - 1)
            ws.lvec .= reshape(view(Ldyn, :, :, t), RJ)
            mul!(ws.lnext, transpose(view(mu_path, :, :, t)), ws.lvec)
            Ldyn[:, :, t + 1] .= reshape(ws.lnext, J, R)
        end
        Ldyn[:, :, time_horizon] .= 0.0

        VALjn0 = copy(base.VALjn00)
        VARjn0 = copy(base.VARjn00)
        Sn = copy(base.Sn00)
        Din = copy(base.Din00)

        fill!(ws.kappa_hat, 1.0)
        fill!(ws.lambda_hat, 1.0)

        for t in 1:(time_horizon - 2)
            if t % 25 == 0
                println("  Solving temporary equilibrium at t=$(t)")
            end
            fill!(ws.snp, 0.0)
            fill!(ws.ljn_hat, 1.0)
            @inbounds for r in 1:R, j in 1:J
                ws.ljn_hat[j, r] = _safe_ratio(Ldyn[j, r, t + 1], Ldyn[j, r, t])
            end

            if warm_start_static
                @views ws.om_init .= ws.om_guesses[:, :, t]
            else
                fill!(ws.om_init, 1.0)
            end

            temp = solve_temporary_equilibrium_inplace!(
                base,
                ws.ljn_hat,
                VARjn0,
                VALjn0,
                Din,
                ws.snp;
                kappa_hat = ws.kappa_hat,
                lambda_hat = ws.lambda_hat,
                params = params,
                om_init = ws.om_init,
                workspace = ws.temp_ws,
                reset_price_guess = reset_price_guess,
            )

            VALjn0 .= temp.VALjn
            VARjn0 .= temp.VARjn
            Sn .= temp.Sn
            Din .= temp.Din
            if warm_start_static
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

        for t in 2:(time_horizon - 1)
            idx = 1
            @inbounds for r in 1:R, j in 1:J
                ws.lvec[idx] = realwages[j, r, t] ^ (1.0 / params.nu)
                ws.hpow[idx] = Hvect[idx, t + 1] ^ params.beta
                idx += 1
            end
            # MATLAB reference: Y_i,t = rw_i,t * sum_j mu_{i,j,t-1} * H_j,t+1^beta
            mul!(ws.ytmp, view(mu_path, :, :, t - 1), ws.hpow)
            @inbounds for i in 1:RJ
                ws.ynew[i, t] = ws.lvec[i] * ws.ytmp[i]
            end
        end

        fill!(ws.checky, 0.0)
        @inbounds for t in 2:time_horizon
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

        if Ymax <= params.tol_dynamic
            converged = true
            break
        elseif use_anderson
            _anderson_update!(acc, Hvect, ws.ynew)
        else
            _relax_hvect!(Hvect, ws.ynew, hvect_relax)
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

    BaselinePath4(
        Ynew_last,
        mu_path,
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
    )
end
