function _cf_max_rel_err(a::AbstractArray{<:Real}, b::AbstractArray{<:Real}; eps::Float64 = 1e-12)
    maximum(abs.(a .- b) ./ max.(abs.(b), eps))
end

function _cf_bool_to_str(x::Bool)
    x ? "PASS" : "FAIL"
end

function _cf_max_abs_diff(a::AbstractArray{<:Real}, b::AbstractArray{<:Real})
    maximum(abs.(a .- b))
end

function _cf_max_abs_diff_t(a::AbstractArray{<:Real, 3}, b::AbstractArray{<:Real, 3}, t_rng)
    isempty(t_rng) ? 0.0 : maximum(abs.(view(a, :, :, t_rng) .- view(b, :, :, t_rng)))
end

function _cf_active_shock_window(shocks::CounterfactualShocks4; tol::Float64 = 1e-12)
    first_t = nothing
    last_t = nothing
    T = size(shocks.lambda_hat, 3)
    @inbounds for t in 1:T
        λ_active = maximum(abs.(view(shocks.lambda_hat, :, :, t) .- 1.0)) > tol
        κ_active = maximum(abs.(view(shocks.kappa_hat, :, :, t) .- 1.0)) > tol
        if λ_active || κ_active
            if isnothing(first_t)
                first_t = t
            end
            last_t = t
        end
    end
    (first_t = first_t, last_t = last_t)
end

function _cf_rows()
    Vector{NamedTuple{(:mode, :check, :value, :threshold, :status),
                      Tuple{String, String, Float64, Float64, String}}}()
end

function validate_counterfactual_core_4sector(path::CounterfactualPath4;
                                              dyn_tol::Float64 = 1e-3,
                                              mode_label::AbstractString = "counterfactual")
    rows = _cf_rows()

    push!(rows, (String(mode_label), "converged", path.converged ? 1.0 : 0.0, 1.0, _cf_bool_to_str(path.converged)))
    push!(rows, (String(mode_label), "final_ymax", path.final_ymax, dyn_tol, _cf_bool_to_str(path.final_ymax <= dyn_tol)))

    finite_ynew = all(isfinite, path.Ynew)
    finite_mu = all(isfinite, path.mu_path)
    finite_L = all(isfinite, path.Ldyn)
    finite_rw = all(isfinite, path.realwages)

    push!(rows, (String(mode_label), "finite_ynew", finite_ynew ? 1.0 : 0.0, 1.0, _cf_bool_to_str(finite_ynew)))
    push!(rows, (String(mode_label), "finite_mu_path", finite_mu ? 1.0 : 0.0, 1.0, _cf_bool_to_str(finite_mu)))
    push!(rows, (String(mode_label), "finite_Ldyn", finite_L ? 1.0 : 0.0, 1.0, _cf_bool_to_str(finite_L)))
    push!(rows, (String(mode_label), "finite_realwages", finite_rw ? 1.0 : 0.0, 1.0, _cf_bool_to_str(finite_rw)))

    t_start = min(1, size(path.mu_path, 3))
    t_last = max(t_start, size(path.mu_path, 3) - 1)
    mu_row_err = 0.0
    @inbounds for t in t_start:t_last
        for i in axes(path.mu_path, 1)
            dev = abs(sum(path.mu_path[i, :, t]) - 1.0)
            if dev > mu_row_err
                mu_row_err = dev
            end
        end
    end
    push!(rows, (String(mode_label), "mu_row_sum_max_abs_error_t_lt_T", mu_row_err, 1e-8, _cf_bool_to_str(mu_row_err <= 1e-8)))

    nonneg_L = minimum(path.Ldyn)
    push!(rows, (String(mode_label), "ldyn_min", nonneg_L, -1e-10, _cf_bool_to_str(nonneg_L >= -1e-10)))

    DataFrame(rows)
end

function validate_counterfactual_identity_4sector(path::CounterfactualPath4,
                                                  baseline_anchor_y::Matrix{Float64};
                                                  dyn_tol::Float64 = 1e-3,
                                                  rel_tol::Float64 = 1e-3,
                                                  mode_label::AbstractString = "identity")
    core = validate_counterfactual_core_4sector(path; dyn_tol = dyn_tol, mode_label = mode_label)
    rows = _cf_rows()

    ynew_abs = maximum(abs.(path.Ynew .- baseline_anchor_y))
    ynew_rel = _cf_max_rel_err(path.Ynew, baseline_anchor_y)

    push!(rows, (String(mode_label), "ynew_max_abs_error", ynew_abs, rel_tol,
                _cf_bool_to_str(ynew_abs <= max(rel_tol, 1e-8))))
    push!(rows, (String(mode_label), "ynew_max_rel_error", ynew_rel, rel_tol,
                _cf_bool_to_str(ynew_rel <= rel_tol)))

    vcat(core, DataFrame(rows))
end

function parity_by_time_counterfactual(path::CounterfactualPath4, target_y::Matrix{Float64};
                                       mode_label::AbstractString = "counterfactual")
    T = size(path.Ynew, 2)
    max_abs = zeros(T)
    mean_abs = zeros(T)
    @inbounds for t in 1:T
        dt = abs.(path.Ynew[:, t] .- target_y[:, t])
        max_abs[t] = maximum(dt)
        mean_abs[t] = mean(dt)
    end
    DataFrame(
        mode = fill(String(mode_label), T),
        t = collect(1:T),
        max_abs_error_t = max_abs,
        mean_abs_error_t = mean_abs,
    )
end

function deterministic_delta_counterfactual(path_a::CounterfactualPath4,
                                            path_b::CounterfactualPath4)
    maximum(abs.(path_a.Ynew .- path_b.Ynew))
end

function fast_reference_delta_counterfactual(path_fast::CounterfactualPath4,
                                             path_reference::CounterfactualPath4)
    maximum(abs.(path_fast.Ynew .- path_reference.Ynew))
end

function validate_counterfactual_response_4sector(path::CounterfactualPath4,
                                                  identity_path::CounterfactualPath4;
                                                  shocks::Union{Nothing, CounterfactualShocks4} = nothing,
                                                  response_tol::Float64 = 1e-12,
                                                  require_t1_response::Bool = false,
                                                  mode_label::AbstractString = "counterfactual")
    if size(path.Ynew) != size(identity_path.Ynew) ||
       size(path.mu_path) != size(identity_path.mu_path) ||
       size(path.Ldyn) != size(identity_path.Ldyn) ||
       size(path.realwages) != size(identity_path.realwages)
        error("Counterfactual and identity paths have mismatched dimensions.")
    end

    rows = _cf_rows()

    max_abs_Y = _cf_max_abs_diff(path.Ynew, identity_path.Ynew)
    max_abs_rw = _cf_max_abs_diff(path.realwages, identity_path.realwages)
    max_abs_L = _cf_max_abs_diff(path.Ldyn, identity_path.Ldyn)
    max_abs_mu = _cf_max_abs_diff(path.mu_path, identity_path.mu_path)
    max_abs_mu_t1 = _cf_max_abs_diff_t(path.mu_path, identity_path.mu_path, 1:1)
    max_abs_L_t2 = size(path.Ldyn, 3) < 2 ? 0.0 : _cf_max_abs_diff_t(path.Ldyn, identity_path.Ldyn, 2:2)

    push!(rows, (String(mode_label), "max_abs_Y_diff_vs_identity", max_abs_Y, response_tol, _cf_bool_to_str(max_abs_Y > response_tol)))
    push!(rows, (String(mode_label), "max_abs_realwage_diff_vs_identity", max_abs_rw, response_tol, _cf_bool_to_str(max_abs_rw > response_tol)))
    push!(rows, (String(mode_label), "max_abs_Ldyn_diff_vs_identity", max_abs_L, response_tol, _cf_bool_to_str(max_abs_L > response_tol)))
    push!(rows, (String(mode_label), "max_abs_mu_diff_vs_identity", max_abs_mu, response_tol, _cf_bool_to_str(max_abs_mu > response_tol)))
    push!(rows, (String(mode_label), "max_abs_mu_t1_diff_vs_identity", max_abs_mu_t1, response_tol, _cf_bool_to_str(max_abs_mu_t1 > response_tol)))
    push!(rows, (String(mode_label), "max_abs_L_t2_diff_vs_identity", max_abs_L_t2, response_tol, _cf_bool_to_str(max_abs_L_t2 > response_tol)))

    stale_path = (max_abs_Y > response_tol || max_abs_rw > response_tol) &&
                 max_abs_L <= response_tol &&
                 max_abs_mu <= response_tol
    push!(rows, (String(mode_label), "stale_path_response_gate", stale_path ? 0.0 : 1.0, 1.0, _cf_bool_to_str(!stale_path)))

    response_gate = max_abs_L > response_tol && max_abs_mu > response_tol
    push!(rows, (String(mode_label), "nonidentity_response_gate", response_gate ? 1.0 : 0.0, 1.0, _cf_bool_to_str(response_gate)))

    t1_response_gate = !require_t1_response || (max_abs_mu_t1 > response_tol && max_abs_L_t2 > response_tol)
    push!(rows, (String(mode_label), "t1_response_gate", t1_response_gate ? 1.0 : 0.0, 1.0, _cf_bool_to_str(t1_response_gate)))

    if !isnothing(shocks)
        active = _cf_active_shock_window(shocks; tol = response_tol)
        if !isnothing(active.first_t) && active.first_t > 1
            # Temporary-equilibrium shocks at period t affect path objects dated t+1.
            response_start_t = min(size(path.Ynew, 2), active.first_t + 1)
            post_rng = response_start_t:(size(path.Ynew, 2) - 1)

            post_L = _cf_max_abs_diff_t(path.Ldyn, identity_path.Ldyn, post_rng)
            post_mu = _cf_max_abs_diff_t(path.mu_path, identity_path.mu_path, post_rng)

            push!(rows, (String(mode_label), "post_activation_max_abs_Ldyn_diff_vs_identity", post_L, response_tol, _cf_bool_to_str(post_L > response_tol)))
            push!(rows, (String(mode_label), "post_activation_max_abs_mu_diff_vs_identity", post_mu, response_tol, _cf_bool_to_str(post_mu > response_tol)))
        end
    end

    DataFrame(rows)
end
