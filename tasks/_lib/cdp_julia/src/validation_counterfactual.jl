function _cf_max_rel_err(a::AbstractArray{<:Real}, b::AbstractArray{<:Real}; eps::Float64 = 1e-12)
    maximum(abs.(a .- b) ./ max.(abs.(b), eps))
end

function _cf_bool_to_str(x::Bool)
    x ? "PASS" : "FAIL"
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

    t_start = min(2, size(path.mu_path, 3))
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
    push!(rows, (String(mode_label), "mu_row_sum_max_abs_error_t_ge_2", mu_row_err, 1e-8, _cf_bool_to_str(mu_row_err <= 1e-8)))

    nonneg_L = minimum(path.Ldyn)
    push!(rows, (String(mode_label), "ldyn_min", nonneg_L, -1e-10, _cf_bool_to_str(nonneg_L >= -1e-10)))

    DataFrame(rows)
end

function validate_counterfactual_identity_4sector(path::CounterfactualPath4,
                                                  baseline_anchor_y::Matrix{Float64};
                                                  rel_tol::Float64 = 1e-3,
                                                  mode_label::AbstractString = "identity")
    core = validate_counterfactual_core_4sector(path; dyn_tol = rel_tol, mode_label = mode_label)
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
