function _max_rel_err(a::AbstractArray{<:Real}, b::AbstractArray{<:Real}; eps::Float64 = 1e-12)
    maximum(abs.(a .- b) ./ max.(abs.(b), eps))
end

function _bool_to_str(x::Bool)
    x ? "PASS" : "FAIL"
end

function validate_baseline_4sector(path::BaselinePath4, matlab_ynew::Matrix{Float64};
                                   rel_tol::Float64 = 1e-6,
                                   mode_label::AbstractString = "smoke")
    rows = Vector{NamedTuple{(:mode, :check, :value, :threshold, :status), Tuple{String, String, Float64, Float64, String}}}()

    ynew_abs = maximum(abs.(path.Ynew .- matlab_ynew))
    ynew_rel = _max_rel_err(path.Ynew, matlab_ynew)
    ynew_ok = ynew_rel <= rel_tol

    push!(rows, (String(mode_label), "ynew_max_abs_error", ynew_abs, rel_tol, _bool_to_str(ynew_abs <= max(rel_tol, 1e-8))))
    push!(rows, (String(mode_label), "ynew_max_rel_error", ynew_rel, rel_tol, _bool_to_str(ynew_ok)))

    finite_ynew = all(isfinite, path.Ynew)
    finite_mu = all(isfinite, path.mu_path)
    finite_L = all(isfinite, path.Ldyn)
    finite_rw = all(isfinite, path.realwages)

    push!(rows, (String(mode_label), "finite_ynew", finite_ynew ? 1.0 : 0.0, 1.0, _bool_to_str(finite_ynew)))
    push!(rows, (String(mode_label), "finite_mu_path", finite_mu ? 1.0 : 0.0, 1.0, _bool_to_str(finite_mu)))
    push!(rows, (String(mode_label), "finite_Ldyn", finite_L ? 1.0 : 0.0, 1.0, _bool_to_str(finite_L)))
    push!(rows, (String(mode_label), "finite_realwages", finite_rw ? 1.0 : 0.0, 1.0, _bool_to_str(finite_rw)))

    t_last = max(1, size(path.mu_path, 3) - 1)
    mu_rowsums = [sum(path.mu_path[i, :, t]) for i in axes(path.mu_path, 1), t in 1:t_last]
    mu_row_err = maximum(abs.(mu_rowsums .- 1.0))
    push!(rows, (String(mode_label), "mu_row_sum_max_abs_error", mu_row_err, 1e-8, _bool_to_str(mu_row_err <= 1e-8)))

    nonneg_L = minimum(path.Ldyn)
    push!(rows, (String(mode_label), "ldyn_min", nonneg_L, -1e-10, _bool_to_str(nonneg_L >= -1e-10)))

    DataFrame(rows)
end

function parity_by_time(path::BaselinePath4, matlab_ynew::Matrix{Float64};
                        mode_label::AbstractString = "smoke")
    T = size(path.Ynew, 2)
    max_abs = zeros(T)
    mean_abs = zeros(T)
    @inbounds for t in 1:T
        dt = abs.(path.Ynew[:, t] .- matlab_ynew[:, t])
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

function deterministic_delta(path_a::BaselinePath4, path_b::BaselinePath4)
    maximum(abs.(path_a.Ynew .- path_b.Ynew))
end
