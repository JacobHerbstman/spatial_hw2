function _as_matrix(x)
    Array{Float64}(x)
end

function _as_vector(x)
    vec(Array{Float64}(x))
end

function _as_int(x)
    if x isa Number
        return Int(round(Float64(x)))
    end
    Int(round(Float64(Array(x)[1])))
end

function _as_float(x)
    if x isa Number
        return Float64(x)
    end
    Float64(Array(x)[1])
end

function _as_array3(x)
    Array{Float64}(x)
end

function load_base_state_4sector(path::AbstractString)
    raw = matread(path)

    dims = ModelDims(
        J = _as_int(raw["J"]),
        N = _as_int(raw["N"]),
        R = _as_int(raw["R"]),
    )

    BaseState4(
        dims,
        _as_matrix(raw["VARjn00"]),
        _as_matrix(raw["VALjn00"]),
        _as_matrix(raw["Din00"]),
        _as_matrix(raw["xbilat00"]),
        _as_vector(raw["Sn00"]),
        _as_matrix(raw["alphas"]),
        _as_vector(raw["io"]),
        _as_vector(raw["T"]),
        _as_matrix(raw["B"]),
        _as_matrix(raw["G"]),
        _as_matrix(raw["gamma"]),
        _as_matrix(raw["mu0"]),
        _as_matrix(raw["L0"]),
        _as_float(raw["tol"]),
        _as_float(raw["vfactor"]),
    )
end

function load_matlab_ynew(path::AbstractString)
    raw = matread(path)
    if haskey(raw, "Ynew")
        return _as_matrix(raw["Ynew"])
    end
    error("Could not find Ynew in $(path)")
end

function load_baseline_anchor_y(path::AbstractString)
    lower = lowercase(path)
    if endswith(lower, ".mat")
        return load_matlab_ynew(path)
    end
    if !endswith(lower, ".jld2")
        error("Unsupported baseline anchor format: $(path). Use .mat or .jld2.")
    end

    raw = load(path)
    if haskey(raw, "path")
        p = raw["path"]
        if hasproperty(p, :Ynew)
            return Array{Float64}(getproperty(p, :Ynew))
        end
    end
    if haskey(raw, "Ynew")
        return _as_matrix(raw["Ynew"])
    end
    error("Could not find Ynew in $(path). Expected key `path.Ynew` or `Ynew`.")
end

function _check_idx(idx::Int, lo::Int, hi::Int, label::AbstractString)
    if idx < lo || idx > hi
        error("Index $(label)=$(idx) out of range [$(lo), $(hi)].")
    end
end

function _as_idx(x)
    if x isa Integer
        return Int(x)
    end
    Int(round(Float64(x)))
end

function _check_shock_dims(base::BaseState4, kappa_hat::Array{Float64,3}, lambda_hat::Array{Float64,3},
                           shock_periods::Int)
    J, N = base.dims.J, base.dims.N
    kappa_expected = (J * N, N, shock_periods)
    lambda_expected = (J, N, shock_periods)
    if size(kappa_hat) != kappa_expected
        error("kappa_hat has size $(size(kappa_hat)); expected $(kappa_expected).")
    end
    if size(lambda_hat) != lambda_expected
        error("lambda_hat has size $(size(lambda_hat)); expected $(lambda_expected).")
    end
end

function load_counterfactual_shocks_4sector(base::BaseState4; time_horizon::Int = 200,
                                            lambda_csv::Union{Nothing, AbstractString} = nothing,
                                            kappa_csv::Union{Nothing, AbstractString} = nothing,
                                            mat_path::Union{Nothing, AbstractString} = nothing)
    J, N = base.dims.J, base.dims.N
    shock_periods = time_horizon - 2
    if shock_periods < 1
        error("time_horizon must be at least 3, got $(time_horizon).")
    end

    lambda_hat = ones(J, N, shock_periods)
    kappa_hat = ones(J * N, N, shock_periods)

    if !isnothing(mat_path)
        if !isfile(mat_path)
            error("Shock MAT file not found: $(mat_path)")
        end
        raw = matread(mat_path)
        if haskey(raw, "lambda_hat")
            lambda_raw = _as_array3(raw["lambda_hat"])
            if size(lambda_raw) != size(lambda_hat)
                error("MAT lambda_hat has size $(size(lambda_raw)); expected $(size(lambda_hat)).")
            end
            lambda_hat .= lambda_raw
        end
        if haskey(raw, "kappa_hat")
            kappa_raw = _as_array3(raw["kappa_hat"])
            if size(kappa_raw) != size(kappa_hat)
                error("MAT kappa_hat has size $(size(kappa_raw)); expected $(size(kappa_hat)).")
            end
            kappa_hat .= kappa_raw
        end
    end

    if !isnothing(lambda_csv)
        if !isfile(lambda_csv)
            error("lambda_csv file not found: $(lambda_csv)")
        end
        for row in CSV.File(lambda_csv)
            t = _as_idx(row.t)
            j = _as_idx(row.j)
            n = _as_idx(row.n)
            _check_idx(t, 1, shock_periods, "t")
            _check_idx(j, 1, J, "j")
            _check_idx(n, 1, N, "n")
            lambda_hat[j, n, t] = Float64(row.value)
        end
    end

    if !isnothing(kappa_csv)
        if !isfile(kappa_csv)
            error("kappa_csv file not found: $(kappa_csv)")
        end
        for row in CSV.File(kappa_csv)
            t = _as_idx(row.t)
            j = _as_idx(row.j)
            n_from = _as_idx(row.n_from)
            n_to = _as_idx(row.n_to)
            _check_idx(t, 1, shock_periods, "t")
            _check_idx(j, 1, J, "j")
            _check_idx(n_from, 1, N, "n_from")
            _check_idx(n_to, 1, N, "n_to")
            idx = n_from + (j - 1) * N
            kappa_hat[idx, n_to, t] = Float64(row.value)
        end
    end

    _check_shock_dims(base, kappa_hat, lambda_hat, shock_periods)
    CounterfactualShocks4(kappa_hat, lambda_hat)
end

function toy_counterfactual_shocks_4sector(base::BaseState4; time_horizon::Int = 200,
                                           sector::Int = 1,
                                           region::Int = 1,
                                           periods::Int = 20,
                                           lambda_multiplier::Float64 = 1.05)
    shocks = load_counterfactual_shocks_4sector(base; time_horizon = time_horizon)
    shock_periods = time_horizon - 2
    _check_idx(sector, 1, base.dims.J, "sector")
    _check_idx(region, 1, base.dims.N, "region")
    tmax = min(periods, shock_periods)
    @inbounds for t in 1:tmax
        shocks.lambda_hat[sector, region, t] = lambda_multiplier
    end
    shocks
end
