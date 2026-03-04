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
