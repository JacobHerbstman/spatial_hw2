Base.@kwdef struct ModelDims
    J::Int
    N::Int
    R::Int
end

function _check_profile(profile::Symbol)
    if !(profile in (:reference, :fast))
        error("Unsupported profile=$(profile). Use :reference or :fast.")
    end
    profile
end

Base.@kwdef struct ModelParams
    beta::Float64 = 0.99
    nu::Float64 = 5.3436
    tol_static::Float64 = 1e-7
    tol_dynamic::Float64 = 1e-5
    max_iter_static::Int = 2000
    max_iter_dynamic::Int = 1000
    vfactor::Float64 = -0.05
    use_threads::Bool = false
    threads_dynamic::Bool = false
    threads_static::Bool = false
    profile::Symbol = :fast
    warm_start_static::Bool = true
    use_anderson::Bool = false
    hvect_relax::Float64 = 0.5
    record_trace::Bool = false
end

struct BaseState4
    dims::ModelDims
    VARjn00::Matrix{Float64}
    VALjn00::Matrix{Float64}
    Din00::Matrix{Float64}
    xbilat00::Matrix{Float64}
    Sn00::Vector{Float64}
    alphas::Matrix{Float64}
    io::Vector{Float64}
    theta::Vector{Float64}
    B::Matrix{Float64}
    G::Matrix{Float64}
    gamma::Matrix{Float64}
    mu0::Matrix{Float64}
    L0::Matrix{Float64}
    tol::Float64
    vfactor::Float64
end

struct TempEqResult
    om::Matrix{Float64}
    wf::Matrix{Float64}
    VARjn::Matrix{Float64}
    VALjn::Matrix{Float64}
    Pidx::Vector{Float64}
    rf::Matrix{Float64}
    phat::Matrix{Float64}
    Din::Matrix{Float64}
    X::Matrix{Float64}
    Sn::Vector{Float64}
    xbilat::Matrix{Float64}
    residual::Float64
    iterations::Int
end

struct TempEqResultView
    om::Matrix{Float64}
    wf::Matrix{Float64}
    VARjn::Matrix{Float64}
    VALjn::Matrix{Float64}
    Pidx::Vector{Float64}
    rf::Matrix{Float64}
    phat::Matrix{Float64}
    Din::Matrix{Float64}
    X::Matrix{Float64}
    Sn::Vector{Float64}
    xbilat::Matrix{Float64}
    residual::Float64
    iterations::Int
end

struct BaselinePath4
    Ynew::Matrix{Float64}
    mu_path::Array{Float64,3}
    Ldyn::Array{Float64,3}
    realwages::Array{Float64,3}
    static_iterations::Vector{Int}
    iterations::Int
    converged::Bool
    final_ymax::Float64
    static_residuals::Vector{Float64}
    outer_ymax::Vector{Float64}
    outer_mean_static_iterations::Vector{Float64}
    outer_max_static_iterations::Vector{Int}
    outer_max_static_residual::Vector{Float64}
    profile::Symbol
end

struct CounterfactualShocks4
    kappa_hat::Array{Float64,3}
    lambda_hat::Array{Float64,3}
end

struct CounterfactualPath4
    Ynew::Matrix{Float64}
    mu_path::Array{Float64,3}
    Ldyn::Array{Float64,3}
    realwages::Array{Float64,3}
    static_iterations::Vector{Int}
    iterations::Int
    converged::Bool
    final_ymax::Float64
    static_residuals::Vector{Float64}
    outer_ymax::Vector{Float64}
    outer_mean_static_iterations::Vector{Float64}
    outer_max_static_iterations::Vector{Int}
    outer_max_static_residual::Vector{Float64}
    profile::Symbol
    mu00_baseline::Matrix{Float64}
    meta_shock_name::String
end

mutable struct AndersonAccelerator
    m::Int
    beta::Float64
    ridge::Float64
    n::Int
    k::Int
    x_hist::Matrix{Float64}
    f_hist::Matrix{Float64}
    f_curr::Vector{Float64}
    x_next::Vector{Float64}
    active_cols::Vector{Int}
    kkt::Matrix{Float64}
    rhs::Vector{Float64}
    sol::Vector{Float64}
end

function AndersonAccelerator(n::Int; m::Int = 5, beta::Float64 = 0.5, ridge::Float64 = 1e-10)
    if n <= 0
        error("AndersonAccelerator requires n > 0, got n=$(n)")
    end
    if m < 1
        error("AndersonAccelerator requires m >= 1, got m=$(m)")
    end
    AndersonAccelerator(
        m,
        beta,
        ridge,
        n,
        0,
        zeros(n, m),
        zeros(n, m),
        zeros(n),
        zeros(n),
        zeros(Int, m),
        zeros(m + 1, m + 1),
        zeros(m + 1),
        zeros(m + 1),
    )
end

struct TempEqWorkspace
    LT::Vector{Float64}
    LT_mat::Matrix{Float64}
    inv_LT_mat::Matrix{Float64}
    theta_mat::Matrix{Float64}
    pf0::Matrix{Float64}
    phat::Matrix{Float64}
    c::Matrix{Float64}
    lc::Matrix{Float64}
    lom::Matrix{Float64}
    lp::Matrix{Float64}
    Din_k::Matrix{Float64}
    tmp_row::Vector{Float64}
    DD::Matrix{Float64}
    Dinp::Matrix{Float64}
    cp::Matrix{Float64}
    phatp::Matrix{Float64}
    inner_term::Matrix{Float64}
    NBP::Matrix{Float64}
    NNBP::Matrix{Float64}
    GG::Matrix{Float64}
    GP::Matrix{Float64}
    OM::Matrix{Float64}
    aux::Vector{Float64}
    rhs::Vector{Float64}
    Xvec::Vector{Float64}
    VARjnp::Matrix{Float64}
    VARp::Vector{Float64}
    Bnp::Vector{Float64}
    DP::Matrix{Float64}
    Exjnp::Matrix{Float64}
    aux4::Matrix{Float64}
    VAR::Vector{Float64}
    VAL::Vector{Float64}
    aux5::Vector{Float64}
    ZW::Matrix{Float64}
    om1::Matrix{Float64}
    om::Matrix{Float64}
    wf::Matrix{Float64}
    rf::Matrix{Float64}
    VALjnp::Matrix{Float64}
    Pidx::Vector{Float64}
    kappa_is_one::Bool
    lambda_is_one::Bool
    PQ_vec::Vector{Float64}
    xbilat::Matrix{Float64}
end

struct BaselineWorkspace4
    hpow::Vector{Float64}
    col_weights::Vector{Float64}
    row_sums::Vector{Float64}
    lvec::Vector{Float64}
    lnext::Vector{Float64}
    ytmp::Vector{Float64}
    ynew::Matrix{Float64}
    checky::Vector{Float64}
    ljn_hat::Matrix{Float64}
    kappa_hat::Matrix{Float64}
    lambda_hat::Matrix{Float64}
    snp::Vector{Float64}
    om_prev::Matrix{Float64}
    om_guesses::Array{Float64,3}
    om_init::Matrix{Float64}
    outer_ymax::Vector{Float64}
    outer_mean_static_iterations::Vector{Float64}
    outer_max_static_iterations::Vector{Int}
    outer_max_static_residual::Vector{Float64}
    temp_ws::TempEqWorkspace
end

struct CounterfactualWorkspace4
    hpow::Vector{Float64}
    hpow_noshock_t1::Vector{Float64}
    col_weights::Vector{Float64}
    ratio_t1::Vector{Float64}
    row_sums::Vector{Float64}
    lvec::Vector{Float64}
    lnext::Vector{Float64}
    ytmp::Vector{Float64}
    ynew::Matrix{Float64}
    checky::Vector{Float64}
    ljn_hat::Matrix{Float64}
    kappa_hat::Matrix{Float64}
    lambda_hat::Matrix{Float64}
    snp::Vector{Float64}
    om_prev::Matrix{Float64}
    om_guesses::Array{Float64,3}
    om_init::Matrix{Float64}
    mu00::Matrix{Float64}
    mu0_tilde::Matrix{Float64}
    mu_path::Array{Float64,3}
    special_num1::Matrix{Float64}
    VARjn0::Matrix{Float64}
    VALjn0::Matrix{Float64}
    Din0::Matrix{Float64}
    Sn0::Vector{Float64}
    outer_ymax::Vector{Float64}
    outer_mean_static_iterations::Vector{Float64}
    outer_max_static_iterations::Vector{Int}
    outer_max_static_residual::Vector{Float64}
    temp_ws::TempEqWorkspace
end

function TempEqWorkspace(base::BaseState4)
    J, N = base.dims.J, base.dims.N
    JN = J * N
    LT = zeros(Float64, JN)
    for j in 1:J
        LT[(1 + (j - 1) * N):(j * N)] .= base.theta[j]
    end
    LT_mat = repeat(reshape(LT, :, 1), 1, N)
    inv_LT_mat = -1.0 ./ LT_mat
    theta_mat = repeat(reshape(base.theta, J, 1), 1, N)
    GG = kron(ones(1, N), base.G)
    TempEqWorkspace(
        LT,
        LT_mat,
        inv_LT_mat,
        theta_mat,
        ones(J, N),
        ones(J, N),
        ones(J, N),
        zeros(J, N),
        zeros(J, N),
        zeros(J, N),
        zeros(JN, N),
        zeros(N),
        zeros(JN, N),
        zeros(JN, N),
        zeros(J, N),
        zeros(J, N),
        zeros(J, N),
        zeros(N, JN),
        zeros(JN, JN),
        GG,
        zeros(JN, JN),
        Matrix{Float64}(I, JN, JN),
        zeros(N),
        zeros(JN),
        zeros(JN),
        zeros(J, N),
        zeros(N),
        zeros(N),
        zeros(JN, N),
        zeros(J, N),
        zeros(J, N),
        zeros(N),
        zeros(N),
        zeros(N),
        zeros(J, N),
        zeros(J, N),
        ones(J, N),
        ones(J, N),
        ones(J, N),
        zeros(J, N),
        ones(N),
        false,
        false,
        zeros(JN),
        zeros(JN, N),
    )
end

function BaselineWorkspace4(base::BaseState4; time_horizon::Int = 200)
    J, N, R = base.dims.J, base.dims.N, base.dims.R
    RJ = R * J
    BaselineWorkspace4(
        zeros(RJ),
        zeros(RJ),
        zeros(RJ),
        zeros(RJ),
        zeros(RJ),
        zeros(RJ),
        zeros(RJ, time_horizon),
        zeros(time_horizon),
        ones(J, N),
        ones(J * N, N),
        ones(J, N),
        zeros(N),
        ones(J, N),
        ones(J, N, time_horizon),
        ones(J, N),
        Float64[],
        Float64[],
        Int[],
        Float64[],
        TempEqWorkspace(base),
    )
end

function CounterfactualWorkspace4(base::BaseState4; time_horizon::Int = 200)
    J, N, R = base.dims.J, base.dims.N, base.dims.R
    RJ = R * J
    CounterfactualWorkspace4(
        zeros(RJ),
        zeros(RJ),
        zeros(RJ),
        zeros(RJ),
        zeros(RJ),
        zeros(RJ),
        zeros(RJ),
        zeros(RJ),
        zeros(RJ, time_horizon),
        zeros(time_horizon),
        ones(J, N),
        ones(J * N, N),
        ones(J, N),
        zeros(N),
        ones(J, N),
        ones(J, N, time_horizon),
        ones(J, N),
        zeros(RJ, RJ),
        zeros(RJ, RJ),
        zeros(RJ, RJ, time_horizon),
        zeros(RJ, RJ),
        copy(base.VARjn00),
        copy(base.VALjn00),
        copy(base.Din00),
        copy(base.Sn00),
        Float64[],
        Float64[],
        Int[],
        Float64[],
        TempEqWorkspace(base),
    )
end

function default_model_params(base::BaseState4; tol_static::Float64 = 1e-7,
                              tol_dynamic::Float64 = 1e-5,
                              max_iter_dynamic::Int = 1000,
                              max_iter_static::Int = 2000,
                              vfactor::Float64 = -0.05,
                              use_threads::Bool = false,
                              threads_dynamic::Bool = false,
                              threads_static::Bool = false,
                              profile::Symbol = :fast,
                              warm_start_static::Union{Nothing, Bool} = nothing,
                              use_anderson::Union{Nothing, Bool} = nothing,
                              hvect_relax::Float64 = 0.5,
                              record_trace::Bool = false)
    profile_checked = _check_profile(profile)
    warm_start = isnothing(warm_start_static) ? (profile_checked == :fast) : warm_start_static
    anderson = isnothing(use_anderson) ? false : use_anderson
    ModelParams(
        beta = 0.99,
        nu = 5.3436,
        tol_static = tol_static,
        tol_dynamic = tol_dynamic,
        max_iter_static = max_iter_static,
        max_iter_dynamic = max_iter_dynamic,
        vfactor = vfactor,
        use_threads = use_threads,
        threads_dynamic = threads_dynamic,
        threads_static = threads_static,
        profile = profile_checked,
        warm_start_static = warm_start,
        use_anderson = anderson,
        hvect_relax = hvect_relax,
        record_trace = record_trace,
    )
end
