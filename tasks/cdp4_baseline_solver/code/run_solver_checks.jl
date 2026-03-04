using Pkg
Pkg.activate(normpath(joinpath(@__DIR__, "../../_lib/cdp_julia")))

using CDPJulia
using JLD2
using CSV
using DataFrames

function _env_profile()
    raw = lowercase(get(ENV, "PROFILE", "fast"))
    if raw == "reference"
        return :reference
    end
    if raw == "fast"
        return :fast
    end
    error("Unsupported PROFILE=$(raw). Use fast or reference.")
end

base = load_base_state_4sector("../input/Base_year_four_sectors.mat")
max_iter_static = parse(Int, get(ENV, "MAX_ITER_STATIC", "2000"))
config_tag = get(ENV, "CONFIG_TAG", "default")
profile = _env_profile()
params = default_model_params(
    base;
    tol_dynamic = 1e-3,
    max_iter_dynamic = 1000,
    max_iter_static = max_iter_static,
    profile = profile,
)

J, N, R = base.dims.J, base.dims.N, base.dims.R
Ljn_hat = ones(J, N)
kappa_hat = ones(J * N, N)
lambda_hat = ones(J, N)
Snp = zeros(N)

solver_stats = @timed solve_temporary_equilibrium!(
    base,
    Ljn_hat,
    copy(base.VARjn00),
    copy(base.VALjn00),
    copy(base.Din00),
    Snp;
    kappa_hat = kappa_hat,
    lambda_hat = lambda_hat,
    params = params,
)
res = solver_stats.value

@save "../output/static_checks.jld2" res

summary = DataFrame(
    metric = [
        "residual",
        "iterations",
        "all_finite_wf",
        "all_finite_Din",
        "all_finite_X",
        "wf_min",
        "wf_max",
    ],
    value = [
        res.residual,
        Float64(res.iterations),
        all(isfinite, res.wf) ? 1.0 : 0.0,
        all(isfinite, res.Din) ? 1.0 : 0.0,
        all(isfinite, res.X) ? 1.0 : 0.0,
        minimum(res.wf),
        maximum(res.wf),
    ],
)

CSV.write("../output/static_checks_summary.csv", summary)

bench = DataFrame(
    stage = ["baseline_solver"],
    wall_seconds = [solver_stats.time],
    alloc_bytes = [Float64(solver_stats.bytes)],
    gc_seconds = [solver_stats.gctime],
    iterations_static = [Float64(res.iterations)],
    iterations_dynamic = [NaN],
    converged = [res.residual <= params.tol_static ? 1.0 : 0.0],
    final_ymax = [NaN],
    profile = [String(profile)],
    config_tag = [config_tag],
)
CSV.write("../output/benchmark_solver_4sector.csv", bench)
CSV.write("../output/benchmark_solver_4sector_$(String(profile)).csv", bench)
println("Wrote static solver checks to ../output")
