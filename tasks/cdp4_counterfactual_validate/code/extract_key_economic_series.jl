using Pkg
Pkg.activate(normpath(joinpath(@__DIR__, "../../_lib/cdp_julia")))

using CDPJulia
using JLD2
using CSV
using DataFrames
using Statistics
using LinearAlgebra

function _env_int(name::String, default::Int)
    parse(Int, get(ENV, name, string(default)))
end

function _slug(x::AbstractString)
    y = replace(lowercase(strip(String(x))), r"[^a-z0-9]+" => "_")
    y = replace(y, r"_+" => "_")
    isempty(y) ? "default" : strip(y, '_')
end

function _safe_pct(shock::Float64, base::Float64)
    abs(base) < 1e-12 ? NaN : 100.0 * (shock / base - 1.0)
end

function _safe_ratio(num::Float64, den::Float64)
    abs(den) < 1e-12 ? NaN : num / den
end

function _nanmean(v::AbstractVector{Float64})
    keep = filter(isfinite, v)
    isempty(keep) ? NaN : mean(keep)
end

function _load_shocks(base::BaseState4, time_horizon::Int)
    mode = lowercase(get(ENV, "SHOCK_INPUT_MODE", "toy"))
    lambda_csv = get(ENV, "LAMBDA_CSV", "")
    kappa_csv = get(ENV, "KAPPA_CSV", "")
    mat_path = get(ENV, "SHOCK_MAT", "")

    if mode == "identity"
        return load_counterfactual_shocks_4sector(base; time_horizon = time_horizon)
    end
    if mode == "csv"
        lambda_path = isempty(lambda_csv) ? nothing : lambda_csv
        kappa_path = isempty(kappa_csv) ? nothing : kappa_csv
        return load_counterfactual_shocks_4sector(
            base;
            time_horizon = time_horizon,
            lambda_csv = lambda_path,
            kappa_csv = kappa_path,
        )
    end
    if mode == "mat"
        if isempty(mat_path)
            error("SHOCK_MAT is required when SHOCK_INPUT_MODE=mat")
        end
        return load_counterfactual_shocks_4sector(base; time_horizon = time_horizon, mat_path = mat_path)
    end
    if mode == "toy"
        toy_sector = _env_int("TOY_SECTOR", 1)
        toy_region = _env_int("TOY_REGION", 1)
        toy_periods = _env_int("TOY_PERIODS", 20)
        toy_lambda = parse(Float64, get(ENV, "TOY_LAMBDA_MULTIPLIER", "1.05"))
        return toy_counterfactual_shocks_4sector(
            base;
            time_horizon = time_horizon,
            sector = toy_sector,
            region = toy_region,
            periods = toy_periods,
            lambda_multiplier = toy_lambda,
        )
    end

    error("Unsupported SHOCK_INPUT_MODE=$(mode). Use identity, csv, mat, or toy.")
end

function _sum_row(M::AbstractMatrix{Float64}, row::Int)
    s = 0.0
    @inbounds for j in axes(M, 2)
        s += M[row, j]
    end
    s
end

function _sum_col(M::AbstractMatrix{Float64}, col::Int)
    s = 0.0
    @inbounds for i in axes(M, 1)
        s += M[i, col]
    end
    s
end

function _sum_rows_col(M::AbstractMatrix{Float64}, rows::Vector{Int}, col::Int)
    s = 0.0
    @inbounds for r in rows
        s += M[r, col]
    end
    s
end

function _sum_rows_all_cols(M::AbstractMatrix{Float64}, rows::UnitRange{Int})
    s = 0.0
    @inbounds for r in rows, c in axes(M, 2)
        s += M[r, c]
    end
    s
end

function _replay_trade_series(base::BaseState4, path, shocks::CounterfactualShocks4;
                              shock_sector::Int, shock_region::Int,
                              max_iter_static::Int = 2000)
    J, N, T = size(path.realwages)
    R = size(path.Ldyn, 2)

    if shock_sector < 1 || shock_sector > J
        error("SHOCK_SECTOR=$(shock_sector) out of range 1:$(J)")
    end
    if shock_region < 1 || shock_region > R
        error("SHOCK_REGION=$(shock_region) out of range 1:$(R)")
    end

    idx_trade = shock_region + (shock_sector - 1) * N
    region_rows = [shock_region + (j - 1) * N for j in 1:J]
    sector_rows = ((shock_sector - 1) * N + 1):(shock_sector * N)

    exports_cell = fill(NaN, T)
    exports_external_cell = fill(NaN, T)
    imports_region = fill(NaN, T)
    imports_external_region = fill(NaN, T)
    domestic_import_share_region = fill(NaN, T)
    cell_import_share_region = fill(NaN, T)
    sector_exports_total = fill(NaN, T)
    trade_balance_region = fill(NaN, T)

    params_static = ModelParams(
        tol_static = 1e-7,
        max_iter_static = max_iter_static,
        vfactor = base.vfactor,
        use_threads = false,
        threads_dynamic = false,
        threads_static = false,
        profile = :reference,
        warm_start_static = false,
        record_trace = false,
    )

    ljn_hat = ones(J, N)
    snp = zeros(N)
    VARjn0 = copy(base.VARjn00)
    VALjn0 = copy(base.VALjn00)
    Din0 = copy(base.Din00)
    kappa_t = ones(J * N, N)
    lambda_t = ones(J, N)
    om_init = ones(J, N)
    ws = TempEqWorkspace(base)

    for t in 1:(T - 2)
        fill!(ljn_hat, 1.0)
        @inbounds for r in 1:R, j in 1:J
            lt = path.Ldyn[j, r, t]
            ltp1 = path.Ldyn[j, r, t + 1]
            ljn_hat[j, r] = abs(lt) < 1e-12 ? 1.0 : ltp1 / lt
        end

        kappa_t .= view(shocks.kappa_hat, :, :, t)
        lambda_t .= view(shocks.lambda_hat, :, :, t)
        temp = solve_temporary_equilibrium_inplace!(
            base,
            ljn_hat,
            VARjn0,
            VALjn0,
            Din0,
            snp;
            kappa_hat = kappa_t,
            lambda_hat = lambda_t,
            params = params_static,
            om_init = om_init,
            workspace = ws,
            reset_price_guess = true,
        )

        VARjn0 .= temp.VARjn
        VALjn0 .= temp.VALjn
        Din0 .= temp.Din

        tt = t + 1

        ex_cell = _sum_row(temp.xbilat, idx_trade)
        ex_domestic = temp.xbilat[idx_trade, shock_region]
        im_region = _sum_col(temp.xbilat, shock_region)
        im_domestic = _sum_rows_col(temp.xbilat, region_rows, shock_region)

        exports_cell[tt] = ex_cell
        exports_external_cell[tt] = ex_cell - ex_domestic
        imports_region[tt] = im_region
        imports_external_region[tt] = im_region - im_domestic
        domestic_import_share_region[tt] = _safe_ratio(im_domestic, im_region)
        cell_import_share_region[tt] = temp.Din[idx_trade, shock_region]
        sector_exports_total[tt] = _sum_rows_all_cols(temp.xbilat, sector_rows)
        trade_balance_region[tt] = temp.Sn[shock_region]
    end

    DataFrame(
        t = collect(1:T),
        exports_cell = exports_cell,
        exports_external_cell = exports_external_cell,
        imports_region = imports_region,
        imports_external_region = imports_external_region,
        domestic_import_share_region = domestic_import_share_region,
        cell_import_share_region = cell_import_share_region,
        sector_exports_total = sector_exports_total,
        trade_balance_region = trade_balance_region,
    )
end

identity_file = get(ENV, "IDENTITY_OUTPUT_FILE", "../input/counterfactual_4sector_path_fast_identity.jld2")
shock_file = get(ENV, "SHOCK_OUTPUT_FILE", "../input/counterfactual_4sector_path_fast_toy_smoke.jld2")
profile_tag = get(ENV, "PROFILE_TAG", "fast")
shock_name = get(ENV, "SHOCK_NAME", "toy_smoke")
output_dir = get(ENV, "OUTPUT_DIR", "../output")
shock_sector = _env_int("SHOCK_SECTOR", 1)
shock_region = _env_int("SHOCK_REGION", 1)

id_raw = load(identity_file)
sh_raw = load(shock_file)
if !(haskey(id_raw, "path") && haskey(sh_raw, "path"))
    error("Expected key `path` in both identity and shock JLD2 files.")
end

id = id_raw["path"]
sh = sh_raw["path"]

J, N, T = size(sh.realwages)
R = size(sh.Ldyn, 2)
if shock_sector < 1 || shock_sector > J
    error("SHOCK_SECTOR=$(shock_sector) out of range 1:$(J)")
end
if shock_region < 1 || shock_region > R
    error("SHOCK_REGION=$(shock_region) out of range 1:$(R)")
end
if size(id.realwages) != size(sh.realwages) || size(id.Ldyn) != size(sh.Ldyn)
    error("Identity and shocked paths have mismatched dimensions.")
end

idx = shock_sector + (shock_region - 1) * J
RJ = J * R

t = collect(1:T)
rw_id = zeros(Float64, T)
rw_sh = zeros(Float64, T)
rw_pct = zeros(Float64, T)

emp_cell_id = zeros(Float64, T)
emp_cell_sh = zeros(Float64, T)
emp_cell_pct = zeros(Float64, T)

emp_sector_id = zeros(Float64, T)
emp_sector_sh = zeros(Float64, T)
emp_sector_pct = zeros(Float64, T)

y_id = zeros(Float64, T)
y_sh = zeros(Float64, T)
y_pct = zeros(Float64, T)

inflow_id = fill(NaN, T)
inflow_sh = fill(NaN, T)
inflow_pct = fill(NaN, T)

stay_id = fill(NaN, T)
stay_sh = fill(NaN, T)
stay_pp = fill(NaN, T)

outflow_id = fill(NaN, T)
outflow_sh = fill(NaN, T)
outflow_pct = fill(NaN, T)

netmig_id = fill(NaN, T)
netmig_sh = fill(NaN, T)
netmig_diff = fill(NaN, T)

for tt in 1:T
    rw_id[tt] = id.realwages[shock_sector, shock_region, tt]
    rw_sh[tt] = sh.realwages[shock_sector, shock_region, tt]
    rw_pct[tt] = _safe_pct(rw_sh[tt], rw_id[tt])

    emp_cell_id[tt] = id.Ldyn[shock_sector, shock_region, tt]
    emp_cell_sh[tt] = sh.Ldyn[shock_sector, shock_region, tt]
    emp_cell_pct[tt] = _safe_pct(emp_cell_sh[tt], emp_cell_id[tt])

    emp_sector_id[tt] = sum(id.Ldyn[shock_sector, 1:R, tt])
    emp_sector_sh[tt] = sum(sh.Ldyn[shock_sector, 1:R, tt])
    emp_sector_pct[tt] = _safe_pct(emp_sector_sh[tt], emp_sector_id[tt])

    y_id[tt] = id.Ynew[idx, tt]
    y_sh[tt] = sh.Ynew[idx, tt]
    y_pct[tt] = _safe_pct(y_sh[tt], y_id[tt])

    if tt <= (T - 1)
        l_id_t = reshape(id.Ldyn[:, :, tt], RJ)
        l_sh_t = reshape(sh.Ldyn[:, :, tt], RJ)
        inflow_id[tt] = dot(l_id_t, id.mu_path[:, idx, tt])
        inflow_sh[tt] = dot(l_sh_t, sh.mu_path[:, idx, tt])
        inflow_pct[tt] = _safe_pct(inflow_sh[tt], inflow_id[tt])

        stay_id[tt] = id.mu_path[idx, idx, tt]
        stay_sh[tt] = sh.mu_path[idx, idx, tt]
        stay_pp[tt] = 100.0 * (stay_sh[tt] - stay_id[tt])

        outflow_id[tt] = id.Ldyn[shock_sector, shock_region, tt] * (1.0 - stay_id[tt])
        outflow_sh[tt] = sh.Ldyn[shock_sector, shock_region, tt] * (1.0 - stay_sh[tt])
        outflow_pct[tt] = _safe_pct(outflow_sh[tt], outflow_id[tt])

        netmig_id[tt] = inflow_id[tt] - outflow_id[tt]
        netmig_sh[tt] = inflow_sh[tt] - outflow_sh[tt]
        netmig_diff[tt] = netmig_sh[tt] - netmig_id[tt]
    end
end

base_state_mat = get(ENV, "BASE_STATE_MAT", "../input/Base_year_four_sectors.mat")
max_iter_static = _env_int("MAX_ITER_STATIC", 2000)
base = load_base_state_4sector(base_state_mat)
id_shocks = load_counterfactual_shocks_4sector(base; time_horizon = T)
sh_shocks = _load_shocks(base, T)

id_trade = _replay_trade_series(
    base,
    id,
    id_shocks;
    shock_sector = shock_sector,
    shock_region = shock_region,
    max_iter_static = max_iter_static,
)
sh_trade = _replay_trade_series(
    base,
    sh,
    sh_shocks;
    shock_sector = shock_sector,
    shock_region = shock_region,
    max_iter_static = max_iter_static,
)

exports_cell_id = Vector{Float64}(id_trade.exports_cell)
exports_cell_sh = Vector{Float64}(sh_trade.exports_cell)
exports_cell_pct = [_safe_pct(exports_cell_sh[tt], exports_cell_id[tt]) for tt in 1:T]

exports_external_cell_id = Vector{Float64}(id_trade.exports_external_cell)
exports_external_cell_sh = Vector{Float64}(sh_trade.exports_external_cell)
exports_external_cell_pct = [_safe_pct(exports_external_cell_sh[tt], exports_external_cell_id[tt]) for tt in 1:T]

imports_region_id = Vector{Float64}(id_trade.imports_region)
imports_region_sh = Vector{Float64}(sh_trade.imports_region)
imports_region_pct = [_safe_pct(imports_region_sh[tt], imports_region_id[tt]) for tt in 1:T]

imports_external_region_id = Vector{Float64}(id_trade.imports_external_region)
imports_external_region_sh = Vector{Float64}(sh_trade.imports_external_region)
imports_external_region_pct = [_safe_pct(imports_external_region_sh[tt], imports_external_region_id[tt]) for tt in 1:T]

domestic_import_share_region_id = Vector{Float64}(id_trade.domestic_import_share_region)
domestic_import_share_region_sh = Vector{Float64}(sh_trade.domestic_import_share_region)
domestic_import_share_region_pp = [100.0 * (domestic_import_share_region_sh[tt] - domestic_import_share_region_id[tt]) for tt in 1:T]

cell_import_share_region_id = Vector{Float64}(id_trade.cell_import_share_region)
cell_import_share_region_sh = Vector{Float64}(sh_trade.cell_import_share_region)
cell_import_share_region_pp = [100.0 * (cell_import_share_region_sh[tt] - cell_import_share_region_id[tt]) for tt in 1:T]

sector_exports_total_id = Vector{Float64}(id_trade.sector_exports_total)
sector_exports_total_sh = Vector{Float64}(sh_trade.sector_exports_total)
sector_exports_total_pct = [_safe_pct(sector_exports_total_sh[tt], sector_exports_total_id[tt]) for tt in 1:T]

trade_balance_region_id = Vector{Float64}(id_trade.trade_balance_region)
trade_balance_region_sh = Vector{Float64}(sh_trade.trade_balance_region)
trade_balance_region_diff = [trade_balance_region_sh[tt] - trade_balance_region_id[tt] for tt in 1:T]

ts = DataFrame(
    t = t,
    rw_cell_id = rw_id,
    rw_cell_shock = rw_sh,
    rw_cell_pct = rw_pct,
    emp_cell_id = emp_cell_id,
    emp_cell_shock = emp_cell_sh,
    emp_cell_pct = emp_cell_pct,
    emp_sector_id = emp_sector_id,
    emp_sector_shock = emp_sector_sh,
    emp_sector_pct = emp_sector_pct,
    y_cell_id = y_id,
    y_cell_shock = y_sh,
    y_cell_pct = y_pct,
    inflow_cell_id = inflow_id,
    inflow_cell_shock = inflow_sh,
    inflow_cell_pct = inflow_pct,
    stay_prob_cell_id = stay_id,
    stay_prob_cell_shock = stay_sh,
    stay_prob_cell_pp = stay_pp,
    outflow_cell_id = outflow_id,
    outflow_cell_shock = outflow_sh,
    outflow_cell_pct = outflow_pct,
    net_migration_cell_id = netmig_id,
    net_migration_cell_shock = netmig_sh,
    net_migration_cell_diff = netmig_diff,
    exports_cell_id = exports_cell_id,
    exports_cell_shock = exports_cell_sh,
    exports_cell_pct = exports_cell_pct,
    exports_external_cell_id = exports_external_cell_id,
    exports_external_cell_shock = exports_external_cell_sh,
    exports_external_cell_pct = exports_external_cell_pct,
    imports_region_id = imports_region_id,
    imports_region_shock = imports_region_sh,
    imports_region_pct = imports_region_pct,
    imports_external_region_id = imports_external_region_id,
    imports_external_region_shock = imports_external_region_sh,
    imports_external_region_pct = imports_external_region_pct,
    domestic_import_share_region_id = domestic_import_share_region_id,
    domestic_import_share_region_shock = domestic_import_share_region_sh,
    domestic_import_share_region_pp = domestic_import_share_region_pp,
    cell_import_share_region_id = cell_import_share_region_id,
    cell_import_share_region_shock = cell_import_share_region_sh,
    cell_import_share_region_pp = cell_import_share_region_pp,
    sector_exports_total_id = sector_exports_total_id,
    sector_exports_total_shock = sector_exports_total_sh,
    sector_exports_total_pct = sector_exports_total_pct,
    trade_balance_region_id = trade_balance_region_id,
    trade_balance_region_shock = trade_balance_region_sh,
    trade_balance_region_diff = trade_balance_region_diff,
)

selected_t = unique(sort([2, 5, 10, 20, min(50, T), min(100, T), max(2, T - 1), T]))
selected_t = filter(x -> 1 <= x <= T, selected_t)

selected_cols = [
    :t,
    :rw_cell_pct,
    :emp_cell_pct,
    :inflow_cell_pct,
    :outflow_cell_pct,
    :stay_prob_cell_pp,
    :net_migration_cell_diff,
    :y_cell_pct,
    :exports_cell_pct,
    :exports_external_cell_pct,
    :imports_region_pct,
    :imports_external_region_pct,
    :cell_import_share_region_pp,
    :domestic_import_share_region_pp,
    :sector_exports_total_pct,
    :trade_balance_region_diff,
]
selected = ts[in.(ts.t, Ref(selected_t)), selected_cols]

window_starts = [2, 21, 101]
window_ends = [min(20, T), min(100, T), T]
window_labels = ["t2_t20", "t21_t100", "t101_tT"]

window_vars = [
    "rw_cell_pct",
    "emp_cell_pct",
    "inflow_cell_pct",
    "outflow_cell_pct",
    "stay_prob_cell_pp",
    "y_cell_pct",
    "exports_cell_pct",
    "exports_external_cell_pct",
    "imports_region_pct",
    "imports_external_region_pct",
    "cell_import_share_region_pp",
    "domestic_import_share_region_pp",
    "sector_exports_total_pct",
    "trade_balance_region_diff",
]
wm = DataFrame(
    variable = window_vars,
    t2_t20 = fill(NaN, length(window_vars)),
    t21_t100 = fill(NaN, length(window_vars)),
    t101_tT = fill(NaN, length(window_vars)),
)

for (k, (a, b)) in enumerate(zip(window_starts, window_ends))
    if a <= b
        idxs = findall(tt -> a <= tt <= b, ts.t)
        vals = Vector{Float64}(undef, length(window_vars))
        for (ii, nm) in enumerate(window_vars)
            vals[ii] = _nanmean(Vector{Float64}(ts[idxs, Symbol(nm)]))
        end
        wm[!, Symbol(window_labels[k])] = vals
    end
end

mkpath(output_dir)
shock_slug = _slug(shock_name)

CSV.write(joinpath(output_dir, "key_econ_timeseries_$(profile_tag)_$(shock_slug).csv"), ts)
CSV.write(joinpath(output_dir, "key_econ_selected_t_$(profile_tag)_$(shock_slug).csv"), selected)
CSV.write(joinpath(output_dir, "key_econ_window_means_$(profile_tag)_$(shock_slug).csv"), wm)

println("Wrote key economic series CSVs to $(output_dir)")
