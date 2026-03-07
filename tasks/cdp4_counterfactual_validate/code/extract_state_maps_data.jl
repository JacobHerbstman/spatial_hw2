using Pkg
Pkg.activate(normpath(joinpath(@__DIR__, "../../_lib/cdp_julia")))

using CDPJulia
using JLD2
using CSV
using DataFrames
using Statistics

const STATE_ABBRS = [
    "AL", "AK", "AZ", "AR", "CA", "CO", "CT", "DE", "FL", "GA",
    "HI", "ID", "IL", "IN", "IA", "KS", "KY", "LA", "ME", "MD",
    "MA", "MI", "MN", "MS", "MO", "MT", "NE", "NV", "NH", "NJ",
    "NM", "NY", "NC", "ND", "OH", "OK", "OR", "PA", "RI", "SC",
    "SD", "TN", "TX", "UT", "VT", "VA", "WA", "WV", "WI", "WY",
]

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

function _safe_share_pp(num_sh::Float64, den_sh::Float64, num_id::Float64, den_id::Float64)
    sh = _safe_ratio(num_sh, den_sh)
    id = _safe_ratio(num_id, den_id)
    (!isfinite(sh) || !isfinite(id)) ? NaN : 100.0 * (sh - id)
end

identity_file = get(ENV, "IDENTITY_OUTPUT_FILE", "../input/counterfactual_4sector_path_fast_identity.jld2")
shock_file = get(ENV, "SHOCK_OUTPUT_FILE", "../input/counterfactual_4sector_path_fast_toy_smoke.jld2")
profile_tag = get(ENV, "PROFILE_TAG", "fast")
shock_name = get(ENV, "SHOCK_NAME", "toy_smoke")
output_dir = get(ENV, "OUTPUT_DIR", "../output")

id_raw = load(identity_file)
sh_raw = load(shock_file)
if !(haskey(id_raw, "path") && haskey(sh_raw, "path"))
    error("Expected key `path` in both identity and shock JLD2 files.")
end

id = id_raw["path"]
sh = sh_raw["path"]

if !(hasproperty(id, :realwages) && hasproperty(id, :Ldyn) && hasproperty(id, :Ynew))
    error("Identity path does not contain expected fields realwages, Ldyn, and Ynew.")
end
if !(hasproperty(sh, :realwages) && hasproperty(sh, :Ldyn) && hasproperty(sh, :Ynew))
    error("Shock path does not contain expected fields realwages, Ldyn, and Ynew.")
end

J = size(sh.Ldyn, 1)
R = size(sh.Ldyn, 2)
T = size(sh.Ynew, 2)

if J != 4
    error("Expected 4 sectors, found J=$(J).")
end
if R != 50
    error("Expected 50 states in Ldyn, found R=$(R).")
end
if length(STATE_ABBRS) != R
    error("STATE_ABBRS length mismatch: expected $(R), found $(length(STATE_ABBRS)).")
end
if size(id.realwages) != size(sh.realwages) || size(id.Ldyn) != size(sh.Ldyn) || size(id.Ynew) != size(sh.Ynew)
    error("Identity and shock paths have mismatched dimensions.")
end
if size(sh.realwages, 2) < R
    error("Expected at least $(R) regions in realwages, found $(size(sh.realwages, 2)).")
end

times = [t for t in (2, 5, 10, 20, 50, 100, 199) if 1 <= t <= T - 1]
if isempty(times)
    error("No valid selected time periods for T=$(T).")
end

rw_id = view(id.realwages, :, 1:R, :)
rw_sh = view(sh.realwages, :, 1:R, :)
L_id = id.Ldyn
L_sh = sh.Ldyn

rows = NamedTuple[]
sizehint!(rows, length(times) * R)

for t in times
    for r in 1:R
        l_id_total = 0.0
        l_sh_total = 0.0
        rw_num = 0.0
        rw_den = 0.0
        @inbounds for j in 1:J
            l_id_total += L_id[j, r, t]
            l_sh_total += L_sh[j, r, t]
            rw_num += L_id[j, r, t] * rw_sh[j, r, t]
            rw_den += L_id[j, r, t] * rw_id[j, r, t]
        end

        net_mig_id = 0.0
        net_mig_sh = 0.0
        @inbounds for j in 1:J
            net_mig_id += L_id[j, r, t + 1] - L_id[j, r, t]
            net_mig_sh += L_sh[j, r, t + 1] - L_sh[j, r, t]
        end

        push!(rows, (
            t = t,
            state_idx = r,
            state_abbr = STATE_ABBRS[r],
            rw_pct_mfg = _safe_pct(rw_sh[1, r, t], rw_id[1, r, t]),
            rw_pct_con = _safe_pct(rw_sh[2, r, t], rw_id[2, r, t]),
            rw_pct_whr = _safe_pct(rw_sh[3, r, t], rw_id[3, r, t]),
            rw_pct_svc = _safe_pct(rw_sh[4, r, t], rw_id[4, r, t]),
            rw_pct_agg = _safe_pct(rw_num, rw_den),
            emp_pct_mfg = _safe_pct(L_sh[1, r, t], L_id[1, r, t]),
            emp_pct_con = _safe_pct(L_sh[2, r, t], L_id[2, r, t]),
            emp_pct_whr = _safe_pct(L_sh[3, r, t], L_id[3, r, t]),
            emp_pct_svc = _safe_pct(L_sh[4, r, t], L_id[4, r, t]),
            emp_pct_agg = _safe_pct(l_sh_total, l_id_total),
            mfg_share_pp = _safe_share_pp(L_sh[1, r, t], l_sh_total, L_id[1, r, t], l_id_total),
            svc_share_pp = _safe_share_pp(L_sh[4, r, t], l_sh_total, L_id[4, r, t], l_id_total),
            net_migration_diff = net_mig_sh - net_mig_id,
        ))
    end
end

df = sort!(DataFrame(rows), [:t, :state_idx])
mkpath(output_dir)
shock_slug = _slug(shock_name)
out_path = joinpath(output_dir, "state_maps_data_$(profile_tag)_$(shock_slug).csv")
CSV.write(out_path, df)
println("Wrote $(nrow(df)) rows to $(out_path)")
