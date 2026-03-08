using Pkg
Pkg.activate(normpath(joinpath(@__DIR__, "../../_lib/cdp_julia")))

using JLD2
using CSV
using DataFrames

const STATE_ABBRS = [
    "AL", "AK", "AZ", "AR", "CA", "CO", "CT", "DE", "FL", "GA",
    "HI", "ID", "IL", "IN", "IA", "KS", "KY", "LA", "ME", "MD",
    "MA", "MI", "MN", "MS", "MO", "MT", "NE", "NV", "NH", "NJ",
    "NM", "NY", "NC", "ND", "OH", "OK", "OR", "PA", "RI", "SC",
    "SD", "TN", "TX", "UT", "VT", "VA", "WA", "WV", "WI", "WY",
]
const SECTOR_NAMES = [
    "Manufacturing",
    "Construction",
    "Wholesale/Retail",
    "Services",
]

function _slug(x::AbstractString)
    y = replace(lowercase(strip(String(x))), r"[^a-z0-9]+" => "_")
    y = replace(y, r"_+" => "_")
    isempty(y) ? "default" : strip(y, '_')
end

function _safe_pct(shock::Float64, base::Float64)
    abs(base) < 1e-12 ? NaN : 100.0 * (shock / base - 1.0)
end

function _weighted_mean(values, weights)
    num = 0.0
    den = 0.0
    @inbounds for (v, w) in zip(values, weights)
        if isfinite(v) && isfinite(w) && w > 0
            num += v * w
            den += w
        end
    end
    den <= 1e-12 ? NaN : num / den
end

identity_file = get(ENV, "IDENTITY_OUTPUT_FILE", "../input/counterfactual_4sector_path_fast_identity.jld2")
shock_file = get(ENV, "SHOCK_OUTPUT_FILE", "../input/counterfactual_4sector_path_fast_toy_smoke.jld2")
profile_tag = get(ENV, "PROFILE_TAG", "fast")
shock_name = get(ENV, "SHOCK_NAME", "generic")
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

selected_t = [t for t in (2, 5, 10, 20, 50, 100, 199) if 1 <= t <= T]
if isempty(selected_t)
    error("No valid selected time periods for T=$(T).")
end
ranking_t = 199 <= T ? 199 : T

national_rows = NamedTuple[]
sector_rows = NamedTuple[]
state_sector_rows = NamedTuple[]
state_rankings_rows = NamedTuple[]

sizehint!(national_rows, T)
sizehint!(sector_rows, T * J)
sizehint!(state_sector_rows, length(selected_t) * R * J)
sizehint!(state_rankings_rows, R)

for t in 1:T
    agg_emp_id = sum(id.Ldyn[:, 1:R, t])
    agg_emp_sh = sum(sh.Ldyn[:, 1:R, t])
    agg_rw_id = _weighted_mean(vec(id.realwages[:, 1:R, t]), vec(id.Ldyn[:, 1:R, t]))
    agg_rw_sh = _weighted_mean(vec(sh.realwages[:, 1:R, t]), vec(sh.Ldyn[:, 1:R, t]))
    push!(national_rows, (
        t = t,
        agg_emp_id = agg_emp_id,
        agg_emp_shock = agg_emp_sh,
        agg_emp_pct = _safe_pct(agg_emp_sh, agg_emp_id),
        agg_rw_id = agg_rw_id,
        agg_rw_shock = agg_rw_sh,
        agg_rw_pct = _safe_pct(agg_rw_sh, agg_rw_id),
    ))

    for j in 1:J
        sector_emp_id = sum(id.Ldyn[j, 1:R, t])
        sector_emp_sh = sum(sh.Ldyn[j, 1:R, t])
        sector_rw_id = _weighted_mean(vec(id.realwages[j, 1:R, t]), vec(id.Ldyn[j, 1:R, t]))
        sector_rw_sh = _weighted_mean(vec(sh.realwages[j, 1:R, t]), vec(sh.Ldyn[j, 1:R, t]))
        push!(sector_rows, (
            t = t,
            sector_idx = j,
            sector_name = SECTOR_NAMES[j],
            emp_id = sector_emp_id,
            emp_shock = sector_emp_sh,
            emp_pct = _safe_pct(sector_emp_sh, sector_emp_id),
            rw_id = sector_rw_id,
            rw_shock = sector_rw_sh,
            rw_pct = _safe_pct(sector_rw_sh, sector_rw_id),
        ))
    end

    if t in selected_t
        for r in 1:R, j in 1:J
            push!(state_sector_rows, (
                t = t,
                state_idx = r,
                state_abbr = STATE_ABBRS[r],
                sector_idx = j,
                sector_name = SECTOR_NAMES[j],
                emp_id = id.Ldyn[j, r, t],
                emp_shock = sh.Ldyn[j, r, t],
                emp_pct = _safe_pct(sh.Ldyn[j, r, t], id.Ldyn[j, r, t]),
                rw_id = id.realwages[j, r, t],
                rw_shock = sh.realwages[j, r, t],
                rw_pct = _safe_pct(sh.realwages[j, r, t], id.realwages[j, r, t]),
            ))
        end
    end

    if t == ranking_t
        for r in 1:R
            emp_id = sum(id.Ldyn[:, r, t])
            emp_sh = sum(sh.Ldyn[:, r, t])
            rw_id = _weighted_mean(vec(id.realwages[:, r, t]), vec(id.Ldyn[:, r, t]))
            rw_sh = _weighted_mean(vec(sh.realwages[:, r, t]), vec(sh.Ldyn[:, r, t]))
            push!(state_rankings_rows, (
                t = t,
                state_idx = r,
                state_abbr = STATE_ABBRS[r],
                agg_emp_id = emp_id,
                agg_emp_shock = emp_sh,
                agg_emp_pct = _safe_pct(emp_sh, emp_id),
                agg_rw_id = rw_id,
                agg_rw_shock = rw_sh,
                agg_rw_pct = _safe_pct(rw_sh, rw_id),
            ))
        end
    end
end

timeseries = DataFrame(national_rows)
sector_timeseries = sort!(DataFrame(sector_rows), [:sector_idx, :t])
selected = sort!(DataFrame(state_sector_rows), [:t, :state_idx, :sector_idx])
rankings = sort!(DataFrame(state_rankings_rows), :state_idx)

mkpath(output_dir)
shock_slug = _slug(shock_name)

CSV.write(joinpath(output_dir, "broad_shock_timeseries_$(profile_tag)_$(shock_slug).csv"), timeseries)
CSV.write(joinpath(output_dir, "broad_shock_sector_timeseries_$(profile_tag)_$(shock_slug).csv"), sector_timeseries)
CSV.write(joinpath(output_dir, "broad_shock_state_sector_selected_t_$(profile_tag)_$(shock_slug).csv"), selected)
CSV.write(joinpath(output_dir, "broad_shock_state_rankings_$(profile_tag)_$(shock_slug).csv"), rankings)

println("Wrote broad shock impact CSVs to $(output_dir)")
