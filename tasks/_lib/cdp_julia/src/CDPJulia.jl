module CDPJulia

using LinearAlgebra
using Statistics
using MAT
using JLD2
using CSV
using DataFrames

include("types.jl")
include("io_mats.jl")
include("temporary_equilibrium.jl")
include("dynamics_baseline.jl")
include("validation.jl")
include("dynamics_counterfactual.jl")
include("validation_counterfactual.jl")

export ModelDims,
       ModelParams,
       BaseState4,
       TempEqResult,
       BaselinePath4,
       CounterfactualShocks4,
       CounterfactualPath4,
       TempEqWorkspace,
       BaselineWorkspace4,
       CounterfactualWorkspace4,
       load_base_state_4sector,
       load_matlab_ynew,
       load_baseline_anchor_y,
       load_counterfactual_shocks_4sector,
       toy_counterfactual_shocks_4sector,
       default_model_params,
       solve_temporary_equilibrium!,
       solve_temporary_equilibrium_inplace!,
       run_baseline_4sector,
       run_counterfactual_4sector,
       validate_baseline_4sector,
       parity_by_time,
       deterministic_delta,
       validate_counterfactual_core_4sector,
       validate_counterfactual_identity_4sector,
       validate_counterfactual_response_4sector,
       parity_by_time_counterfactual,
       deterministic_delta_counterfactual,
       fast_reference_delta_counterfactual

end
