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

export ModelDims,
       ModelParams,
       BaseState4,
       TempEqResult,
       BaselinePath4,
       TempEqWorkspace,
       BaselineWorkspace4,
       load_base_state_4sector,
       load_matlab_ynew,
       default_model_params,
       solve_temporary_equilibrium!,
       solve_temporary_equilibrium_inplace!,
       run_baseline_4sector,
       validate_baseline_4sector,
       parity_by_time,
       deterministic_delta

end
