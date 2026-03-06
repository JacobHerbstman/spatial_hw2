SHELL := bash
include ../../shell_functions.make

.PHONY: all smoke toy_smoke identity_reference identity_fast toy_reference toy_fast prepare_inputs

all: identity_fast

prepare_inputs: ../input/Base_year_four_sectors.mat ../input/baseline_4sector_path_reference.jld2 | ../output

smoke: prepare_inputs
	@source ../../shell_functions.sh; PROFILE=fast SHOCK_INPUT_MODE=identity SHOCK_NAME=identity RECORD_TRACE=1 TOL_DYNAMIC=1e-3 MAX_ITER_DYNAMIC=5 MAX_ITER_STATIC=2000 USE_THREADS=1 THREADS_DYNAMIC=1 THREADS_STATIC=1 CONFIG_TAG=smoke Julia_pc_and_local --project=../../_lib/cdp_julia run_counterfactual.jl

toy_smoke: prepare_inputs
	@source ../../shell_functions.sh; PROFILE=fast SHOCK_INPUT_MODE=toy SHOCK_NAME=toy_smoke RECORD_TRACE=1 TOL_DYNAMIC=1e-3 MAX_ITER_DYNAMIC=5 MAX_ITER_STATIC=2000 USE_THREADS=1 THREADS_DYNAMIC=1 THREADS_STATIC=1 CONFIG_TAG=toy_smoke Julia_pc_and_local --project=../../_lib/cdp_julia run_counterfactual.jl

identity_reference: prepare_inputs
	@source ../../shell_functions.sh; PROFILE=reference SHOCK_INPUT_MODE=identity SHOCK_NAME=identity RECORD_TRACE=1 TOL_DYNAMIC=1e-3 MAX_ITER_DYNAMIC=1000 MAX_ITER_STATIC=2000 USE_THREADS=0 THREADS_DYNAMIC=0 THREADS_STATIC=0 CONFIG_TAG=identity_reference Julia_pc_and_local --project=../../_lib/cdp_julia run_counterfactual.jl

identity_fast: prepare_inputs
	@source ../../shell_functions.sh; PROFILE=fast SHOCK_INPUT_MODE=identity SHOCK_NAME=identity RECORD_TRACE=1 TOL_DYNAMIC=1e-3 MAX_ITER_DYNAMIC=1000 MAX_ITER_STATIC=2000 USE_THREADS=1 THREADS_DYNAMIC=1 THREADS_STATIC=1 CONFIG_TAG=identity_fast Julia_pc_and_local --project=../../_lib/cdp_julia run_counterfactual.jl

toy_reference: prepare_inputs
	@source ../../shell_functions.sh; PROFILE=reference SHOCK_INPUT_MODE=toy SHOCK_NAME=toy RECORD_TRACE=1 TOL_DYNAMIC=1e-3 MAX_ITER_DYNAMIC=1000 MAX_ITER_STATIC=2000 USE_THREADS=0 THREADS_DYNAMIC=0 THREADS_STATIC=0 CONFIG_TAG=toy_reference Julia_pc_and_local --project=../../_lib/cdp_julia run_counterfactual.jl

toy_fast: prepare_inputs
	@source ../../shell_functions.sh; PROFILE=fast SHOCK_INPUT_MODE=toy SHOCK_NAME=toy RECORD_TRACE=1 TOL_DYNAMIC=1e-3 MAX_ITER_DYNAMIC=1000 MAX_ITER_STATIC=2000 USE_THREADS=1 THREADS_DYNAMIC=1 THREADS_STATIC=1 CONFIG_TAG=toy_fast Julia_pc_and_local --project=../../_lib/cdp_julia run_counterfactual.jl

../input/Base_year_four_sectors.mat: ../../../CDP_codes_four_sectors/Base_year_four_sectors.mat | ../input
	ln -sf $< $@

../input/baseline_4sector_path_reference.jld2: ../../cdp4_baseline_dynamics/output/baseline_4sector_path_reference.jld2 | ../input
	ln -sf $< $@

include ../../generic.make
