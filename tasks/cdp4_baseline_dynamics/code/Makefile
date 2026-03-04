SHELL := bash
include ../../shell_functions.make

.PHONY: all smoke replication_reference replication_fast prepare_inputs

all: replication_fast

prepare_inputs: ../input/Base_year_four_sectors.mat ../input/static_checks.jld2 | ../output

smoke: prepare_inputs
	@source ../../shell_functions.sh; PROFILE=fast RECORD_TRACE=1 MAX_ITER_DYNAMIC=5 MAX_ITER_STATIC=2000 USE_THREADS=1 THREADS_DYNAMIC=1 THREADS_STATIC=1 CONFIG_TAG=smoke Julia_pc_and_local --project=../../_lib/cdp_julia run_baseline.jl

replication_reference: prepare_inputs
	@source ../../shell_functions.sh; PROFILE=reference RECORD_TRACE=1 MAX_ITER_DYNAMIC=1000 MAX_ITER_STATIC=2000 USE_THREADS=0 THREADS_DYNAMIC=0 THREADS_STATIC=0 CONFIG_TAG=replication_reference Julia_pc_and_local --project=../../_lib/cdp_julia run_baseline.jl

replication_fast: prepare_inputs
	@source ../../shell_functions.sh; PROFILE=fast RECORD_TRACE=1 MAX_ITER_DYNAMIC=1000 MAX_ITER_STATIC=2000 USE_THREADS=1 THREADS_DYNAMIC=1 THREADS_STATIC=1 CONFIG_TAG=replication_fast Julia_pc_and_local --project=../../_lib/cdp_julia run_baseline.jl

../input/Base_year_four_sectors.mat: ../../../CDP_codes_four_sectors/Base_year_four_sectors.mat | ../input
	ln -sf $< $@

../input/static_checks.jld2: ../../cdp4_baseline_solver/output/static_checks.jld2 | ../input
	ln -sf $< $@

include ../../generic.make
