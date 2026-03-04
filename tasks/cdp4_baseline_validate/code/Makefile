SHELL := bash
include ../../shell_functions.make

.PHONY: all smoke replication_reference replication_fast prepare_inputs

all: replication_fast

prepare_inputs: ../input/Hvectnoshock.mat ../input/Base_year_four_sectors.mat | ../output

smoke: prepare_inputs ../input/baseline_4sector_path_fast.jld2
	@source ../../shell_functions.sh; PROFILE=fast VALIDATION_MODE=smoke REL_TOL=5e-2 BASELINE_OUTPUT_FILE=../input/baseline_4sector_path_fast.jld2 RECORD_TRACE=1 MAX_ITER_DYNAMIC=5 MAX_ITER_STATIC=2000 USE_THREADS=1 THREADS_DYNAMIC=1 THREADS_STATIC=1 CONFIG_TAG=smoke Julia_pc_and_local --project=../../_lib/cdp_julia validate_baseline.jl

replication_reference: prepare_inputs ../input/baseline_4sector_path_reference.jld2
	@source ../../shell_functions.sh; PROFILE=reference VALIDATION_MODE=replication_reference REL_TOL=1e-3 BASELINE_OUTPUT_FILE=../input/baseline_4sector_path_reference.jld2 RECORD_TRACE=1 MAX_ITER_DYNAMIC=1000 MAX_ITER_STATIC=2000 USE_THREADS=0 THREADS_DYNAMIC=0 THREADS_STATIC=0 CONFIG_TAG=replication_reference Julia_pc_and_local --project=../../_lib/cdp_julia validate_baseline.jl

replication_fast: prepare_inputs ../input/baseline_4sector_path_fast.jld2
	@source ../../shell_functions.sh; PROFILE=fast VALIDATION_MODE=replication_fast REL_TOL=1e-3 BASELINE_OUTPUT_FILE=../input/baseline_4sector_path_fast.jld2 RECORD_TRACE=1 MAX_ITER_DYNAMIC=1000 MAX_ITER_STATIC=2000 USE_THREADS=1 THREADS_DYNAMIC=1 THREADS_STATIC=1 CONFIG_TAG=replication_fast Julia_pc_and_local --project=../../_lib/cdp_julia validate_baseline.jl

../input/baseline_4sector_path_fast.jld2: ../../cdp4_baseline_dynamics/output/baseline_4sector_path_fast.jld2 | ../input
	ln -sf $< $@

../input/baseline_4sector_path_reference.jld2: ../../cdp4_baseline_dynamics/output/baseline_4sector_path_reference.jld2 | ../input
	ln -sf $< $@

../input/Hvectnoshock.mat: ../../../CDP_codes_four_sectors/Hvectnoshock.mat | ../input
	ln -sf $< $@

../input/Base_year_four_sectors.mat: ../../../CDP_codes_four_sectors/Base_year_four_sectors.mat | ../input
	ln -sf $< $@

include ../../generic.make
