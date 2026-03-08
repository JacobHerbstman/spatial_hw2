SHELL := bash
include ../../shell_functions.make
PYTHON ?= $(shell for p in /Users/jacobherbstman/miniconda3/bin/python3 python3 /Library/Frameworks/Python.framework/Versions/3.13/bin/python3; do if [ -x "$$p" ] && "$$p" -c "import pandas,matplotlib" >/dev/null 2>&1; then echo "$$p"; exit 0; fi; done; echo python3)

.PHONY: all smoke identity_reference identity_fast toy_reference toy_fast generic_validate report report_smoke report_identity_fast report_identity_reference report_toy_fast report_toy_reference impact_report_toy_smoke_fast impact_report_toy_reference impact_report_generic impact_report_focus state_maps state_maps_toy_reference state_maps_toy_fast state_maps_toy_smoke_fast prepare_inputs

all: identity_fast

prepare_inputs: ../input/Base_year_four_sectors.mat ../input/baseline_4sector_path_reference.jld2 | ../output

smoke: prepare_inputs ../input/counterfactual_4sector_path_fast_identity.jld2
	@source ../../shell_functions.sh; VALIDATION_KIND=identity VALIDATION_MODE=smoke_identity SHOCK_INPUT_MODE=identity SHOCK_NAME=identity REL_TOL=5e-2 TOL_DYNAMIC=1e-3 COUNTERFACTUAL_OUTPUT_FILE=../input/counterfactual_4sector_path_fast_identity.jld2 BASELINE_ANCHOR_FILE=../input/baseline_4sector_path_reference.jld2 RECORD_TRACE=1 MAX_ITER_DYNAMIC=5 MAX_ITER_STATIC=2000 USE_THREADS=1 THREADS_DYNAMIC=1 THREADS_STATIC=1 CONFIG_TAG=smoke Julia_pc_and_local --project=../../_lib/cdp_julia validate_counterfactual.jl

identity_reference: prepare_inputs ../input/counterfactual_4sector_path_reference_identity.jld2
	@source ../../shell_functions.sh; PROFILE=reference VALIDATION_KIND=identity VALIDATION_MODE=identity_reference SHOCK_INPUT_MODE=identity SHOCK_NAME=identity REL_TOL=1e-3 ROUNDTRIP_TOL=1e-3 TOL_DYNAMIC=1e-3 COUNTERFACTUAL_OUTPUT_FILE=../input/counterfactual_4sector_path_reference_identity.jld2 BASELINE_ANCHOR_FILE=../input/baseline_4sector_path_reference.jld2 RECORD_TRACE=1 MAX_ITER_DYNAMIC=1000 MAX_ITER_STATIC=2000 USE_THREADS=0 THREADS_DYNAMIC=0 THREADS_STATIC=0 CONFIG_TAG=identity_reference Julia_pc_and_local --project=../../_lib/cdp_julia validate_counterfactual.jl

identity_fast: prepare_inputs ../input/counterfactual_4sector_path_fast_identity.jld2 ../input/counterfactual_4sector_path_reference_identity.jld2
	@source ../../shell_functions.sh; PROFILE=fast VALIDATION_KIND=identity VALIDATION_MODE=identity_fast SHOCK_INPUT_MODE=identity SHOCK_NAME=identity REL_TOL=1e-3 ROUNDTRIP_TOL=1e-3 TOL_DYNAMIC=1e-3 FAST_REF_TOL=1e-4 COUNTERFACTUAL_OUTPUT_FILE=../input/counterfactual_4sector_path_fast_identity.jld2 REFERENCE_OUTPUT_FILE=../input/counterfactual_4sector_path_reference_identity.jld2 BASELINE_ANCHOR_FILE=../input/baseline_4sector_path_reference.jld2 RECORD_TRACE=1 MAX_ITER_DYNAMIC=1000 MAX_ITER_STATIC=2000 USE_THREADS=1 THREADS_DYNAMIC=1 THREADS_STATIC=1 CONFIG_TAG=identity_fast Julia_pc_and_local --project=../../_lib/cdp_julia validate_counterfactual.jl

toy_reference: prepare_inputs ../input/counterfactual_4sector_path_reference_toy.jld2 ../input/counterfactual_4sector_path_reference_identity.jld2
	@source ../../shell_functions.sh; PROFILE=reference VALIDATION_KIND=toy VALIDATION_MODE=toy_reference SHOCK_INPUT_MODE=toy SHOCK_NAME=toy REL_TOL=1e-3 TOL_DYNAMIC=1e-3 COUNTERFACTUAL_OUTPUT_FILE=../input/counterfactual_4sector_path_reference_toy.jld2 IDENTITY_OUTPUT_FILE=../input/counterfactual_4sector_path_reference_identity.jld2 BASELINE_ANCHOR_FILE=../input/baseline_4sector_path_reference.jld2 RECORD_TRACE=1 MAX_ITER_DYNAMIC=1000 MAX_ITER_STATIC=2000 USE_THREADS=0 THREADS_DYNAMIC=0 THREADS_STATIC=0 CONFIG_TAG=toy_reference Julia_pc_and_local --project=../../_lib/cdp_julia validate_counterfactual.jl

toy_fast: prepare_inputs ../input/counterfactual_4sector_path_fast_toy.jld2 ../input/counterfactual_4sector_path_reference_toy.jld2 ../input/counterfactual_4sector_path_fast_identity.jld2
	@source ../../shell_functions.sh; PROFILE=fast VALIDATION_KIND=toy VALIDATION_MODE=toy_fast SHOCK_INPUT_MODE=toy SHOCK_NAME=toy REL_TOL=1e-3 TOL_DYNAMIC=1e-3 FAST_REF_TOL=1e-4 COUNTERFACTUAL_OUTPUT_FILE=../input/counterfactual_4sector_path_fast_toy.jld2 IDENTITY_OUTPUT_FILE=../input/counterfactual_4sector_path_fast_identity.jld2 REFERENCE_OUTPUT_FILE=../input/counterfactual_4sector_path_reference_toy.jld2 BASELINE_ANCHOR_FILE=../input/baseline_4sector_path_reference.jld2 RECORD_TRACE=1 MAX_ITER_DYNAMIC=1000 MAX_ITER_STATIC=2000 USE_THREADS=1 THREADS_DYNAMIC=1 THREADS_STATIC=1 CONFIG_TAG=toy_fast Julia_pc_and_local --project=../../_lib/cdp_julia validate_counterfactual.jl

generic_validate: prepare_inputs
	@test -n "$${COUNTERFACTUAL_OUTPUT_FILE:-}" || (echo "COUNTERFACTUAL_OUTPUT_FILE is required for generic_validate" >&2; exit 1)
	@if [ "$${SHOCK_INPUT_MODE:-csv}" = "csv" ] && [ -z "$${LAMBDA_CSV:-}" ]; then echo "LAMBDA_CSV is required when SHOCK_INPUT_MODE=csv" >&2; exit 1; fi
	@if [ "$${SHOCK_INPUT_MODE:-csv}" = "mat" ] && [ -z "$${SHOCK_MAT:-}" ]; then echo "SHOCK_MAT is required when SHOCK_INPUT_MODE=mat" >&2; exit 1; fi
	@source ../../shell_functions.sh; PROFILE=$${PROFILE:-fast} VALIDATION_KIND=generic VALIDATION_MODE=$${VALIDATION_MODE:-generic_$${PROFILE:-fast}} SHOCK_INPUT_MODE=$${SHOCK_INPUT_MODE:-csv} SHOCK_NAME=$${SHOCK_NAME:-generic} REL_TOL=$${REL_TOL:-1e-3} FAST_REF_TOL=$${FAST_REF_TOL:-1e-4} TOL_DYNAMIC=$${TOL_DYNAMIC:-1e-3} COUNTERFACTUAL_OUTPUT_FILE=$${COUNTERFACTUAL_OUTPUT_FILE} IDENTITY_OUTPUT_FILE=$${IDENTITY_OUTPUT_FILE:-} REFERENCE_OUTPUT_FILE=$${REFERENCE_OUTPUT_FILE:-} BASELINE_ANCHOR_FILE=$${BASELINE_ANCHOR_FILE:-../input/baseline_4sector_path_reference.jld2} LAMBDA_CSV=$${LAMBDA_CSV:-} KAPPA_CSV=$${KAPPA_CSV:-} SHOCK_MAT=$${SHOCK_MAT:-} TIME_HORIZON=$${TIME_HORIZON:-200} RECORD_TRACE=1 MAX_ITER_DYNAMIC=$${MAX_ITER_DYNAMIC:-1000} MAX_ITER_STATIC=$${MAX_ITER_STATIC:-2000} USE_THREADS=$${USE_THREADS:-1} THREADS_DYNAMIC=$${THREADS_DYNAMIC:-1} THREADS_STATIC=$${THREADS_STATIC:-1} FAIL_ON_CHECKS=$${FAIL_ON_CHECKS:-0} RESPONSE_TOL=$${RESPONSE_TOL:-1e-12} PRE_ACTIVATION_TOL=$${PRE_ACTIVATION_TOL:-1e-12} CONFIG_TAG=$${CONFIG_TAG:-generic_validate} Julia_pc_and_local --project=../../_lib/cdp_julia validate_counterfactual.jl

report:
	@mkdir -p ../temp/mpl
	@MPLCONFIGDIR=../temp/mpl $(PYTHON) make_dynamics_report.py --profile $${PROFILE:-fast} --shock $${SHOCK_NAME:-identity}

report_smoke:
	@PROFILE=fast SHOCK_NAME=identity $(MAKE) report

report_identity_fast:
	@PROFILE=fast SHOCK_NAME=identity $(MAKE) report

report_identity_reference:
	@PROFILE=reference SHOCK_NAME=identity $(MAKE) report

report_toy_fast:
	@PROFILE=fast SHOCK_NAME=toy $(MAKE) report

report_toy_reference:
	@PROFILE=reference SHOCK_NAME=toy $(MAKE) report

impact_report_toy_smoke_fast: ../input/counterfactual_4sector_path_fast_identity.jld2 ../input/counterfactual_4sector_path_fast_toy_smoke.jld2
	@source ../../shell_functions.sh; PROFILE_TAG=fast SHOCK_NAME=toy_smoke IDENTITY_OUTPUT_FILE=../input/counterfactual_4sector_path_fast_identity.jld2 SHOCK_OUTPUT_FILE=../input/counterfactual_4sector_path_fast_toy_smoke.jld2 OUTPUT_DIR=../output SHOCK_SECTOR=1 SHOCK_REGION=1 Julia_pc_and_local --project=../../_lib/cdp_julia extract_key_economic_series.jl
	@mkdir -p ../temp/mpl
	@MPLCONFIGDIR=../temp/mpl $(PYTHON) make_key_econ_report.py --profile fast --shock toy_smoke --input-dir ../output --output-dir ../output

impact_report_toy_reference: ../input/counterfactual_4sector_path_reference_identity.jld2 ../input/counterfactual_4sector_path_reference_toy.jld2 ../input/Base_year_four_sectors.mat
	@source ../../shell_functions.sh; PROFILE_TAG=reference SHOCK_NAME=toy SHOCK_INPUT_MODE=toy IDENTITY_OUTPUT_FILE=../input/counterfactual_4sector_path_reference_identity.jld2 SHOCK_OUTPUT_FILE=../input/counterfactual_4sector_path_reference_toy.jld2 BASE_STATE_MAT=../input/Base_year_four_sectors.mat OUTPUT_DIR=../output SHOCK_SECTOR=1 SHOCK_REGION=1 TOY_SECTOR=1 TOY_REGION=1 TOY_PERIODS=20 TOY_LAMBDA_MULTIPLIER=1.05 MAX_ITER_STATIC=2000 Julia_pc_and_local --project=../../_lib/cdp_julia extract_key_economic_series.jl
	@mkdir -p ../temp/mpl
	@MPLCONFIGDIR=../temp/mpl $(PYTHON) make_key_econ_report.py --profile reference --shock toy --input-dir ../output --output-dir ../output

impact_report_generic:
	@test -n "$${IDENTITY_OUTPUT_FILE:-}" || (echo "IDENTITY_OUTPUT_FILE is required for impact_report_generic" >&2; exit 1)
	@test -n "$${SHOCK_OUTPUT_FILE:-}" || (echo "SHOCK_OUTPUT_FILE is required for impact_report_generic" >&2; exit 1)
	@source ../../shell_functions.sh; PROFILE_TAG=$${PROFILE_TAG:-$${PROFILE:-fast}} SHOCK_NAME=$${SHOCK_NAME:-generic} IDENTITY_OUTPUT_FILE=$${IDENTITY_OUTPUT_FILE} SHOCK_OUTPUT_FILE=$${SHOCK_OUTPUT_FILE} OUTPUT_DIR=$${OUTPUT_DIR:-../output} Julia_pc_and_local --project=../../_lib/cdp_julia extract_broad_shock_impacts.jl
	@mkdir -p ../temp/mpl
	@MPLCONFIGDIR=../temp/mpl $(PYTHON) make_broad_shock_report.py --profile $${PROFILE_TAG:-$${PROFILE:-fast}} --shock $${SHOCK_NAME:-generic} --input-dir $${OUTPUT_DIR:-../output} --output-dir $${OUTPUT_DIR:-../output}

impact_report_focus:
	@test -n "$${IDENTITY_OUTPUT_FILE:-}" || (echo "IDENTITY_OUTPUT_FILE is required for impact_report_focus" >&2; exit 1)
	@test -n "$${SHOCK_OUTPUT_FILE:-}" || (echo "SHOCK_OUTPUT_FILE is required for impact_report_focus" >&2; exit 1)
	@if [ "$${SHOCK_INPUT_MODE:-csv}" = "csv" ] && [ -z "$${LAMBDA_CSV:-}" ]; then echo "LAMBDA_CSV is required when SHOCK_INPUT_MODE=csv" >&2; exit 1; fi
	@if [ "$${SHOCK_INPUT_MODE:-csv}" = "mat" ] && [ -z "$${SHOCK_MAT:-}" ]; then echo "SHOCK_MAT is required when SHOCK_INPUT_MODE=mat" >&2; exit 1; fi
	@source ../../shell_functions.sh; PROFILE_TAG=$${PROFILE_TAG:-$${PROFILE:-fast}} SHOCK_NAME=$${SHOCK_NAME:-generic} SHOCK_INPUT_MODE=$${SHOCK_INPUT_MODE:-csv} LAMBDA_CSV=$${LAMBDA_CSV:-} KAPPA_CSV=$${KAPPA_CSV:-} SHOCK_MAT=$${SHOCK_MAT:-} IDENTITY_OUTPUT_FILE=$${IDENTITY_OUTPUT_FILE} SHOCK_OUTPUT_FILE=$${SHOCK_OUTPUT_FILE} BASE_STATE_MAT=$${BASE_STATE_MAT:-../input/Base_year_four_sectors.mat} OUTPUT_DIR=$${OUTPUT_DIR:-../output} SHOCK_SECTOR=$${SHOCK_SECTOR:-4} SHOCK_REGION=$${SHOCK_REGION:-32} MAX_ITER_STATIC=$${MAX_ITER_STATIC:-2000} Julia_pc_and_local --project=../../_lib/cdp_julia extract_key_economic_series.jl
	@mkdir -p ../temp/mpl
	@MPLCONFIGDIR=../temp/mpl $(PYTHON) make_key_econ_report.py --profile $${PROFILE_TAG:-$${PROFILE:-fast}} --shock $${SHOCK_NAME:-generic} --input-dir $${OUTPUT_DIR:-../output} --output-dir $${OUTPUT_DIR:-../output}

state_maps:
	@source ../../shell_functions.sh; PROFILE_TAG=$${PROFILE_TAG:-$${PROFILE:-fast}} SHOCK_NAME=$${SHOCK_NAME:-toy} IDENTITY_OUTPUT_FILE=$${IDENTITY_OUTPUT_FILE:-../input/counterfactual_4sector_path_$${PROFILE_TAG:-$${PROFILE:-fast}}_identity.jld2} SHOCK_OUTPUT_FILE=$${SHOCK_OUTPUT_FILE:-../input/counterfactual_4sector_path_$${PROFILE_TAG:-$${PROFILE:-fast}}_$${SHOCK_NAME:-toy}.jld2} OUTPUT_DIR=$${OUTPUT_DIR:-../output} Julia_pc_and_local --project=../../_lib/cdp_julia extract_state_maps_data.jl
	@mkdir -p ../temp/mpl
	@MAP_ARGS="--renderer $${MAP_RENDERER:-auto}"; if [ -n "$${STATE_GEOMETRY_FILE:-}" ]; then MAP_ARGS="$$MAP_ARGS --geometry-file $$STATE_GEOMETRY_FILE"; fi; MPLCONFIGDIR=../temp/mpl $(PYTHON) make_state_maps.py --profile $${PROFILE_TAG:-$${PROFILE:-fast}} --shock $${SHOCK_NAME:-toy} --input-dir $${OUTPUT_DIR:-../output} --output-dir $${OUTPUT_DIR:-../output} $$MAP_ARGS

state_maps_toy_reference: ../input/counterfactual_4sector_path_reference_identity.jld2 ../input/counterfactual_4sector_path_reference_toy.jld2
	@PROFILE_TAG=reference SHOCK_NAME=toy IDENTITY_OUTPUT_FILE=../input/counterfactual_4sector_path_reference_identity.jld2 SHOCK_OUTPUT_FILE=../input/counterfactual_4sector_path_reference_toy.jld2 OUTPUT_DIR=../output $(MAKE) state_maps

state_maps_toy_fast: ../input/counterfactual_4sector_path_fast_identity.jld2 ../input/counterfactual_4sector_path_fast_toy.jld2
	@PROFILE_TAG=fast SHOCK_NAME=toy IDENTITY_OUTPUT_FILE=../input/counterfactual_4sector_path_fast_identity.jld2 SHOCK_OUTPUT_FILE=../input/counterfactual_4sector_path_fast_toy.jld2 OUTPUT_DIR=../output $(MAKE) state_maps

state_maps_toy_smoke_fast: ../input/counterfactual_4sector_path_fast_identity.jld2 ../input/counterfactual_4sector_path_fast_toy_smoke.jld2
	@PROFILE_TAG=fast SHOCK_NAME=toy_smoke IDENTITY_OUTPUT_FILE=../input/counterfactual_4sector_path_fast_identity.jld2 SHOCK_OUTPUT_FILE=../input/counterfactual_4sector_path_fast_toy_smoke.jld2 OUTPUT_DIR=../output $(MAKE) state_maps

../input/counterfactual_4sector_path_reference_identity.jld2: ../../cdp4_counterfactual_dynamics/output/counterfactual_4sector_path_reference_identity.jld2 | ../input
	ln -sf $< $@

../input/counterfactual_4sector_path_fast_identity.jld2: ../../cdp4_counterfactual_dynamics/output/counterfactual_4sector_path_fast_identity.jld2 | ../input
	ln -sf $< $@

../input/counterfactual_4sector_path_reference_toy.jld2: ../../cdp4_counterfactual_dynamics/output/counterfactual_4sector_path_reference_toy.jld2 | ../input
	ln -sf $< $@

../input/counterfactual_4sector_path_fast_toy.jld2: ../../cdp4_counterfactual_dynamics/output/counterfactual_4sector_path_fast_toy.jld2 | ../input
	ln -sf $< $@

../input/counterfactual_4sector_path_fast_toy_smoke.jld2: ../../cdp4_counterfactual_dynamics/output/counterfactual_4sector_path_fast_toy_smoke.jld2 | ../input
	ln -sf $< $@

../input/Base_year_four_sectors.mat: ../../../CDP_codes_four_sectors/Base_year_four_sectors.mat | ../input
	ln -sf $< $@

../input/baseline_4sector_path_reference.jld2: ../../cdp4_baseline_dynamics/output/baseline_4sector_path_reference.jld2 | ../input
	ln -sf $< $@

include ../../generic.make
