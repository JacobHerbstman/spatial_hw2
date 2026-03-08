SHELL := bash
include ../../shell_functions.make
PYTHON ?= $(shell for p in /Users/jacobherbstman/miniconda3/bin/python3 python3 /Library/Frameworks/Python.framework/Versions/3.13/bin/python3; do if [ -x "$$p" ] && "$$p" -c "import pandas,json" >/dev/null 2>&1; then echo "$$p"; exit 0; fi; done; echo python3)

MATRIX ?= ../../cognitive_intensity_data/output/cognitive_intensity_matrix.csv
DELTA_CANDIDATES ?= 0.01,0.005,0.0025,0.001,0.0005,0.00025
TIME_HORIZON ?= 200
FOCUS_SECTOR ?= 4
FOCUS_REGION ?= 32
REFERENCE_TOL_DYNAMIC ?= 1e-3

IMMEDIATE_SHOCK_NAME := cognitive_immediate
ANTICIPATED_SHOCK_NAME := cognitive_anticipated

DYNAMICS_DIR := ../../cdp4_counterfactual_dynamics/code
VALIDATE_DIR := ../../cdp4_counterfactual_validate/code
DYNAMICS_OUTPUT_DIR := $(abspath ../../cdp4_counterfactual_dynamics/output)
VALIDATE_OUTPUT_DIR := $(abspath ../../cdp4_counterfactual_validate/output)
TASK_OUTPUT_DIR := $(abspath ../output)
TASK_TEMP_DIR := $(abspath ../temp)
DELTA_SELECTION_FILE := $(abspath ../output/selected_reference_delta.json)
SOLVER_SETTINGS_FILE := $(abspath ../output/reference_solver_settings.json)
BASELINE_VALIDATION_SOURCE := $(abspath ../../cdp4_baseline_validate/output/validation_report_4sector_reference.csv)
BASELINE_VALIDATION_LINK := ../output/baseline_validation_reference.csv
REFERENCE_BRIEF_FILE := $(abspath ../output/reference_figure_brief.md)

REFERENCE_ENV := WARM_START_STATIC=0 USE_ANDERSON=0 HVECT_RELAX=0.5 CONFIRM_FIXED_POINT=1 USE_THREADS=0 THREADS_DYNAMIC=0 THREADS_STATIC=0
FAST_ENV := WARM_START_STATIC=1 USE_ANDERSON=0 HVECT_RELAX=0.05 CONFIRM_FIXED_POINT=1 USE_THREADS=1 THREADS_DYNAMIC=1 THREADS_STATIC=1

.PHONY: all generate_shocks select_reference_delta prepare_reference_inputs prepare_fast_inputs reference fast submission_reference submission_fast run_reference run_fast validate_reference validate_fast report_reference report_fast immediate_reference anticipated_reference immediate_fast anticipated_fast validate_immediate_reference validate_anticipated_reference validate_immediate_fast validate_anticipated_fast report_immediate_reference report_anticipated_reference report_immediate_fast report_anticipated_fast reference_figure_brief collect_baseline_reference clean_submission_outputs all_reference all_fast clean

all: submission_reference

reference: submission_reference

fast: submission_fast

generate_shocks: select_reference_delta

select_reference_delta: $(DELTA_SELECTION_FILE)

prepare_reference_inputs: ../input/Base_year_four_sectors.mat ../input/baseline_4sector_path_reference.jld2 ../input/counterfactual_4sector_path_reference_identity.jld2 ../../cdp4_baseline_validate/output/validation_report_4sector_reference.csv | ../output ../temp

prepare_fast_inputs: prepare_reference_inputs ../input/counterfactual_4sector_path_fast_identity.jld2

submission_reference:
	$(MAKE) clean_submission_outputs
	$(MAKE) collect_baseline_reference
	$(MAKE) report_reference
	$(MAKE) reference_figure_brief

submission_fast:
	$(MAKE) clean_submission_outputs
	$(MAKE) collect_baseline_reference
	$(MAKE) report_fast

run_reference: immediate_reference anticipated_reference

run_fast: immediate_fast anticipated_fast

validate_reference: validate_immediate_reference validate_anticipated_reference

validate_fast: validate_immediate_fast validate_anticipated_fast

report_reference: report_immediate_reference report_anticipated_reference

report_fast: report_immediate_fast report_anticipated_fast

reference_figure_brief: report_reference
	$(PYTHON) write_reference_figure_brief.py --output-dir "$(TASK_OUTPUT_DIR)" --dynamics-output-dir "$(DYNAMICS_OUTPUT_DIR)" --validate-output-dir "$(VALIDATE_OUTPUT_DIR)" --profile reference --immediate-shock "$(IMMEDIATE_SHOCK_NAME)" --anticipated-shock "$(ANTICIPATED_SHOCK_NAME)"

all_reference: submission_reference

all_fast: submission_fast

immediate_reference: $(DELTA_SELECTION_FILE) prepare_reference_inputs
	$(REFERENCE_ENV) $(MAKE) -C $(DYNAMICS_DIR) csv_reference LAMBDA_CSV="$(abspath ../output/lambda_immediate.csv)" SHOCK_NAME="$(IMMEDIATE_SHOCK_NAME)" TIME_HORIZON="$(TIME_HORIZON)" BASELINE_ANCHOR_FILE="$(abspath ../input/baseline_4sector_path_reference.jld2)" CONFIG_TAG="$(IMMEDIATE_SHOCK_NAME)_reference_submission"

anticipated_reference: $(DELTA_SELECTION_FILE) prepare_reference_inputs
	$(REFERENCE_ENV) $(MAKE) -C $(DYNAMICS_DIR) csv_reference LAMBDA_CSV="$(abspath ../output/lambda_anticipated.csv)" SHOCK_NAME="$(ANTICIPATED_SHOCK_NAME)" TIME_HORIZON="$(TIME_HORIZON)" BASELINE_ANCHOR_FILE="$(abspath ../input/baseline_4sector_path_reference.jld2)" CONFIG_TAG="$(ANTICIPATED_SHOCK_NAME)_reference_submission"

immediate_fast: $(DELTA_SELECTION_FILE) prepare_fast_inputs
	$(FAST_ENV) $(MAKE) -C $(DYNAMICS_DIR) csv_fast LAMBDA_CSV="$(abspath ../output/lambda_immediate.csv)" SHOCK_NAME="$(IMMEDIATE_SHOCK_NAME)" TIME_HORIZON="$(TIME_HORIZON)" BASELINE_ANCHOR_FILE="$(abspath ../input/baseline_4sector_path_reference.jld2)" CONFIG_TAG="$(IMMEDIATE_SHOCK_NAME)_fast_debug"

anticipated_fast: $(DELTA_SELECTION_FILE) prepare_fast_inputs
	$(FAST_ENV) $(MAKE) -C $(DYNAMICS_DIR) csv_fast LAMBDA_CSV="$(abspath ../output/lambda_anticipated.csv)" SHOCK_NAME="$(ANTICIPATED_SHOCK_NAME)" TIME_HORIZON="$(TIME_HORIZON)" BASELINE_ANCHOR_FILE="$(abspath ../input/baseline_4sector_path_reference.jld2)" CONFIG_TAG="$(ANTICIPATED_SHOCK_NAME)_fast_debug"

validate_immediate_reference: immediate_reference
	$(REFERENCE_ENV) $(MAKE) -C $(VALIDATE_DIR) generic_validate PROFILE=reference SHOCK_INPUT_MODE=csv SHOCK_NAME="$(IMMEDIATE_SHOCK_NAME)" LAMBDA_CSV="$(abspath ../output/lambda_immediate.csv)" TIME_HORIZON="$(TIME_HORIZON)" COUNTERFACTUAL_OUTPUT_FILE="$(abspath ../../cdp4_counterfactual_dynamics/output/counterfactual_4sector_path_reference_$(IMMEDIATE_SHOCK_NAME).jld2)" IDENTITY_OUTPUT_FILE="$(abspath ../input/counterfactual_4sector_path_reference_identity.jld2)" BASELINE_ANCHOR_FILE="$(abspath ../input/baseline_4sector_path_reference.jld2)" TOL_DYNAMIC=$(REFERENCE_TOL_DYNAMIC) FAIL_ON_CHECKS=1 REQUIRE_T1_RESPONSE=1 CONFIG_TAG="$(IMMEDIATE_SHOCK_NAME)_reference_validate_submission"

validate_anticipated_reference: anticipated_reference
	$(REFERENCE_ENV) $(MAKE) -C $(VALIDATE_DIR) generic_validate PROFILE=reference SHOCK_INPUT_MODE=csv SHOCK_NAME="$(ANTICIPATED_SHOCK_NAME)" LAMBDA_CSV="$(abspath ../output/lambda_anticipated.csv)" TIME_HORIZON="$(TIME_HORIZON)" COUNTERFACTUAL_OUTPUT_FILE="$(abspath ../../cdp4_counterfactual_dynamics/output/counterfactual_4sector_path_reference_$(ANTICIPATED_SHOCK_NAME).jld2)" IDENTITY_OUTPUT_FILE="$(abspath ../input/counterfactual_4sector_path_reference_identity.jld2)" BASELINE_ANCHOR_FILE="$(abspath ../input/baseline_4sector_path_reference.jld2)" TOL_DYNAMIC=$(REFERENCE_TOL_DYNAMIC) FAIL_ON_CHECKS=1 REQUIRE_T1_RESPONSE=1 CONFIG_TAG="$(ANTICIPATED_SHOCK_NAME)_reference_validate_submission"

validate_immediate_fast: immediate_fast
	$(FAST_ENV) $(MAKE) -C $(VALIDATE_DIR) generic_validate PROFILE=fast SHOCK_INPUT_MODE=csv SHOCK_NAME="$(IMMEDIATE_SHOCK_NAME)" LAMBDA_CSV="$(abspath ../output/lambda_immediate.csv)" TIME_HORIZON="$(TIME_HORIZON)" COUNTERFACTUAL_OUTPUT_FILE="$(abspath ../../cdp4_counterfactual_dynamics/output/counterfactual_4sector_path_fast_$(IMMEDIATE_SHOCK_NAME).jld2)" IDENTITY_OUTPUT_FILE="$(abspath ../input/counterfactual_4sector_path_fast_identity.jld2)" BASELINE_ANCHOR_FILE="$(abspath ../input/baseline_4sector_path_reference.jld2)" TOL_DYNAMIC=1e-3 FAIL_ON_CHECKS=1 REQUIRE_T1_RESPONSE=1 CONFIG_TAG="$(IMMEDIATE_SHOCK_NAME)_fast_validate_debug"

validate_anticipated_fast: anticipated_fast
	$(FAST_ENV) $(MAKE) -C $(VALIDATE_DIR) generic_validate PROFILE=fast SHOCK_INPUT_MODE=csv SHOCK_NAME="$(ANTICIPATED_SHOCK_NAME)" LAMBDA_CSV="$(abspath ../output/lambda_anticipated.csv)" TIME_HORIZON="$(TIME_HORIZON)" COUNTERFACTUAL_OUTPUT_FILE="$(abspath ../../cdp4_counterfactual_dynamics/output/counterfactual_4sector_path_fast_$(ANTICIPATED_SHOCK_NAME).jld2)" IDENTITY_OUTPUT_FILE="$(abspath ../input/counterfactual_4sector_path_fast_identity.jld2)" BASELINE_ANCHOR_FILE="$(abspath ../input/baseline_4sector_path_reference.jld2)" TOL_DYNAMIC=1e-3 FAIL_ON_CHECKS=1 REQUIRE_T1_RESPONSE=1 CONFIG_TAG="$(ANTICIPATED_SHOCK_NAME)_fast_validate_debug"

report_immediate_reference: validate_immediate_reference
	$(MAKE) -C $(VALIDATE_DIR) report PROFILE=reference SHOCK_NAME="$(IMMEDIATE_SHOCK_NAME)"
	$(MAKE) -C $(VALIDATE_DIR) impact_report_focus PROFILE_TAG=reference SHOCK_INPUT_MODE=csv SHOCK_NAME="$(IMMEDIATE_SHOCK_NAME)" LAMBDA_CSV="$(abspath ../output/lambda_immediate.csv)" IDENTITY_OUTPUT_FILE="$(abspath ../input/counterfactual_4sector_path_reference_identity.jld2)" SHOCK_OUTPUT_FILE="$(abspath ../../cdp4_counterfactual_dynamics/output/counterfactual_4sector_path_reference_$(IMMEDIATE_SHOCK_NAME).jld2)" BASE_STATE_MAT="$(abspath ../input/Base_year_four_sectors.mat)" OUTPUT_DIR="$(VALIDATE_OUTPUT_DIR)" SHOCK_SECTOR="$(FOCUS_SECTOR)" SHOCK_REGION="$(FOCUS_REGION)"
	$(MAKE) -C $(VALIDATE_DIR) state_maps PROFILE_TAG=reference SHOCK_NAME="$(IMMEDIATE_SHOCK_NAME)" IDENTITY_OUTPUT_FILE="$(abspath ../input/counterfactual_4sector_path_reference_identity.jld2)" SHOCK_OUTPUT_FILE="$(abspath ../../cdp4_counterfactual_dynamics/output/counterfactual_4sector_path_reference_$(IMMEDIATE_SHOCK_NAME).jld2)" OUTPUT_DIR="$(VALIDATE_OUTPUT_DIR)"
	$(PYTHON) collect_outputs.py --scenario-name "$(IMMEDIATE_SHOCK_NAME)" --profile reference --dynamics-output-dir "$(DYNAMICS_OUTPUT_DIR)" --validate-output-dir "$(VALIDATE_OUTPUT_DIR)" --output-dir "$(TASK_OUTPUT_DIR)" --lambda-file "$(abspath ../output/lambda_immediate.csv)" --delta-selection-file "$(DELTA_SELECTION_FILE)" --solver-settings-file "$(SOLVER_SETTINGS_FILE)" --baseline-validation-file "$(abspath $(BASELINE_VALIDATION_LINK))"

report_anticipated_reference: validate_anticipated_reference
	$(MAKE) -C $(VALIDATE_DIR) report PROFILE=reference SHOCK_NAME="$(ANTICIPATED_SHOCK_NAME)"
	$(MAKE) -C $(VALIDATE_DIR) impact_report_focus PROFILE_TAG=reference SHOCK_INPUT_MODE=csv SHOCK_NAME="$(ANTICIPATED_SHOCK_NAME)" LAMBDA_CSV="$(abspath ../output/lambda_anticipated.csv)" IDENTITY_OUTPUT_FILE="$(abspath ../input/counterfactual_4sector_path_reference_identity.jld2)" SHOCK_OUTPUT_FILE="$(abspath ../../cdp4_counterfactual_dynamics/output/counterfactual_4sector_path_reference_$(ANTICIPATED_SHOCK_NAME).jld2)" BASE_STATE_MAT="$(abspath ../input/Base_year_four_sectors.mat)" OUTPUT_DIR="$(VALIDATE_OUTPUT_DIR)" SHOCK_SECTOR="$(FOCUS_SECTOR)" SHOCK_REGION="$(FOCUS_REGION)"
	$(MAKE) -C $(VALIDATE_DIR) state_maps PROFILE_TAG=reference SHOCK_NAME="$(ANTICIPATED_SHOCK_NAME)" IDENTITY_OUTPUT_FILE="$(abspath ../input/counterfactual_4sector_path_reference_identity.jld2)" SHOCK_OUTPUT_FILE="$(abspath ../../cdp4_counterfactual_dynamics/output/counterfactual_4sector_path_reference_$(ANTICIPATED_SHOCK_NAME).jld2)" OUTPUT_DIR="$(VALIDATE_OUTPUT_DIR)"
	$(PYTHON) collect_outputs.py --scenario-name "$(ANTICIPATED_SHOCK_NAME)" --profile reference --dynamics-output-dir "$(DYNAMICS_OUTPUT_DIR)" --validate-output-dir "$(VALIDATE_OUTPUT_DIR)" --output-dir "$(TASK_OUTPUT_DIR)" --lambda-file "$(abspath ../output/lambda_anticipated.csv)" --delta-selection-file "$(DELTA_SELECTION_FILE)" --solver-settings-file "$(SOLVER_SETTINGS_FILE)" --baseline-validation-file "$(abspath $(BASELINE_VALIDATION_LINK))"

report_immediate_fast: validate_immediate_fast
	$(MAKE) -C $(VALIDATE_DIR) report PROFILE=fast SHOCK_NAME="$(IMMEDIATE_SHOCK_NAME)"
	$(MAKE) -C $(VALIDATE_DIR) impact_report_focus PROFILE_TAG=fast SHOCK_INPUT_MODE=csv SHOCK_NAME="$(IMMEDIATE_SHOCK_NAME)" LAMBDA_CSV="$(abspath ../output/lambda_immediate.csv)" IDENTITY_OUTPUT_FILE="$(abspath ../input/counterfactual_4sector_path_fast_identity.jld2)" SHOCK_OUTPUT_FILE="$(abspath ../../cdp4_counterfactual_dynamics/output/counterfactual_4sector_path_fast_$(IMMEDIATE_SHOCK_NAME).jld2)" BASE_STATE_MAT="$(abspath ../input/Base_year_four_sectors.mat)" OUTPUT_DIR="$(VALIDATE_OUTPUT_DIR)" SHOCK_SECTOR="$(FOCUS_SECTOR)" SHOCK_REGION="$(FOCUS_REGION)"
	$(MAKE) -C $(VALIDATE_DIR) state_maps PROFILE_TAG=fast SHOCK_NAME="$(IMMEDIATE_SHOCK_NAME)" IDENTITY_OUTPUT_FILE="$(abspath ../input/counterfactual_4sector_path_fast_identity.jld2)" SHOCK_OUTPUT_FILE="$(abspath ../../cdp4_counterfactual_dynamics/output/counterfactual_4sector_path_fast_$(IMMEDIATE_SHOCK_NAME).jld2)" OUTPUT_DIR="$(VALIDATE_OUTPUT_DIR)"
	$(PYTHON) collect_outputs.py --scenario-name "$(IMMEDIATE_SHOCK_NAME)" --profile fast --dynamics-output-dir "$(DYNAMICS_OUTPUT_DIR)" --validate-output-dir "$(VALIDATE_OUTPUT_DIR)" --output-dir "$(TASK_OUTPUT_DIR)" --lambda-file "$(abspath ../output/lambda_immediate.csv)" --delta-selection-file "$(DELTA_SELECTION_FILE)" --solver-settings-file "$(SOLVER_SETTINGS_FILE)" --baseline-validation-file "$(abspath $(BASELINE_VALIDATION_LINK))"

report_anticipated_fast: validate_anticipated_fast
	$(MAKE) -C $(VALIDATE_DIR) report PROFILE=fast SHOCK_NAME="$(ANTICIPATED_SHOCK_NAME)"
	$(MAKE) -C $(VALIDATE_DIR) impact_report_focus PROFILE_TAG=fast SHOCK_INPUT_MODE=csv SHOCK_NAME="$(ANTICIPATED_SHOCK_NAME)" LAMBDA_CSV="$(abspath ../output/lambda_anticipated.csv)" IDENTITY_OUTPUT_FILE="$(abspath ../input/counterfactual_4sector_path_fast_identity.jld2)" SHOCK_OUTPUT_FILE="$(abspath ../../cdp4_counterfactual_dynamics/output/counterfactual_4sector_path_fast_$(ANTICIPATED_SHOCK_NAME).jld2)" BASE_STATE_MAT="$(abspath ../input/Base_year_four_sectors.mat)" OUTPUT_DIR="$(VALIDATE_OUTPUT_DIR)" SHOCK_SECTOR="$(FOCUS_SECTOR)" SHOCK_REGION="$(FOCUS_REGION)"
	$(MAKE) -C $(VALIDATE_DIR) state_maps PROFILE_TAG=fast SHOCK_NAME="$(ANTICIPATED_SHOCK_NAME)" IDENTITY_OUTPUT_FILE="$(abspath ../input/counterfactual_4sector_path_fast_identity.jld2)" SHOCK_OUTPUT_FILE="$(abspath ../../cdp4_counterfactual_dynamics/output/counterfactual_4sector_path_fast_$(ANTICIPATED_SHOCK_NAME).jld2)" OUTPUT_DIR="$(VALIDATE_OUTPUT_DIR)"
	$(PYTHON) collect_outputs.py --scenario-name "$(ANTICIPATED_SHOCK_NAME)" --profile fast --dynamics-output-dir "$(DYNAMICS_OUTPUT_DIR)" --validate-output-dir "$(VALIDATE_OUTPUT_DIR)" --output-dir "$(TASK_OUTPUT_DIR)" --lambda-file "$(abspath ../output/lambda_anticipated.csv)" --delta-selection-file "$(DELTA_SELECTION_FILE)" --solver-settings-file "$(SOLVER_SETTINGS_FILE)" --baseline-validation-file "$(abspath $(BASELINE_VALIDATION_LINK))"

clean_submission_outputs:
	rm -f ../output/*$(IMMEDIATE_SHOCK_NAME)* ../output/*$(ANTICIPATED_SHOCK_NAME)* ../output/baseline_validation_reference.csv "$(REFERENCE_BRIEF_FILE)"

collect_baseline_reference: $(BASELINE_VALIDATION_LINK)

$(DELTA_SELECTION_FILE) $(SOLVER_SETTINGS_FILE) ../output/lambda_immediate.csv ../output/lambda_anticipated.csv ../output/shock_calibration.json ../output/data_notes.md: select_reference_delta.py generate_lambda_csvs.py $(MATRIX) ../input/Base_year_four_sectors.mat ../input/baseline_4sector_path_reference.jld2 | ../output ../temp
	$(PYTHON) select_reference_delta.py --matrix "$(abspath $(MATRIX))" --output-dir "$(TASK_OUTPUT_DIR)" --temp-dir "$(TASK_TEMP_DIR)" --dynamics-dir "$(abspath $(DYNAMICS_DIR))" --dynamics-output-dir "$(DYNAMICS_OUTPUT_DIR)" --validate-dir "$(abspath $(VALIDATE_DIR))" --validate-output-dir "$(VALIDATE_OUTPUT_DIR)" --identity-output-file "$(abspath ../input/counterfactual_4sector_path_reference_identity.jld2)" --baseline-anchor-file "$(abspath ../input/baseline_4sector_path_reference.jld2)" --candidates "$(DELTA_CANDIDATES)" --time-horizon "$(TIME_HORIZON)"

$(BASELINE_VALIDATION_LINK): ../../cdp4_baseline_validate/output/validation_report_4sector_reference.csv | ../output
	ln -sf ../../cdp4_baseline_validate/output/validation_report_4sector_reference.csv $@

../../cognitive_intensity_data/output/cognitive_intensity_matrix.csv:
	$(MAKE) -C ../../cognitive_intensity_data/code compute

../../cdp4_baseline_dynamics/output/baseline_4sector_path_reference.jld2:
	$(MAKE) -C ../../cdp4_baseline_dynamics/code replication_reference

../../cdp4_baseline_validate/output/validation_report_4sector_reference.csv:
	$(MAKE) -C ../../cdp4_baseline_validate/code replication_reference

../../cdp4_counterfactual_dynamics/output/counterfactual_4sector_path_reference_identity.jld2:
	$(MAKE) -C ../../cdp4_counterfactual_dynamics/code identity_reference

../../cdp4_counterfactual_dynamics/output/counterfactual_4sector_path_fast_identity.jld2:
	$(MAKE) -C ../../cdp4_counterfactual_dynamics/code identity_fast

../input/Base_year_four_sectors.mat: ../../../CDP_codes_four_sectors/Base_year_four_sectors.mat | ../input
	ln -sf $< $@

../input/baseline_4sector_path_reference.jld2: ../../cdp4_baseline_dynamics/output/baseline_4sector_path_reference.jld2 | ../input
	ln -sf $< $@

../input/counterfactual_4sector_path_reference_identity.jld2: ../../cdp4_counterfactual_dynamics/output/counterfactual_4sector_path_reference_identity.jld2 | ../input
	ln -sf $< $@

../input/counterfactual_4sector_path_fast_identity.jld2: ../../cdp4_counterfactual_dynamics/output/counterfactual_4sector_path_fast_identity.jld2 | ../input
	ln -sf $< $@

clean:
	rm -f ../input/Base_year_four_sectors.mat
	rm -f ../input/baseline_4sector_path_reference.jld2
	rm -f ../input/counterfactual_4sector_path_reference_identity.jld2
	rm -f ../input/counterfactual_4sector_path_fast_identity.jld2
	rm -f ../output/*
	rm -rf ../temp/*

include ../../generic.make
