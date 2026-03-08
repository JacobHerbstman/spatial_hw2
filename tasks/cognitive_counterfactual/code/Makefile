SHELL := bash
include ../../shell_functions.make
PYTHON ?= $(shell for p in /Users/jacobherbstman/miniconda3/bin/python3 python3 /Library/Frameworks/Python.framework/Versions/3.13/bin/python3; do if [ -x "$$p" ] && "$$p" -c "import pandas,json" >/dev/null 2>&1; then echo "$$p"; exit 0; fi; done; echo python3)

MATRIX ?= ../../cognitive_intensity_data/output/cognitive_intensity_matrix.csv
DELTA ?= 0.01
TIME_HORIZON ?= 200
FOCUS_SECTOR ?= 4
FOCUS_REGION ?= 32

IMMEDIATE_SHOCK_NAME := cognitive_immediate
ANTICIPATED_SHOCK_NAME := cognitive_anticipated

DYNAMICS_DIR := ../../cdp4_counterfactual_dynamics/code
VALIDATE_DIR := ../../cdp4_counterfactual_validate/code
DYNAMICS_OUTPUT_DIR := $(abspath ../../cdp4_counterfactual_dynamics/output)
VALIDATE_OUTPUT_DIR := $(abspath ../../cdp4_counterfactual_validate/output)
TASK_OUTPUT_DIR := $(abspath ../output)

.PHONY: all generate_shocks immediate_fast immediate_reference anticipated_fast anticipated_reference validate_immediate_fast validate_anticipated_fast validate_immediate_reference validate_anticipated_reference report_immediate_fast report_anticipated_fast report_immediate_reference report_anticipated_reference maps_immediate_fast maps_anticipated_fast maps_immediate_reference maps_anticipated_reference run_fast run_reference validate_fast validate_reference report_fast report_reference all_fast all_reference collect_immediate_fast collect_anticipated_fast collect_immediate_reference collect_anticipated_reference clean prepare_inputs

all: generate_shocks

generate_shocks: ../output/lambda_immediate.csv ../output/lambda_anticipated.csv ../output/shock_calibration.json ../output/data_notes.md

prepare_inputs: ../input/Base_year_four_sectors.mat ../input/baseline_4sector_path_reference.jld2 ../input/counterfactual_4sector_path_fast_identity.jld2 ../input/counterfactual_4sector_path_reference_identity.jld2 | ../output

run_fast: immediate_fast anticipated_fast

run_reference: immediate_reference anticipated_reference

validate_fast: validate_immediate_fast validate_anticipated_fast

validate_reference: validate_immediate_reference validate_anticipated_reference

report_fast: report_immediate_fast report_anticipated_fast

report_reference: report_immediate_reference report_anticipated_reference

all_fast: report_fast

all_reference: report_reference

immediate_fast: generate_shocks prepare_inputs
	WARM_START_STATIC=1 USE_ANDERSON=0 HVECT_RELAX=0.05 $(MAKE) -C $(DYNAMICS_DIR) csv_fast LAMBDA_CSV="$(abspath ../output/lambda_immediate.csv)" SHOCK_NAME="$(IMMEDIATE_SHOCK_NAME)" TIME_HORIZON="$(TIME_HORIZON)" BASELINE_ANCHOR_FILE="$(abspath ../input/baseline_4sector_path_reference.jld2)" CONFIG_TAG="$(IMMEDIATE_SHOCK_NAME)_fast"

anticipated_fast: generate_shocks prepare_inputs
	WARM_START_STATIC=1 USE_ANDERSON=0 HVECT_RELAX=0.05 $(MAKE) -C $(DYNAMICS_DIR) csv_fast LAMBDA_CSV="$(abspath ../output/lambda_anticipated.csv)" SHOCK_NAME="$(ANTICIPATED_SHOCK_NAME)" TIME_HORIZON="$(TIME_HORIZON)" BASELINE_ANCHOR_FILE="$(abspath ../input/baseline_4sector_path_reference.jld2)" CONFIG_TAG="$(ANTICIPATED_SHOCK_NAME)_fast"

immediate_reference: generate_shocks prepare_inputs
	HVECT_RELAX=0.05 $(MAKE) -C $(DYNAMICS_DIR) csv_reference LAMBDA_CSV="$(abspath ../output/lambda_immediate.csv)" SHOCK_NAME="$(IMMEDIATE_SHOCK_NAME)" TIME_HORIZON="$(TIME_HORIZON)" BASELINE_ANCHOR_FILE="$(abspath ../input/baseline_4sector_path_reference.jld2)" CONFIG_TAG="$(IMMEDIATE_SHOCK_NAME)_reference"

anticipated_reference: generate_shocks prepare_inputs
	HVECT_RELAX=0.05 $(MAKE) -C $(DYNAMICS_DIR) csv_reference LAMBDA_CSV="$(abspath ../output/lambda_anticipated.csv)" SHOCK_NAME="$(ANTICIPATED_SHOCK_NAME)" TIME_HORIZON="$(TIME_HORIZON)" BASELINE_ANCHOR_FILE="$(abspath ../input/baseline_4sector_path_reference.jld2)" CONFIG_TAG="$(ANTICIPATED_SHOCK_NAME)_reference"

validate_immediate_fast: immediate_fast
	WARM_START_STATIC=1 USE_ANDERSON=0 $(MAKE) -C $(VALIDATE_DIR) generic_validate PROFILE=fast SHOCK_INPUT_MODE=csv SHOCK_NAME="$(IMMEDIATE_SHOCK_NAME)" LAMBDA_CSV="$(abspath ../output/lambda_immediate.csv)" TIME_HORIZON="$(TIME_HORIZON)" COUNTERFACTUAL_OUTPUT_FILE="$(abspath ../../cdp4_counterfactual_dynamics/output/counterfactual_4sector_path_fast_$(IMMEDIATE_SHOCK_NAME).jld2)" IDENTITY_OUTPUT_FILE="$(abspath ../input/counterfactual_4sector_path_fast_identity.jld2)" BASELINE_ANCHOR_FILE="$(abspath ../input/baseline_4sector_path_reference.jld2)" CONFIG_TAG="$(IMMEDIATE_SHOCK_NAME)_validate_fast"

validate_anticipated_fast: anticipated_fast
	WARM_START_STATIC=1 USE_ANDERSON=0 $(MAKE) -C $(VALIDATE_DIR) generic_validate PROFILE=fast SHOCK_INPUT_MODE=csv SHOCK_NAME="$(ANTICIPATED_SHOCK_NAME)" LAMBDA_CSV="$(abspath ../output/lambda_anticipated.csv)" TIME_HORIZON="$(TIME_HORIZON)" COUNTERFACTUAL_OUTPUT_FILE="$(abspath ../../cdp4_counterfactual_dynamics/output/counterfactual_4sector_path_fast_$(ANTICIPATED_SHOCK_NAME).jld2)" IDENTITY_OUTPUT_FILE="$(abspath ../input/counterfactual_4sector_path_fast_identity.jld2)" BASELINE_ANCHOR_FILE="$(abspath ../input/baseline_4sector_path_reference.jld2)" CONFIG_TAG="$(ANTICIPATED_SHOCK_NAME)_validate_fast"

validate_immediate_reference: immediate_reference
	$(MAKE) -C $(VALIDATE_DIR) generic_validate PROFILE=reference SHOCK_INPUT_MODE=csv SHOCK_NAME="$(IMMEDIATE_SHOCK_NAME)" LAMBDA_CSV="$(abspath ../output/lambda_immediate.csv)" TIME_HORIZON="$(TIME_HORIZON)" COUNTERFACTUAL_OUTPUT_FILE="$(abspath ../../cdp4_counterfactual_dynamics/output/counterfactual_4sector_path_reference_$(IMMEDIATE_SHOCK_NAME).jld2)" IDENTITY_OUTPUT_FILE="$(abspath ../input/counterfactual_4sector_path_reference_identity.jld2)" BASELINE_ANCHOR_FILE="$(abspath ../input/baseline_4sector_path_reference.jld2)" USE_THREADS=0 THREADS_DYNAMIC=0 THREADS_STATIC=0 CONFIG_TAG="$(IMMEDIATE_SHOCK_NAME)_validate_reference"

validate_anticipated_reference: anticipated_reference
	$(MAKE) -C $(VALIDATE_DIR) generic_validate PROFILE=reference SHOCK_INPUT_MODE=csv SHOCK_NAME="$(ANTICIPATED_SHOCK_NAME)" LAMBDA_CSV="$(abspath ../output/lambda_anticipated.csv)" TIME_HORIZON="$(TIME_HORIZON)" COUNTERFACTUAL_OUTPUT_FILE="$(abspath ../../cdp4_counterfactual_dynamics/output/counterfactual_4sector_path_reference_$(ANTICIPATED_SHOCK_NAME).jld2)" IDENTITY_OUTPUT_FILE="$(abspath ../input/counterfactual_4sector_path_reference_identity.jld2)" BASELINE_ANCHOR_FILE="$(abspath ../input/baseline_4sector_path_reference.jld2)" USE_THREADS=0 THREADS_DYNAMIC=0 THREADS_STATIC=0 CONFIG_TAG="$(ANTICIPATED_SHOCK_NAME)_validate_reference"

report_immediate_fast: validate_immediate_fast
	$(MAKE) -C $(VALIDATE_DIR) report PROFILE=fast SHOCK_NAME="$(IMMEDIATE_SHOCK_NAME)"
	$(MAKE) -C $(VALIDATE_DIR) impact_report_focus PROFILE_TAG=fast SHOCK_INPUT_MODE=csv SHOCK_NAME="$(IMMEDIATE_SHOCK_NAME)" LAMBDA_CSV="$(abspath ../output/lambda_immediate.csv)" IDENTITY_OUTPUT_FILE="$(abspath ../input/counterfactual_4sector_path_fast_identity.jld2)" SHOCK_OUTPUT_FILE="$(abspath ../../cdp4_counterfactual_dynamics/output/counterfactual_4sector_path_fast_$(IMMEDIATE_SHOCK_NAME).jld2)" BASE_STATE_MAT="$(abspath ../input/Base_year_four_sectors.mat)" OUTPUT_DIR="$(VALIDATE_OUTPUT_DIR)" SHOCK_SECTOR="$(FOCUS_SECTOR)" SHOCK_REGION="$(FOCUS_REGION)"
	$(MAKE) -C $(VALIDATE_DIR) state_maps PROFILE_TAG=fast SHOCK_NAME="$(IMMEDIATE_SHOCK_NAME)" IDENTITY_OUTPUT_FILE="$(abspath ../input/counterfactual_4sector_path_fast_identity.jld2)" SHOCK_OUTPUT_FILE="$(abspath ../../cdp4_counterfactual_dynamics/output/counterfactual_4sector_path_fast_$(IMMEDIATE_SHOCK_NAME).jld2)" OUTPUT_DIR="$(VALIDATE_OUTPUT_DIR)"
	$(MAKE) collect_immediate_fast

report_anticipated_fast: validate_anticipated_fast
	$(MAKE) -C $(VALIDATE_DIR) report PROFILE=fast SHOCK_NAME="$(ANTICIPATED_SHOCK_NAME)"
	$(MAKE) -C $(VALIDATE_DIR) impact_report_focus PROFILE_TAG=fast SHOCK_INPUT_MODE=csv SHOCK_NAME="$(ANTICIPATED_SHOCK_NAME)" LAMBDA_CSV="$(abspath ../output/lambda_anticipated.csv)" IDENTITY_OUTPUT_FILE="$(abspath ../input/counterfactual_4sector_path_fast_identity.jld2)" SHOCK_OUTPUT_FILE="$(abspath ../../cdp4_counterfactual_dynamics/output/counterfactual_4sector_path_fast_$(ANTICIPATED_SHOCK_NAME).jld2)" BASE_STATE_MAT="$(abspath ../input/Base_year_four_sectors.mat)" OUTPUT_DIR="$(VALIDATE_OUTPUT_DIR)" SHOCK_SECTOR="$(FOCUS_SECTOR)" SHOCK_REGION="$(FOCUS_REGION)"
	$(MAKE) -C $(VALIDATE_DIR) state_maps PROFILE_TAG=fast SHOCK_NAME="$(ANTICIPATED_SHOCK_NAME)" IDENTITY_OUTPUT_FILE="$(abspath ../input/counterfactual_4sector_path_fast_identity.jld2)" SHOCK_OUTPUT_FILE="$(abspath ../../cdp4_counterfactual_dynamics/output/counterfactual_4sector_path_fast_$(ANTICIPATED_SHOCK_NAME).jld2)" OUTPUT_DIR="$(VALIDATE_OUTPUT_DIR)"
	$(MAKE) collect_anticipated_fast

report_immediate_reference: validate_immediate_reference
	$(MAKE) -C $(VALIDATE_DIR) report PROFILE=reference SHOCK_NAME="$(IMMEDIATE_SHOCK_NAME)"
	$(MAKE) -C $(VALIDATE_DIR) impact_report_focus PROFILE_TAG=reference SHOCK_INPUT_MODE=csv SHOCK_NAME="$(IMMEDIATE_SHOCK_NAME)" LAMBDA_CSV="$(abspath ../output/lambda_immediate.csv)" IDENTITY_OUTPUT_FILE="$(abspath ../input/counterfactual_4sector_path_reference_identity.jld2)" SHOCK_OUTPUT_FILE="$(abspath ../../cdp4_counterfactual_dynamics/output/counterfactual_4sector_path_reference_$(IMMEDIATE_SHOCK_NAME).jld2)" BASE_STATE_MAT="$(abspath ../input/Base_year_four_sectors.mat)" OUTPUT_DIR="$(VALIDATE_OUTPUT_DIR)" SHOCK_SECTOR="$(FOCUS_SECTOR)" SHOCK_REGION="$(FOCUS_REGION)"
	$(MAKE) -C $(VALIDATE_DIR) state_maps PROFILE_TAG=reference SHOCK_NAME="$(IMMEDIATE_SHOCK_NAME)" IDENTITY_OUTPUT_FILE="$(abspath ../input/counterfactual_4sector_path_reference_identity.jld2)" SHOCK_OUTPUT_FILE="$(abspath ../../cdp4_counterfactual_dynamics/output/counterfactual_4sector_path_reference_$(IMMEDIATE_SHOCK_NAME).jld2)" OUTPUT_DIR="$(VALIDATE_OUTPUT_DIR)"
	$(MAKE) collect_immediate_reference

report_anticipated_reference: validate_anticipated_reference
	$(MAKE) -C $(VALIDATE_DIR) report PROFILE=reference SHOCK_NAME="$(ANTICIPATED_SHOCK_NAME)"
	$(MAKE) -C $(VALIDATE_DIR) impact_report_focus PROFILE_TAG=reference SHOCK_INPUT_MODE=csv SHOCK_NAME="$(ANTICIPATED_SHOCK_NAME)" LAMBDA_CSV="$(abspath ../output/lambda_anticipated.csv)" IDENTITY_OUTPUT_FILE="$(abspath ../input/counterfactual_4sector_path_reference_identity.jld2)" SHOCK_OUTPUT_FILE="$(abspath ../../cdp4_counterfactual_dynamics/output/counterfactual_4sector_path_reference_$(ANTICIPATED_SHOCK_NAME).jld2)" BASE_STATE_MAT="$(abspath ../input/Base_year_four_sectors.mat)" OUTPUT_DIR="$(VALIDATE_OUTPUT_DIR)" SHOCK_SECTOR="$(FOCUS_SECTOR)" SHOCK_REGION="$(FOCUS_REGION)"
	$(MAKE) -C $(VALIDATE_DIR) state_maps PROFILE_TAG=reference SHOCK_NAME="$(ANTICIPATED_SHOCK_NAME)" IDENTITY_OUTPUT_FILE="$(abspath ../input/counterfactual_4sector_path_reference_identity.jld2)" SHOCK_OUTPUT_FILE="$(abspath ../../cdp4_counterfactual_dynamics/output/counterfactual_4sector_path_reference_$(ANTICIPATED_SHOCK_NAME).jld2)" OUTPUT_DIR="$(VALIDATE_OUTPUT_DIR)"
	$(MAKE) collect_anticipated_reference

maps_immediate_fast: report_immediate_fast

maps_anticipated_fast: report_anticipated_fast

maps_immediate_reference: report_immediate_reference

maps_anticipated_reference: report_anticipated_reference

collect_immediate_fast:
	$(PYTHON) collect_outputs.py --scenario-name "$(IMMEDIATE_SHOCK_NAME)" --profile fast --dynamics-output-dir "$(DYNAMICS_OUTPUT_DIR)" --validate-output-dir "$(VALIDATE_OUTPUT_DIR)" --output-dir "$(TASK_OUTPUT_DIR)"

collect_anticipated_fast:
	$(PYTHON) collect_outputs.py --scenario-name "$(ANTICIPATED_SHOCK_NAME)" --profile fast --dynamics-output-dir "$(DYNAMICS_OUTPUT_DIR)" --validate-output-dir "$(VALIDATE_OUTPUT_DIR)" --output-dir "$(TASK_OUTPUT_DIR)"

collect_immediate_reference:
	$(PYTHON) collect_outputs.py --scenario-name "$(IMMEDIATE_SHOCK_NAME)" --profile reference --dynamics-output-dir "$(DYNAMICS_OUTPUT_DIR)" --validate-output-dir "$(VALIDATE_OUTPUT_DIR)" --output-dir "$(TASK_OUTPUT_DIR)"

collect_anticipated_reference:
	$(PYTHON) collect_outputs.py --scenario-name "$(ANTICIPATED_SHOCK_NAME)" --profile reference --dynamics-output-dir "$(DYNAMICS_OUTPUT_DIR)" --validate-output-dir "$(VALIDATE_OUTPUT_DIR)" --output-dir "$(TASK_OUTPUT_DIR)"

../output/lambda_immediate.csv ../output/lambda_anticipated.csv ../output/shock_calibration.json ../output/data_notes.md: generate_lambda_csvs.py $(MATRIX) | ../output
	$(PYTHON) generate_lambda_csvs.py --matrix "$(abspath $(MATRIX))" --output-dir "$(abspath ../output)" --delta "$(DELTA)" --time-horizon "$(TIME_HORIZON)"

../../cdp4_baseline_dynamics/output/baseline_4sector_path_reference.jld2:
	$(MAKE) -C ../../cdp4_baseline_dynamics/code replication_reference

../../cdp4_counterfactual_dynamics/output/counterfactual_4sector_path_fast_identity.jld2:
	$(MAKE) -C ../../cdp4_counterfactual_dynamics/code identity_fast

../../cdp4_counterfactual_dynamics/output/counterfactual_4sector_path_reference_identity.jld2:
	$(MAKE) -C ../../cdp4_counterfactual_dynamics/code identity_reference

../input/Base_year_four_sectors.mat: ../../../CDP_codes_four_sectors/Base_year_four_sectors.mat | ../input
	ln -sf $< $@

../input/baseline_4sector_path_reference.jld2: ../../cdp4_baseline_dynamics/output/baseline_4sector_path_reference.jld2 | ../input
	ln -sf $< $@

../input/counterfactual_4sector_path_fast_identity.jld2: ../../cdp4_counterfactual_dynamics/output/counterfactual_4sector_path_fast_identity.jld2 | ../input
	ln -sf $< $@

../input/counterfactual_4sector_path_reference_identity.jld2: ../../cdp4_counterfactual_dynamics/output/counterfactual_4sector_path_reference_identity.jld2 | ../input
	ln -sf $< $@

clean:
	rm -f ../input/Base_year_four_sectors.mat
	rm -f ../input/baseline_4sector_path_reference.jld2
	rm -f ../input/counterfactual_4sector_path_fast_identity.jld2
	rm -f ../input/counterfactual_4sector_path_reference_identity.jld2
	rm -f ../output/*

include ../../generic.make
