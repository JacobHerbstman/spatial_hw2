SHELL := bash
include ../../shell_functions.make
PYTHON ?= $(shell for p in /Users/jacobherbstman/miniconda3/bin/python3 python3 /Library/Frameworks/Python.framework/Versions/3.13/bin/python3; do if [ -x "$$p" ] && "$$p" -c "import json" >/dev/null 2>&1; then echo "$$p"; exit 0; fi; done; echo python3)

SCENARIO_NAME ?= cognitive_firstpass_d5_perm
DELTA ?= 0.05
TIME_HORIZON ?= 200
SCHEDULE ?= permanent
MATRIX ?= $(abspath ../../cognitive_intensity_data/output/cognitive_intensity_matrix.csv)
SHOCKS_OUTPUT_DIR := $(abspath ../../cdp4_cognitive_shocks/output)
DYNAMICS_OUTPUT_DIR := $(abspath ../../cdp4_counterfactual_dynamics/output)
VALIDATE_OUTPUT_DIR := $(abspath ../../cdp4_counterfactual_validate/output)

.PHONY: all shocks run_fast validate_fast report_fast collect_fast run_reference validate_reference report_reference collect_reference compare_fast_reference clean prepare_inputs

all: report_fast

shocks: ../input/lambda_hat_$(SCENARIO_NAME).csv ../input/lambda_snapshot_$(SCENARIO_NAME).csv ../input/shock_manifest_$(SCENARIO_NAME).json

prepare_inputs: ../input/Base_year_four_sectors.mat ../input/baseline_4sector_path_reference.jld2 ../input/counterfactual_4sector_path_fast_identity.jld2 | ../output

run_fast: shocks prepare_inputs
	$(MAKE) -C ../../cdp4_counterfactual_dynamics/code csv_fast LAMBDA_CSV="$(abspath ../input/lambda_hat_$(SCENARIO_NAME).csv)" SHOCK_NAME="$(SCENARIO_NAME)" TIME_HORIZON="$(TIME_HORIZON)" BASELINE_ANCHOR_FILE="$(abspath ../input/baseline_4sector_path_reference.jld2)" CONFIG_TAG="$(SCENARIO_NAME)_fast"

validate_fast: run_fast ../input/counterfactual_4sector_path_fast_identity.jld2
	$(MAKE) -C ../../cdp4_counterfactual_validate/code generic_validate PROFILE=fast SHOCK_INPUT_MODE=csv SHOCK_NAME="$(SCENARIO_NAME)" LAMBDA_CSV="$(abspath ../input/lambda_hat_$(SCENARIO_NAME).csv)" TIME_HORIZON="$(TIME_HORIZON)" COUNTERFACTUAL_OUTPUT_FILE="$(abspath ../../cdp4_counterfactual_dynamics/output/counterfactual_4sector_path_fast_$(SCENARIO_NAME).jld2)" IDENTITY_OUTPUT_FILE="$(abspath ../input/counterfactual_4sector_path_fast_identity.jld2)" BASELINE_ANCHOR_FILE="$(abspath ../input/baseline_4sector_path_reference.jld2)" CONFIG_TAG="$(SCENARIO_NAME)_validate_fast"

report_fast: validate_fast
	$(MAKE) -C ../../cdp4_counterfactual_validate/code report PROFILE=fast SHOCK_NAME="$(SCENARIO_NAME)"
	$(MAKE) -C ../../cdp4_counterfactual_validate/code impact_report_generic PROFILE_TAG=fast SHOCK_NAME="$(SCENARIO_NAME)" IDENTITY_OUTPUT_FILE="$(abspath ../input/counterfactual_4sector_path_fast_identity.jld2)" SHOCK_OUTPUT_FILE="$(abspath ../../cdp4_counterfactual_dynamics/output/counterfactual_4sector_path_fast_$(SCENARIO_NAME).jld2)" OUTPUT_DIR="$(VALIDATE_OUTPUT_DIR)"
	$(MAKE) -C ../../cdp4_counterfactual_validate/code state_maps PROFILE_TAG=fast SHOCK_NAME="$(SCENARIO_NAME)" IDENTITY_OUTPUT_FILE="$(abspath ../input/counterfactual_4sector_path_fast_identity.jld2)" SHOCK_OUTPUT_FILE="$(abspath ../../cdp4_counterfactual_dynamics/output/counterfactual_4sector_path_fast_$(SCENARIO_NAME).jld2)" OUTPUT_DIR="$(VALIDATE_OUTPUT_DIR)"
	$(MAKE) collect_fast

collect_fast:
	$(PYTHON) collect_outputs.py --scenario-name "$(SCENARIO_NAME)" --profile fast --shock-output-dir "$(SHOCKS_OUTPUT_DIR)" --dynamics-output-dir "$(DYNAMICS_OUTPUT_DIR)" --validate-output-dir "$(VALIDATE_OUTPUT_DIR)" --output-dir ../output

run_reference: shocks prepare_inputs ../input/counterfactual_4sector_path_reference_identity.jld2
	$(MAKE) -C ../../cdp4_counterfactual_dynamics/code csv_reference LAMBDA_CSV="$(abspath ../input/lambda_hat_$(SCENARIO_NAME).csv)" SHOCK_NAME="$(SCENARIO_NAME)" TIME_HORIZON="$(TIME_HORIZON)" BASELINE_ANCHOR_FILE="$(abspath ../input/baseline_4sector_path_reference.jld2)" CONFIG_TAG="$(SCENARIO_NAME)_reference"

validate_reference: run_reference ../input/counterfactual_4sector_path_reference_identity.jld2
	$(MAKE) -C ../../cdp4_counterfactual_validate/code generic_validate PROFILE=reference SHOCK_INPUT_MODE=csv SHOCK_NAME="$(SCENARIO_NAME)" LAMBDA_CSV="$(abspath ../input/lambda_hat_$(SCENARIO_NAME).csv)" TIME_HORIZON="$(TIME_HORIZON)" COUNTERFACTUAL_OUTPUT_FILE="$(abspath ../../cdp4_counterfactual_dynamics/output/counterfactual_4sector_path_reference_$(SCENARIO_NAME).jld2)" IDENTITY_OUTPUT_FILE="$(abspath ../input/counterfactual_4sector_path_reference_identity.jld2)" BASELINE_ANCHOR_FILE="$(abspath ../input/baseline_4sector_path_reference.jld2)" USE_THREADS=0 THREADS_DYNAMIC=0 THREADS_STATIC=0 CONFIG_TAG="$(SCENARIO_NAME)_validate_reference"

report_reference: validate_reference
	$(MAKE) -C ../../cdp4_counterfactual_validate/code report PROFILE=reference SHOCK_NAME="$(SCENARIO_NAME)"
	$(MAKE) -C ../../cdp4_counterfactual_validate/code impact_report_generic PROFILE_TAG=reference SHOCK_NAME="$(SCENARIO_NAME)" IDENTITY_OUTPUT_FILE="$(abspath ../input/counterfactual_4sector_path_reference_identity.jld2)" SHOCK_OUTPUT_FILE="$(abspath ../../cdp4_counterfactual_dynamics/output/counterfactual_4sector_path_reference_$(SCENARIO_NAME).jld2)" OUTPUT_DIR="$(VALIDATE_OUTPUT_DIR)"
	$(MAKE) -C ../../cdp4_counterfactual_validate/code state_maps PROFILE_TAG=reference SHOCK_NAME="$(SCENARIO_NAME)" IDENTITY_OUTPUT_FILE="$(abspath ../input/counterfactual_4sector_path_reference_identity.jld2)" SHOCK_OUTPUT_FILE="$(abspath ../../cdp4_counterfactual_dynamics/output/counterfactual_4sector_path_reference_$(SCENARIO_NAME).jld2)" OUTPUT_DIR="$(VALIDATE_OUTPUT_DIR)"
	$(MAKE) collect_reference

collect_reference:
	$(PYTHON) collect_outputs.py --scenario-name "$(SCENARIO_NAME)" --profile reference --shock-output-dir "$(SHOCKS_OUTPUT_DIR)" --dynamics-output-dir "$(DYNAMICS_OUTPUT_DIR)" --validate-output-dir "$(VALIDATE_OUTPUT_DIR)" --output-dir ../output

compare_fast_reference: run_fast run_reference ../input/counterfactual_4sector_path_fast_identity.jld2
	$(MAKE) -C ../../cdp4_counterfactual_validate/code generic_validate PROFILE=fast SHOCK_INPUT_MODE=csv SHOCK_NAME="$(SCENARIO_NAME)" LAMBDA_CSV="$(abspath ../input/lambda_hat_$(SCENARIO_NAME).csv)" TIME_HORIZON="$(TIME_HORIZON)" COUNTERFACTUAL_OUTPUT_FILE="$(abspath ../../cdp4_counterfactual_dynamics/output/counterfactual_4sector_path_fast_$(SCENARIO_NAME).jld2)" IDENTITY_OUTPUT_FILE="$(abspath ../input/counterfactual_4sector_path_fast_identity.jld2)" REFERENCE_OUTPUT_FILE="$(abspath ../../cdp4_counterfactual_dynamics/output/counterfactual_4sector_path_reference_$(SCENARIO_NAME).jld2)" BASELINE_ANCHOR_FILE="$(abspath ../input/baseline_4sector_path_reference.jld2)" CONFIG_TAG="$(SCENARIO_NAME)_compare_fast_reference"

../../cdp4_baseline_dynamics/output/baseline_4sector_path_reference.jld2:
	$(MAKE) -C ../../cdp4_baseline_dynamics/code replication_reference

../../cdp4_counterfactual_dynamics/output/counterfactual_4sector_path_fast_identity.jld2:
	$(MAKE) -C ../../cdp4_counterfactual_dynamics/code identity_fast

../../cdp4_counterfactual_dynamics/output/counterfactual_4sector_path_reference_identity.jld2:
	$(MAKE) -C ../../cdp4_counterfactual_dynamics/code identity_reference

../input/lambda_hat_$(SCENARIO_NAME).csv ../input/lambda_snapshot_$(SCENARIO_NAME).csv ../input/shock_manifest_$(SCENARIO_NAME).json: | ../input
	$(MAKE) -C ../../cdp4_cognitive_shocks/code build
	ln -sf ../../cdp4_cognitive_shocks/output/lambda_hat_$(SCENARIO_NAME).csv ../input/lambda_hat_$(SCENARIO_NAME).csv
	ln -sf ../../cdp4_cognitive_shocks/output/lambda_snapshot_$(SCENARIO_NAME).csv ../input/lambda_snapshot_$(SCENARIO_NAME).csv
	ln -sf ../../cdp4_cognitive_shocks/output/shock_manifest_$(SCENARIO_NAME).json ../input/shock_manifest_$(SCENARIO_NAME).json

../input/Base_year_four_sectors.mat: ../../../CDP_codes_four_sectors/Base_year_four_sectors.mat | ../input
	ln -sf $< $@

../input/baseline_4sector_path_reference.jld2: ../../cdp4_baseline_dynamics/output/baseline_4sector_path_reference.jld2 | ../input
	ln -sf $< $@

../input/counterfactual_4sector_path_fast_identity.jld2: ../../cdp4_counterfactual_dynamics/output/counterfactual_4sector_path_fast_identity.jld2 | ../input
	ln -sf $< $@

../input/counterfactual_4sector_path_reference_identity.jld2: ../../cdp4_counterfactual_dynamics/output/counterfactual_4sector_path_reference_identity.jld2 | ../input
	ln -sf $< $@

clean:
	rm -f ../input/lambda_hat_*.csv ../input/lambda_snapshot_*.csv ../input/shock_manifest_*.json
	rm -f ../output/*

include ../../generic.make
