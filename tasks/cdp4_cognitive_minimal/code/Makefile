SHELL := bash
include ../../shell_functions.make

BASE_YEAR_MAT ?= ../input/Base_year_four_sectors.mat
HNO_SHOCK_MAT ?= ../input/Hvectnoshock.mat
COGNITIVE_MATRIX ?= ../input/cognitive_intensity_matrix.csv
BASELINE_VALIDATION ?= ../input/validation_report_4sector_reference.csv
DELTA ?= 0.05
TIME_HORIZON ?= 200
PROFILE ?= reference
TOL_DYNAMIC ?= 1e-3
MAX_ITER_DYNAMIC ?= 1000
MAX_ITER_STATIC ?= 2000
USE_THREADS ?= 0
THREADS_STATIC ?= 0
THREADS_DYNAMIC ?= 0
RECORD_TRACE ?= 1

.PHONY: all prepare_inputs build_shock identity_run identity_check tiny_run tiny_smoke immediate_run immediate_summary

all: build_shock identity_check immediate_run immediate_summary

prepare_inputs: $(BASE_YEAR_MAT) $(HNO_SHOCK_MAT) $(COGNITIVE_MATRIX) $(BASELINE_VALIDATION) | ../output

build_shock: prepare_inputs
	python3 build_immediate_shock.py \
		--matrix "$(COGNITIVE_MATRIX)" \
		--delta "$(DELTA)" \
		--time-horizon "$(TIME_HORIZON)" \
		--output-dir ../output

identity_run: build_shock
	@source ../../shell_functions.sh; PROFILE=$(PROFILE) BASE_YEAR_MAT="$(BASE_YEAR_MAT)" HNO_SHOCK_MAT="$(HNO_SHOCK_MAT)" LAMBDA_MAT="$(abspath ../output/lambda_identity.mat)" RUN_NAME=identity TIME_HORIZON=$(TIME_HORIZON) TOL_DYNAMIC=$(TOL_DYNAMIC) MAX_ITER_DYNAMIC=$(MAX_ITER_DYNAMIC) MAX_ITER_STATIC=$(MAX_ITER_STATIC) USE_THREADS=$(USE_THREADS) THREADS_DYNAMIC=$(THREADS_DYNAMIC) THREADS_STATIC=$(THREADS_STATIC) RECORD_TRACE=$(RECORD_TRACE) Julia_pc_and_local --project=../../_lib/cdp_julia run_cognitive_minimal.jl

identity_check: identity_run
	python3 summarize_results.py identity-check \
		--baseline-validation "$(BASELINE_VALIDATION)" \
		--summary ../output/summary_identity.csv \
		--ynew ../output/ynew_matrix_identity.csv \
		--hno-shock "$(HNO_SHOCK_MAT)" \
		--selected-sector ../output/selected_state_sector_identity.csv \
		--output-report ../output/identity_check_report.csv \
		--output-note ../output/identity_check.md

tiny_run: identity_check
	@source ../../shell_functions.sh; PROFILE=$(PROFILE) BASE_YEAR_MAT="$(BASE_YEAR_MAT)" HNO_SHOCK_MAT="$(HNO_SHOCK_MAT)" LAMBDA_MAT="$(abspath ../output/lambda_tiny_ny_services.mat)" RUN_NAME=tiny_ny_services TIME_HORIZON=$(TIME_HORIZON) TOL_DYNAMIC=$(TOL_DYNAMIC) MAX_ITER_DYNAMIC=$(MAX_ITER_DYNAMIC) MAX_ITER_STATIC=$(MAX_ITER_STATIC) USE_THREADS=$(USE_THREADS) THREADS_DYNAMIC=$(THREADS_DYNAMIC) THREADS_STATIC=$(THREADS_STATIC) RECORD_TRACE=$(RECORD_TRACE) Julia_pc_and_local --project=../../_lib/cdp_julia run_cognitive_minimal.jl

tiny_smoke: tiny_run
	python3 summarize_results.py tiny-smoke \
		--summary ../output/summary_tiny_ny_services.csv \
		--identity-sector ../output/selected_state_sector_identity.csv \
		--shock-sector ../output/selected_state_sector_tiny_ny_services.csv \
		--output-report ../output/tiny_smoke_report.csv \
		--output-note ../output/tiny_smoke.md \
		--focus-state 32 \
		--focus-sector 4 \
		--focus-time 2

immediate_run: identity_check
	@source ../../shell_functions.sh; PROFILE=$(PROFILE) BASE_YEAR_MAT="$(BASE_YEAR_MAT)" HNO_SHOCK_MAT="$(HNO_SHOCK_MAT)" LAMBDA_MAT="$(abspath ../output/lambda_cognitive_immediate.mat)" RUN_NAME=cognitive_immediate TIME_HORIZON=$(TIME_HORIZON) TOL_DYNAMIC=$(TOL_DYNAMIC) MAX_ITER_DYNAMIC=$(MAX_ITER_DYNAMIC) MAX_ITER_STATIC=$(MAX_ITER_STATIC) USE_THREADS=$(USE_THREADS) THREADS_DYNAMIC=$(THREADS_DYNAMIC) THREADS_STATIC=$(THREADS_STATIC) RECORD_TRACE=$(RECORD_TRACE) Julia_pc_and_local --project=../../_lib/cdp_julia run_cognitive_minimal.jl

immediate_summary:
	python3 summarize_results.py immediate-summary \
		--summary ../output/summary_cognitive_immediate.csv \
		--identity-sector ../output/selected_state_sector_identity.csv \
		--identity-aggregate ../output/selected_state_aggregate_identity.csv \
		--shock-sector ../output/selected_state_sector_cognitive_immediate.csv \
		--shock-aggregate ../output/selected_state_aggregate_cognitive_immediate.csv \
		--output-dir ../output

../input/Base_year_four_sectors.mat: ../../../CDP_codes_four_sectors/Base_year_four_sectors.mat | ../input
	ln -sf $< $@

../input/Hvectnoshock.mat: ../../../CDP_codes_four_sectors/Hvectnoshock.mat | ../input
	ln -sf $< $@

../input/cognitive_intensity_matrix.csv: ../../cognitive_intensity_data/output/cognitive_intensity_matrix.csv | ../input
	ln -sf $< $@

../input/validation_report_4sector_reference.csv: ../../cdp4_baseline_validate/output/validation_report_4sector_reference.csv | ../input
	ln -sf $< $@

include ../../generic.make
