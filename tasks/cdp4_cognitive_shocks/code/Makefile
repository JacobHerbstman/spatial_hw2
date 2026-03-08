SHELL := bash
PYTHON ?= $(shell for p in /Users/jacobherbstman/miniconda3/bin/python3 python3 /Library/Frameworks/Python.framework/Versions/3.13/bin/python3; do if [ -x "$$p" ] && "$$p" -c "import pandas" >/dev/null 2>&1; then echo "$$p"; exit 0; fi; done; echo python3)

SCENARIO_NAME ?= cognitive_firstpass_d5_perm
DELTA ?= 0.05
TIME_HORIZON ?= 200
SCHEDULE ?= permanent
DEFAULT_MATRIX := $(abspath ../../cognitive_intensity_data/output/cognitive_intensity_matrix.csv)
MATRIX ?= $(DEFAULT_MATRIX)
EXTRA_DEPS := $(MATRIX)

.PHONY: all build clean

all: build

build: ../output/lambda_hat_$(SCENARIO_NAME).csv ../output/lambda_snapshot_$(SCENARIO_NAME).csv ../output/shock_manifest_$(SCENARIO_NAME).json

ifeq ($(MATRIX),$(DEFAULT_MATRIX))
build: ../../cognitive_intensity_data/output/cognitive_intensity_metadata.json
EXTRA_DEPS += ../../cognitive_intensity_data/output/cognitive_intensity_metadata.json

../../cognitive_intensity_data/output/cognitive_intensity_metadata.json:
	$(MAKE) -C ../../cognitive_intensity_data/code compute
endif

../output/lambda_hat_$(SCENARIO_NAME).csv ../output/lambda_snapshot_$(SCENARIO_NAME).csv ../output/shock_manifest_$(SCENARIO_NAME).json: 01_build_cognitive_shocks.py $(EXTRA_DEPS) | ../output ../temp
	$(PYTHON) 01_build_cognitive_shocks.py --matrix "$(MATRIX)" --output-dir ../output --scenario-name "$(SCENARIO_NAME)" --delta "$(DELTA)" --time-horizon "$(TIME_HORIZON)" --schedule "$(SCHEDULE)"

../output ../temp:
	mkdir -p $@

clean:
	rm -f ../output/lambda_hat_*.csv ../output/lambda_snapshot_*.csv ../output/shock_manifest_*.json
