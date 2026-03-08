SHELL := bash

.PHONY: all tasks paper reference fast setup baseline_solver baseline_reference baseline_strict_debug baseline_fast baseline_smoke baseline_validate_reference baseline_validate_strict_debug baseline_validate_fast baseline_validate_smoke counterfactual_identity_reference counterfactual_identity_fast counterfactual_toy_reference counterfactual_toy_fast counterfactual_validate_identity_reference counterfactual_validate_identity_fast counterfactual_validate_toy_reference counterfactual_validate_toy_fast ai_reference ai_fast diagnostic check clean

all: paper

tasks: setup baseline_solver baseline_reference baseline_validate_reference counterfactual_identity_reference counterfactual_validate_identity_reference counterfactual_toy_reference counterfactual_validate_toy_reference ai_reference

paper: tasks
	$(MAKE) -C paper all

reference: ai_reference

fast: ai_fast

setup:
	$(MAKE) -C tasks/setup_environment/code all

baseline_solver:
	$(MAKE) -C tasks/cdp4_baseline_solver/code all

baseline_reference:
	$(MAKE) -C tasks/cdp4_baseline_dynamics/code replication_reference

baseline_strict_debug:
	$(MAKE) -C tasks/cdp4_baseline_dynamics/code strict_debug

baseline_fast:
	$(MAKE) -C tasks/cdp4_baseline_dynamics/code replication_fast

baseline_smoke:
	$(MAKE) -C tasks/cdp4_baseline_dynamics/code smoke

baseline_validate_reference:
	$(MAKE) -C tasks/cdp4_baseline_validate/code replication_reference

baseline_validate_strict_debug:
	$(MAKE) -C tasks/cdp4_baseline_validate/code strict_debug

baseline_validate_fast:
	$(MAKE) -C tasks/cdp4_baseline_validate/code replication_fast

baseline_validate_smoke:
	$(MAKE) -C tasks/cdp4_baseline_validate/code smoke

counterfactual_identity_reference:
	$(MAKE) -C tasks/cdp4_counterfactual_dynamics/code identity_reference

counterfactual_identity_fast:
	$(MAKE) -C tasks/cdp4_counterfactual_dynamics/code identity_fast

counterfactual_toy_reference:
	$(MAKE) -C tasks/cdp4_counterfactual_dynamics/code toy_reference

counterfactual_toy_fast:
	$(MAKE) -C tasks/cdp4_counterfactual_dynamics/code toy_fast

counterfactual_validate_identity_reference:
	$(MAKE) -C tasks/cdp4_counterfactual_validate/code identity_reference

counterfactual_validate_identity_fast:
	$(MAKE) -C tasks/cdp4_counterfactual_validate/code identity_fast

counterfactual_validate_toy_reference:
	$(MAKE) -C tasks/cdp4_counterfactual_validate/code toy_reference

counterfactual_validate_toy_fast:
	$(MAKE) -C tasks/cdp4_counterfactual_validate/code toy_fast

ai_reference:
	$(MAKE) -C tasks/cognitive_intensity_data/code all
	$(MAKE) -C tasks/cognitive_counterfactual/code submission_reference

ai_fast:
	$(MAKE) -C tasks/cognitive_intensity_data/code all
	$(MAKE) -C tasks/cognitive_counterfactual/code submission_fast

diagnostic:
	$(MAKE) -C tasks/cdp4_cognitive_minimal/code all

check:
	@echo "Canonical submission build:"
	@echo "  make"
	@echo "Reference AI outputs only:"
	@echo "  make reference"
	@echo "Fast AI debug outputs only:"
	@echo "  make fast"
	@echo "Regression-only build:"
	@echo "  make tasks"
	@echo "Strict baseline debug:"
	@echo "  make baseline_strict_debug baseline_validate_strict_debug"
	@echo "MATLAB-faithful diagnostic:"
	@echo "  make diagnostic"

clean:
	@echo "Cleaning generated task, intermediate, and paper outputs..."
	@find tasks -type f \\( -path '*/output/*' -o -path '*/temp/*' -o -path '*/intermediate/*' \\) ! -name '.gitkeep' -delete
	@$(MAKE) -C paper clean
