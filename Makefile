SHELL := bash

.PHONY: all tasks smoke_pipeline setup baseline_solver baseline_dynamics baseline_dynamics_smoke baseline_dynamics_reference baseline_dynamics_fast validate validate_smoke validate_reference validate_fast counterfactual_smoke counterfactual_toy_smoke counterfactual_identity_reference counterfactual_identity_fast counterfactual_toy_reference counterfactual_toy_fast counterfactual_validate_identity_reference counterfactual_validate_identity_fast counterfactual_validate_toy_reference counterfactual_validate_toy_fast counterfactual_report_smoke counterfactual_report_identity_fast counterfactual_report_identity_reference counterfactual_report_toy_fast counterfactual_report_toy_reference counterfactual_impact_report_toy_smoke_fast counterfactual_impact_report_toy_reference cognitive_shocks cognitive_counterfactual_fast cognitive_validate_fast cognitive_report_fast cognitive_pipeline_fast cognitive_minimal check clean

all: tasks

tasks: setup baseline_solver baseline_dynamics validate

smoke_pipeline: setup baseline_solver baseline_dynamics_smoke validate_smoke

setup:
	$(MAKE) -C tasks/setup_environment/code all

baseline_solver:
	$(MAKE) -C tasks/cdp4_baseline_solver/code all

baseline_dynamics:
	$(MAKE) -C tasks/cdp4_baseline_dynamics/code all

baseline_dynamics_smoke:
	$(MAKE) -C tasks/cdp4_baseline_dynamics/code smoke

baseline_dynamics_reference:
	$(MAKE) -C tasks/cdp4_baseline_dynamics/code replication_reference

baseline_dynamics_fast:
	$(MAKE) -C tasks/cdp4_baseline_dynamics/code replication_fast

validate:
	$(MAKE) -C tasks/cdp4_baseline_validate/code all

validate_smoke:
	$(MAKE) -C tasks/cdp4_baseline_validate/code smoke

validate_reference:
	$(MAKE) -C tasks/cdp4_baseline_validate/code replication_reference

validate_fast:
	$(MAKE) -C tasks/cdp4_baseline_validate/code replication_fast

counterfactual_smoke:
	$(MAKE) -C tasks/cdp4_counterfactual_dynamics/code smoke

counterfactual_toy_smoke:
	$(MAKE) -C tasks/cdp4_counterfactual_dynamics/code toy_smoke

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

counterfactual_report_smoke:
	$(MAKE) -C tasks/cdp4_counterfactual_validate/code report_smoke

counterfactual_report_identity_fast:
	$(MAKE) -C tasks/cdp4_counterfactual_validate/code report_identity_fast

counterfactual_report_identity_reference:
	$(MAKE) -C tasks/cdp4_counterfactual_validate/code report_identity_reference

counterfactual_report_toy_fast:
	$(MAKE) -C tasks/cdp4_counterfactual_validate/code report_toy_fast

counterfactual_report_toy_reference:
	$(MAKE) -C tasks/cdp4_counterfactual_validate/code report_toy_reference

counterfactual_impact_report_toy_smoke_fast:
	$(MAKE) -C tasks/cdp4_counterfactual_validate/code impact_report_toy_smoke_fast

counterfactual_impact_report_toy_reference:
	$(MAKE) -C tasks/cdp4_counterfactual_validate/code impact_report_toy_reference

cognitive_shocks:
	$(MAKE) -C tasks/cognitive_counterfactual generate_shocks

cognitive_counterfactual_fast:
	$(MAKE) -C tasks/cognitive_counterfactual run_fast

cognitive_validate_fast:
	$(MAKE) -C tasks/cognitive_counterfactual validate_fast

cognitive_report_fast:
	$(MAKE) -C tasks/cognitive_counterfactual report_fast

cognitive_pipeline_fast:
	$(MAKE) -C tasks/cognitive_counterfactual all_fast

cognitive_minimal:
	$(MAKE) -C tasks/cdp4_cognitive_minimal all

check:
	@echo "Pipeline: setup -> baseline_solver -> baseline_dynamics -> validate"
	@echo "Run: make"

clean:
	@echo "Cleaning task outputs..."
	@find tasks -type f \\( -path '*/output/*' -o -path '*/temp/*' \\) ! -name '.gitkeep' -delete
