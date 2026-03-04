SHELL := bash

.PHONY: all tasks smoke_pipeline setup baseline_solver baseline_dynamics baseline_dynamics_smoke baseline_dynamics_reference baseline_dynamics_fast validate validate_smoke validate_reference validate_fast check clean

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

check:
	@echo "Pipeline: setup -> baseline_solver -> baseline_dynamics -> validate"
	@echo "Run: make"

clean:
	@echo "Cleaning task outputs..."
	@find tasks -type f \\( -path '*/output/*' -o -path '*/temp/*' \\) ! -name '.gitkeep' -delete
