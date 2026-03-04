SHELL := bash
include ../../shell_functions.make

all: ../output/benchmark_solver_4sector.csv

../output/benchmark_solver_4sector.csv: run_solver_checks.jl ../input/Base_year_four_sectors.mat | ../output
	$(JULIA) --project=../../_lib/cdp_julia run_solver_checks.jl

../input/Base_year_four_sectors.mat: ../../../CDP_codes_four_sectors/Base_year_four_sectors.mat | ../input
	ln -sf $< $@

include ../../generic.make
