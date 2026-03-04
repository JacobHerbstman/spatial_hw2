SHELL := bash

all: ../output/environment_report.txt

../output/environment_report.txt ../output/julia_packages_status.txt: check_environment.sh | ../output
	bash check_environment.sh

include ../../generic.make
