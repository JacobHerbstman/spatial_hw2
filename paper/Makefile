SHELL := bash
PYTHON ?= $(shell for p in /Users/jacobherbstman/miniconda3/bin/python3 python3 /Library/Frameworks/Python.framework/Versions/3.13/bin/python3; do if [ -x "$$p" ] && "$$p" -c "import pandas,json" >/dev/null 2>&1; then echo "$$p"; exit 0; fi; done; echo python3)

CANONICAL_OUTPUT_DIR := ../tasks/cognitive_counterfactual/output
GENERATED := generated/macros.tex generated/table_baseline_validation.tex generated/table_ai_results.tex
FIGURES := figures/dynamics_immediate.pdf figures/dynamics_anticipated.pdf figures/state_map_immediate.pdf

all: paper.pdf

paper.pdf: paper.tex sections/*.tex $(GENERATED) $(FIGURES) spatial_hw2.bib
	latexmk -pdf -pdflatex="pdflatex -interaction=nonstopmode" -use-make paper.tex

generated/macros.tex generated/table_baseline_validation.tex generated/table_ai_results.tex: code/build_paper_assets.py \
	$(CANONICAL_OUTPUT_DIR)/baseline_validation_reference.csv \
	$(CANONICAL_OUTPUT_DIR)/selected_reference_delta.json \
	$(CANONICAL_OUTPUT_DIR)/summary_counterfactual_4sector_reference_cognitive_immediate.csv \
	$(CANONICAL_OUTPUT_DIR)/summary_counterfactual_4sector_reference_cognitive_anticipated.csv \
	$(CANONICAL_OUTPUT_DIR)/key_econ_window_means_reference_cognitive_immediate.csv \
	$(CANONICAL_OUTPUT_DIR)/key_econ_window_means_reference_cognitive_anticipated.csv | generated
	$(PYTHON) code/build_paper_assets.py --input-dir $(CANONICAL_OUTPUT_DIR) --output-dir generated

figures/dynamics_immediate.pdf: ../tasks/cognitive_counterfactual/output/dynamics_counterfactual_reference_cognitive_immediate.pdf | figures
	ln -sf ../../tasks/cognitive_counterfactual/output/dynamics_counterfactual_reference_cognitive_immediate.pdf $@

figures/dynamics_anticipated.pdf: ../tasks/cognitive_counterfactual/output/dynamics_counterfactual_reference_cognitive_anticipated.pdf | figures
	ln -sf ../../tasks/cognitive_counterfactual/output/dynamics_counterfactual_reference_cognitive_anticipated.pdf $@

figures/state_map_immediate.pdf: ../tasks/cognitive_counterfactual/output/state_map_realwages_reference_cognitive_immediate.pdf | figures
	ln -sf ../../tasks/cognitive_counterfactual/output/state_map_realwages_reference_cognitive_immediate.pdf $@

generated:
	mkdir -p generated

figures:
	mkdir -p figures

clean:
	rm -f generated/*.tex
	rm -f figures/*.pdf
	latexmk -CA

.PHONY: all clean
