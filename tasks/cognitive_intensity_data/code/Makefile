SHELL := bash
PYTHON ?= $(shell for p in /Users/jacobherbstman/miniconda3/bin/python3 python3 /Library/Frameworks/Python.framework/Versions/3.13/bin/python3; do if [ -x "$$p" ] && "$$p" -c "import pandas,numpy,matplotlib,requests,scipy,openpyxl" >/dev/null 2>&1; then echo "$$p"; exit 0; fi; done; echo python3)

.PHONY: all download compute summary clean

all: summary

download: ../intermediate/download_provenance.json

compute: ../output/cognitive_intensity_matrix.csv ../output/cognitive_intensity_wide.csv ../output/cognitive_intensity_metadata.json

summary: ../output/cognitive_intensity_summary.pdf

../intermediate/download_provenance.json: 01_download_data.py | ../input ../intermediate ../output ../temp
	$(PYTHON) 01_download_data.py

../output/cognitive_intensity_matrix.csv ../output/cognitive_intensity_wide.csv ../output/cognitive_intensity_metadata.json: 02_compute_intensity.py ../intermediate/download_provenance.json
	$(PYTHON) 02_compute_intensity.py

../output/cognitive_intensity_summary.pdf: 03_summary_stats.py ../output/cognitive_intensity_matrix.csv ../intermediate/national_sector_cognitive.csv
	$(PYTHON) 03_summary_stats.py

../input ../intermediate ../output ../temp:
	mkdir -p $@

clean:
	rm -f ../intermediate/*.json ../intermediate/*.csv
	rm -f ../output/*.csv ../output/*.pdf ../output/*.txt ../output/*.json
