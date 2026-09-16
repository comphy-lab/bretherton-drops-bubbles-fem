LATEXMK ?= latexmk
PYTHON ?= python3
BUILD_DIR := build
TARGET := main.pdf
FIGURES := figures/film-thickness-vs-ca.pdf figures/rear-cap.pdf \
           figures/rear-flow.pdf

.PHONY: all figures clean

all: $(TARGET)

$(TARGET): main.tex references.bib $(FIGURES)
	mkdir -p $(BUILD_DIR)
	$(LATEXMK) -pdf -interaction=nonstopmode -halt-on-error -outdir=$(BUILD_DIR) main.tex
	cp $(BUILD_DIR)/$(TARGET) $(TARGET)

# The figures read only the CSVs under data/, so this needs NumPy and
# Matplotlib but neither pyoomph nor any simulation output. Both PDFs are
# committed, but they carry rules so that a clean checkout with the data and a
# deleted figure can rebuild it rather than failing inside LaTeX.
figures: $(FIGURES)

DATA := $(wildcard data/*.csv) $(wildcard data/*.csv.gz) \
        $(wildcard data/profiles/*.csv)

figures/film-thickness-vs-ca.pdf: figures/plot_film.py $(DATA)
	$(PYTHON) figures/plot_film.py

figures/rear-cap.pdf: figures/plot_rear.py $(DATA)
	$(PYTHON) figures/plot_rear.py

figures/rear-flow.pdf: figures/plot_rear_flow.py $(DATA)
	$(PYTHON) figures/plot_rear_flow.py

clean:
	$(LATEXMK) -C -outdir=$(BUILD_DIR) main.tex
	rm -f $(TARGET)
	rmdir $(BUILD_DIR) 2>/dev/null || true
