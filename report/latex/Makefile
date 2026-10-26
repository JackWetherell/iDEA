.PHONY: all clean

all: report.pdf

report.pdf: report.tex refs.bib figures/values.tex $(wildcard figures/*.pdf)
	latexmk -pdf -interaction=nonstopmode report.tex

clean:
	latexmk -c
