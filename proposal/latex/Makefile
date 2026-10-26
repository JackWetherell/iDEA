.PHONY: all clean

all: proposal.pdf

proposal.pdf: proposal.tex refs.bib
	latexmk -pdf -interaction=nonstopmode proposal.tex

clean:
	latexmk -c
