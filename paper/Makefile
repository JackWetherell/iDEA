# Build the iDEA CPC paper.
#   make figures   - regenerate all figures and numbers (runs iDEA; ~30 min on a laptop)
#   make           - compile main.pdf

main.pdf: main.tex refs.bib figures/numbers.tex
	latexmk -pdf -interaction=nonstopmode -halt-on-error main.tex

figures figures/numbers.tex:
	cd .. && python paper/scripts/make_figures.py

clean:
	latexmk -C main.tex

.PHONY: figures clean
