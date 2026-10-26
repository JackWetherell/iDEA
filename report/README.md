# Technical report: exact Green's function and self-energy

`latex/report.pdf` describes how iDEA builds the exact one-body Green's function G(x,x';t or ω) and the exact exchange-correlation self-energy Σxc(x,x';t or ω) from the exact eigenstates of the N−1, N and N+1 electron systems. It covers:

- **Background and intuition:** what each element of G and Σ means physically.
- **The method:**
  - sector Hamiltonians in a Slater-determinant basis;
  - Jordan–Wigner field operators;
  - block Krylov with exact moments;
  - the self-energy by Löwdin partitioning and block Lanczos;
  - the time ordering of the self-energy poles.
- **How the code works:** `iDEA.methods.interacting.green_function`, `iDEA.state.GreenFunction`, `iDEA.observables` and `iDEA.plotting`.
- **How it is tested:** `tests/test_green_function.py`.
- **Illustrative results:** the one-dimensional hydrogen atom, compared with GW.
- **Production calculations:** the runs to do on a GPU (Appendix A).

## Building

```
make -C report/latex          # needs pdflatex, bibtex and latexmk
```

## Regenerating the figures and numbers

Every figure in `latex/figures/` and every number quoted in the text (`latex/figures/values.tex`) is written by `make_figures.py`. Re-running it on a finer grid and rebuilding updates the whole report:

```
python report/make_figures.py                                    # nx = 61, about a minute on a laptop
python report/make_figures.py --nx 151 --dense-max 30000 --gpu   # production grid (CuPy)
make -C report/latex
```
