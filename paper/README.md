# iDEA paper (Computer Physics Communications)

Source for the manuscript describing iDEA, prepared for the *Computer Programs in Physics* section of Computer Physics Communications.

| Path | Contents |
|---|---|
| `main.tex` | Manuscript (`elsarticle`, `5p` two-column layout) including the CPC program summary |
| `refs.bib` | Bibliography |
| `scripts/make_figures.py` | Regenerates every figure, table and number in the paper by running iDEA |
| `figures/` | Generated figures (`*.pdf`), `numbers.tex` and `table_validation.tex`; `figures/cache/` holds intermediate results |
| `reviews/` | Internal referee reports and responses from the pre-submission review rounds |

## Building

```bash
pip install -e ..            # from the paper/ directory, install iDEA
make figures                 # optional: recompute all results (delete figures/cache to start fresh)
make                         # compile main.pdf (needs latexmk and elsarticle)
```
