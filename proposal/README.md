# Research proposal: Nearsightedness at strong coupling

A research proposal and work plan for a theoretical model-systems paper built on iDEA. It asks whether the exact many-body self-energy remains nearsighted for strongly correlated electrons (Wigner molecules, stretched bonds), and compares it with *GW* and exact Kohn–Sham potentials.

| Path | Contents |
|---|---|
| `research_proposal.md` | The proposal (Markdown, renders on GitHub) |
| `latex/proposal.tex` | The same proposal in LaTeX |
| `latex/refs.bib` | Bibliography |
| `latex/proposal.pdf` | Compiled PDF |

## Building the PDF

```bash
cd latex
make            # latexmk -pdf proposal.tex
make clean
```
