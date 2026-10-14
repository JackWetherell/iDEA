# Referee report, round 3 (final)

**Manuscript:** "iDEA: an open-source Python framework for exact many-electron quantum mechanics and density functional theory in one dimension" (CPC, Computer Programs in Physics)

**Revision reviewed:** commit b22326c, together with the authors' response (`paper/reviews/response2.md`).

**Recommendation: accept.** One administrative item must be settled before the manuscript is submitted: the canonical repository URL (item B1 below). It needs no further review. Everything else listed below is optional polish.

---

## Verification of the round-2 points

I checked each point against `main.tex`, the regenerated `table_validation.tex`, `numbers.tex`, the figure PDFs and the compiled `main.pdf` (6 pages). All `\num…` macros used in the text are defined, LaTeX reports no undefined references or citations, and the quoted numbers agree with the table and the figures.

1. **Table 1 caption: resolved.**
   - Part (b) now describes all three checks and their parameters: the Kohn mode; the single electron over 500 steps; and the driven two-electron test with ω = 0.25, N_x = 101, L = 20 and δv = 0.02 sin(0.4t) x for t ≤ 30.
   - These match `make_figures.py`, lines 141–158.
   - Part (d) names the hardware.
2. **Timings: resolved.** The response letter corrects the earlier statement. The text now says that the Table 1(d) and Examples timings come from a different machine from the GPU benchmarks.
3. **"Exact for any δt": resolved.**
   - The statement now applies only to the exact propagation, and the approximate methods are described as first order in δt.
   - This matches my independent check in round 2:
     - **Exact dipole:** unchanged at round-off when δt is halved.
     - **Adiabatic LDA:** changes by 6×10⁻⁴.
   - The script still does not compute the δt/2 run (optional item O1).
4. **Midpoint Hamiltonian: accepted as future code work.** This was optional.
5. **Restricted HF: resolved.** The text now says "The restricted option is intended for closed shells."
6. **Time-dependent inversion: resolved.** The paper now says that the root finder's convergence is not enforced and that users should check the per-step density error.
7. **Listing runtime: resolved.** The listing passes `tol=1e-8`. In round 2 I confirmed that the listing runs and reproduces the energies in `numbers.tex`. With this tolerance the reverse-engineering step is the setting used for Fig. 2, which took a few minutes.
8. **Fig. 2: resolved.**
   - The −1/(|x|+1) asymptote is overlaid for |x| > 3.
   - The wording is now "tends towards", with the remaining difference attributed to the slow approach to the asymptotic region and to the finite box.
   - Box-length convergence is still not shown. Given the masking and the honest wording, I no longer require it.
   - The overlaid curve is hard to see (optional item O2).
9. **Fig. 1 legend: resolved.** The legend no longer overlaps the curves.
10. **KS-SCE: resolved.** The text now says that the module "can fail for densities with long low-density tails."
11. **Program summary.**
    - The Python versions are now stated accurately: CI runs 3.10, and the code has also been run with 3.13.
    - **The repository URL is still unresolved** (item B1).
12. **Eigenpair heuristic: accepted.** It is disclosed in the text, and a runtime check is deferred to future code work.
13. **Interaction citation: resolved.** It now reads "used in earlier iDEA work (e.g. Ref. [Hodgson2013])".

All round-1 major issues were resolved by round 2, and every round-2 minor issue is now resolved or accepted.

---

## Must be done before submission

**B1. Canonical repository URL (administrative).**
- The program summary and CITATION.cff give `https://github.com/JackWetherell/iDEA`.
- The README badges, the "View on GitHub" link, the tutorial links and `pyproject.toml`'s Bug Tracker all point to `https://github.com/iDEA-org/iDEA`.
- CPC archives the code from the developer's repository link, so the authors must choose one, make the README, CITATION.cff, `pyproject.toml` and the paper agree, and make sure the chosen repository contains this revision (including `paper/scripts/make_figures.py` and the bug fixes).

---

## Optional polish (non-blocking)

**O1. Reproducibility of one sentence.**
- Sec. 4 says `make_figures.py` "regenerates every figure and number in this paper". The Fig. 3 statement that halving δt changes the exact d(t) only at round-off level relies on a δt/2 run (`d_exact_half` in the untracked cache) that the script does not perform.
- Either add that run to `fig3()` (about two extra minutes) or soften the sentence. I have verified the claim independently.

**O2. Fig. 2(a) asymptote is hard to see.**
- The −1/(|x|+1) curve is drawn as a thin dotted line in the same blue as the exact v_xc, so it is nearly invisible where the two curves are close.
- A grey or black dotted line, or a slightly thicker one, would make the comparison in the text visible to the reader.

**O3. Wording.**
- Sec. 2, "Systems": "…used in earlier iDEA work (e.g. Ref. [17]) (Hartree atomic units), a variant of…" has two consecutive parentheticals. Suggest: "…$1/(|x-x'|+1)$, used in earlier iDEA work [17], in Hartree atomic units throughout; it is a variant of…".
- Sec. 4, "Dynamics": consecutive sentences both start "the approximate methods" ("…the approximate methods, which update their Hamiltonian explicitly, are first order in δt. The approximate methods, propagated through the same interface…"). Merge them, or start the second with "Propagated through the same interface from their own ground states, they differ…".
- The abstract says "illustrate its use", where "its" now grammatically refers to "the exact solver". The examples also exercise the approximate methods and reverse engineering, so "illustrate the code" is more accurate.

**O4. Typography.**
- In the PDF, the line "All results below are produced by paper/scripts/make_figures.py in the repository" is badly stretched by the unbreakable `\texttt` path, leaving large gaps between words.
- Use `\path{}` from the `url` package (already loaded through hyperref), or put the path at the start of the sentence.
- The figure legends use a hyphen ("Hartree-Fock") while the text uses an en dash ("Hartree–Fock"); a Unicode "–" in the Matplotlib label would match.

**O5. Spelling.** The manuscript consistently uses -ise forms, except "generalized KS theory" in the Introduction. Keeping the American spelling to match the cited title is acceptable, but "generalised" would be consistent.

**O6. Future code work (no change to the paper needed).** The authors already list as future work:
- a runtime check that at least k+1 antisymmetric states were found;
- a warning when `scipy.optimize.root` fails in `reverse_propagation`;
- iteration caps in the SCF and inversion loops;
- the midpoint Hamiltonian for second-order time stepping.

I encourage them to implement these, as they are the main remaining robustness gaps in the code.

---

## Final assessment

The manuscript is now an accurate, concise and well-validated description of the code.

- **Claims match the code.** Every technical claim I checked is consistent with it.
- **Reproducible numbers.** All numbers are generated by a distributed script.
- **Honest limitations.** They are stated clearly: system size, the spin-polarised LDA, first-order propagation, the inversion safeguards and the experimental KS-SCE module.
- **Bugs fixed.** The code bugs found during review were fixed and are covered by regression tests.

I recommend acceptance once the repository URL (B1) is settled. The optional items can be handled at the authors' discretion during proof correction.
