# Response to referee report, round 1

We thank the referee for a careful report. Several points found genuine bugs in the code, which are now fixed with regression tests. Every other point is addressed in the manuscript.

## Major issues

| # | Issue | Action |
|---|---|---|
| M1 | Restricted HF and hybrid double the exchange | **Code fixed** (`hartree_fock.py`, `hybrid.py`): the restricted exchange now uses the per-spin density matrix. New test `test_restricted_matches_unrestricted_for_closed_shell` fails before the fix and passes after. |
| M2 | Listing does not converge; no safeguards | The listing now uses `mixing=0.2`, and the text states that the default mixing is 0.5 with no iteration limit, and that sloshing-prone systems need smaller mixing. We deliberately do not change solver behaviour in this paper; safeguards are listed in the Conclusions as future work. The time-dependent inversion is described as practical only on small grids. |
| M3 | LDA is spin-polarised only; hybrid form | The text states that the LDA is the spin-polarised 1D HEG functional, evaluated on the total density and appropriate for fully spin-polarised systems. The program summary lists this as a restriction. Default α = 0.8 is stated, as is the form of the hybrid potential. |
| M4 | KS-SCE not usable as advertised; energy wrong | **Code fixed** (`kssce.py`): the gauge constant now satisfies E_SCE = ∫v_SCE n, and `total_energy` is gauge-independent, with a new regression test. The paper now describes KS-SCE as an experimental contributed module that is outside the common interface and needs the interaction functions. It is removed from the abstract's method list. |
| M5 | Spin handling, eigenpair heuristic, indexing bug | **Code fixed** (`interacting.antisymmetrize`): spaces and spins are now filtered with the same indices as fulls and energies. The text states that the spin string fixes S_z, so "ud" excited states include S_z = 0 triplets, and that the eigenpair count is a heuristic that can be set explicitly. |
| M6 | TD accuracy not demonstrated | The first-order character of the propagator is stated. A new validation drives two interacting electrons with a time-dependent field and compares with the dynamic harmonic-potential theorem: the error is 4.0e-3 at δt = 0.05 and 2.0e-3 at δt = 0.025, confirming first-order convergence (Table 1b). For Fig. 3 we note that the field is piecewise constant, so the propagator is exact for any δt (verified: halving δt changes d(t) at round-off level). |
| M7 | Fig. 2 gauge and edge artefacts | Potentials are now aligned physically, with the highest occupied KS eigenvalue set to −I and I from exact N and N−1 energies. They are masked where n < 1e-5. In this gauge the exact v_xc approaches −1/(\|x\|+1) (−0.133 vs −0.111 at x = −8), the LDA decays far too quickly (−0.029), and the singlet v_c is between −0.021 and 0.013 Ha with no spurious plateau. The text now uses these numbers. |
| M8 | GPU claims | The time-dependent GPU speed-up claim is removed. We now say that CuPy's Lanczos solver and a scaled Taylor series are used on the GPU, that Al-Mohy–Higham is CPU only, and that the quoted ground-state numbers are single-run benchmarks on stated hardware. |
| M9 | Validation and CI overstated | The abstract now says "validate the exact solver". The text says which tests run in CI and which are run locally, and that the approximate methods are not compared with independent implementations. |
| M10 | Positioning and citations | The Octopus sentence is corrected. The \|x\|+1 form is attributed to earlier iDEA work and contrasted with soft-Coulomb [Javanainen]. Snyder and Li are moved to a separate clause, Ruggenthaler to the inversion and mapping literature, and Lein & Kümmel added for steps. The Faraday Discussions general discussions are removed from the text. Eq. 2 is credited to earlier iDEA work and related to van Leeuwen–Baerends. Wang–Parr and Nielsen et al. are added, and the current-density TD variant is mentioned. |

## Minor issues

1. **Program summary.** Done:
   - restrictions field, supplementary material, version, documentation URL;
   - licence "GPLv3 (or later)", with `pyproject.toml` changed to GPL-3.0-or-later;
   - JupyterLab listed;
   - the repository link is kept as the current development repository; the authors should confirm the canonical URL.
2. Done.
3. Defaults stated (μ, p, mixing, tolerance, α), along with the mixing used in the figures.
4. Done.
5. Qualified: only the single-particle `solve` signatures coincide.
6. Done: "same stencil and grid spacing", and the system for 1(b) is given.
7. Reworded as "centre-of-mass (Kohn) excitation", with the `uu` case and parameters stated.
8. Hardware stated. Table 1(d) was re-measured on an idle machine; the earlier non-monotonic timings came from concurrent jobs.
9. Reworded.
10. Done: symlog axis for Fig. 1b, legend moved, the non-interacting error quoted, the LDA density shift described, and "for this system" dropped.
11. The pulse now lasts exactly 1 a.u., with the field set for 0 < t_j ≤ 1. The text notes that each method starts from its own ground state.
12. Legend text fixed to "Hartree-Fock".
13. Stated as a limitation and future work.
14. Unused macros are harmless (the generated macro file); elsarticle-num prints only cited entries.
15. One-line justification given.
16. Done.
17. Version unified to 1.1.0. The affiliation is York at the authors' request. The acknowledgements follow the README and git contributors.
18. The main text is about 4.3 pages.
