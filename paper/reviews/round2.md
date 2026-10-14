# Referee report, round 2

**Manuscript:** "iDEA: an open-source Python framework for exact many-electron quantum mechanics and density functional theory in one dimension" (CPC, Computer Programs in Physics)

**Revision reviewed:** commits f8afd2c, 7829452 and d600b87, together with the authors' response (`paper/reviews/response1.md`).

**Recommendation: minor revision**

---

## Overall assessment

The authors have taken the round-1 report seriously.

- **Code bugs fixed.** The three genuine code bugs (restricted exchange, KS-SCE gauge and energy, excited-state bookkeeping) are fixed and covered by regression tests. All 11 tests in `tests/test_methods.py` pass, which I checked by running them.
- **Clearer manuscript.** The text now states the restrictions of the exact solver and of the LDA, and gives the numerical defaults.
- **Fig. 2 corrected.** The gauge problem in Fig. 2 is resolved with a physically motivated choice.
- **TD validation added.** A proper time-dependent many-body test (the driven harmonic-potential theorem) is now included.
- **Overclaims withdrawn.** The GPU and CI claims that were overstated have been withdrawn or corrected.

I re-ran the key checks myself (details below). No remaining issue prevents publication. What is left is a set of caption, wording and reproducibility points. One of them corrects an inaccurate statement in the response letter about Table 1(d).

---

## Verification of the responses to the round-1 major issues

**M1. Restricted HF and hybrid: resolved.**
- Fix: `hartree_fock.py:67-68` and `hybrid.py:57-58` now use ½(ρ↑+ρ↓).
- Re-run: for the singlet atom (101 points), restricted and unrestricted HF both give −1.711297 Ha, and restricted and unrestricted hybrid both give −1.709364 Ha.
- A new test covers it. One residual point for open shells is minor issue 5.

**M2. Listing and safeguards: resolved for the paper; safeguards only partly addressed.**
- The revised listing (181 points, `mixing=0.2`) now runs to completion. The printed energies match `numbers.tex` exactly: −1.6388, −0.8852, −1.3388, −1.3348 and −1.3379 Ha.
- It took about 17 minutes on 4 cores. About 13 minutes of that was the final `reverse(...)` call at its default `tol=1e-12`, while sharing the CPU with another job. See minor issue 7.
- The text now discloses that the SCF loop has no iteration limit, and lists safeguards as future work. I accept this. One silent-failure path remains (minor issue 6).

**M3. LDA is spin-polarised only; hybrid form: resolved.** The restriction is stated in Sec. 2 and in the program summary. The hybrid potential's form and the default α = 0.8 are given.

**M4. KS-SCE: resolved.**
- Fix: `kssce.py:206` now enforces ∫v_SCE n = E_SCE, and `total_energy` is gauge-independent.
- `test_kssce_potential_gauge_and_energy` passes.
- The module is now described as experimental and is outside the common interface, and it is removed from the abstract. A documentation point remains (minor issue 10).

**M5. Exact solver: resolved.**
- (a) The S_z-versus-S statement is now correct.
- (b) The eigenpair heuristic is disclosed (see minor issue 12).
- (c) The indexing bug is fixed. For `"uud"` with `level=40`, ⟨space_k|H|space_k⟩ now equals E_k for every returned state to 1.6×10⁻¹⁴ (it was off by one level before).

**M6. Time-dependent validation: resolved.**
- The driven harmonic-potential-theorem test is correct. I re-derived the classical centre-of-mass solution x_cm(t) = −E₀/(ω²−Ω²)[sin Ωt − (Ω/ω) sin ωt].
- The cached errors (3.99×10⁻³ at δt = 0.05 and 2.00×10⁻³ at δt = 0.025) confirm first-order convergence.
- The pulse in Fig. 3 now lasts exactly 1 a.u.
- I checked the Fig. 3 claim independently with the same system (121 points, 1000 versus 2000 steps):
  - **Exact dipole:** unchanged at round-off level, as stated.
  - **Adiabatic LDA:** changes by 6×10⁻⁴ (amplitude 0.79), so the "exact for any δt" statement applies only to the exact propagation (minor issue 3).

**M7. Fig. 2 gauge: resolved.**
- The ionisation-energy gauge is implemented correctly: the ion is a one-electron problem, so the non-interacting solve is exact.
- The two ionisation energies are mutually consistent. E_ion = −1.3413 + 0.202 = −1.7162 + 0.577 = −1.139 Ha.
- The spurious v_c plateau has gone, and the masked curves are now credible.
- Remaining requests (box-size convergence, plotting the asymptote) are minor issue 8.

**M8. GPU claims: resolved.** The time-dependent speed-up claim has been removed. The GPU algorithms (CuPy's Lanczos solver and a scaled Taylor series) are now named correctly.

**M9. Validation and CI claims: resolved.** The text now correctly says which tests run in CI, and that the approximate methods are not compared with independent implementations.

**M10. Positioning and citations: resolved.**
- The Octopus sentence is corrected.
- The Snyder and Li citations are moved to their own clause.
- The new references (Lein & Kümmel 2005; Nielsen, Ruggenthaler & van Leeuwen 2013; Wang & Parr 1993) are correct and appropriate.
- The Faraday Discussions general discussions are removed.

**Round-1 minor issues.**
- **Resolved:** 1 (except the repository URL, minor 11), 2, 3, 4, 5, 6, 7, 9, 10, 11, 12, 13, 15, 16, 17 and 18.
- **14:** accepted (unused macros are harmless).
- **8: not resolved**; see minor issue 2 below.

---

## Remaining MAJOR issues

None.

---

## Remaining MINOR issues

**1. Table 1 caption not updated for the revision.**
- Part (b) still reads "Lowest dipole excitation energy ... and the time-dependent density error over 500 steps". It says nothing about the two new "driven HPT" rows. Please add their parameters: ω = 0.25, E₀ = 0.02, Ω = 0.4, N_x = 101, L = 20, t ≤ 30, `uu`. The main text also now calls this the "centre-of-mass (Kohn) excitation", so align the wording.
- Part (d) still says "on 4 CPU cores of the test machine". Use the hardware description from Sec. 4: "4 cores of an Intel Xeon (2.8 GHz) cloud machine".

**2. Table 1(d) was not re-measured, contrary to the response.**
- The response to round-1 minor 8 says Table 1(d) "was re-measured on an idle machine; the earlier non-monotonic timings came from concurrent jobs".
- The table values (0.2, 1.4, 3.4, 9.7, 21.5 s; 1.1, 6.9, 19.7 s) are identical to round 1. The cached raw values in `figures/cache/validation_timing.npz` (0.2419, 1.3956, …) are the same run.
- The non-monotonic timing raised in round 1 was in the GPU benchmark (`benchmarking/gpu_scaling.png`), not in Table 1(d).
- Please correct the response letter. In the text, also make clear that the RTX 4090/i9 figures and the Table 1(d) timings come from different machines; the round-1 inconsistency between them (21.5 s at N_x = 220 here against about 7–16 s in the benchmark plot) is then explained.

**3. Fig. 3: "the exponential propagator is exact for any δt here".**
- Restrict this to the exact many-body propagation.
- The time-dependent mean-field methods rebuild H from the previous step's density, so they are not exact. For adiabatic LDA I find a 6×10⁻⁴ change on halving δt, which is negligible for the figure but not zero.
- The δt/2 run behind the statement (`d_exact_half` in `figures/cache/fig3.npz`) is not produced by the distributed `make_figures.py`. Please add it to the script, since the paper claims the script regenerates every number.

**4. Optional: a second-order time step.** The first-order error is now documented: 0.75 % of the amplitude at δt = 0.05 for a continuously varying field. Evaluating the Hamiltonian at the midpoint, H(t_j − δt/2), would make the step second order at no extra cost (one line in `propagate_step`). I suggest it, but I do not require it.

**5. Restricted HF for open shells.**
- For open-shell strings, the restricted Hamiltonian uses the spin-averaged exchange ½(ρ↑+ρ↓). This is neither ROHF nor variationally consistent with `total_energy`, which uses spin-resolved exchange.
- Example: for `"uud"` in the atom, restricted HF gives −1.532 Ha against −1.657 Ha for UHF.
- Please state in Sec. 2 that `restricted=True` is intended for closed shells, or emit a warning when `up_count != down_count`.

**6. Silent failure in the time-dependent inversion.**
- `reverse_propagation` still calls `scipy.optimize.root(..., method="hybr", options={"maxfev": 200})` (`reverse_engineering.py:207-224`).
- It ignores `result.success`, and 200 evaluations cannot complete a finite-difference Jacobian for N_x ≥ 200.
- "Practical on small grids" is a fair description, but please add to the text that per-step convergence is not checked, and that users should inspect the returned error array. A one-line warning on `not result.success` would be better still.

**7. Listing run time.** The listing now works, but with the default `tol=1e-12` the final `reverse` call dominates: about 13 minutes on 4 cores. Either pass `tol=1e-8`, as used for Fig. 2, or note that this step takes minutes. Otherwise readers trying it interactively will assume it has hung.

**8. Fig. 2, remaining points.**
- **Overlay the asymptote.** The text compares the exact v_xc with −1/(|x|+1) (−0.133 against −0.111 at x = −8). Please plot this asymptote as a thin reference curve in panel (a). The gap is still about 20 % at x = −8 and 45 % at x = −4, so "tends towards" is more accurate than "approaches".
- **Convergence.** Box-length and inversion-tolerance convergence of the masked tail, requested in round 1, is not reported. One sentence would do, for example "unchanged to X mHa for L = 36".
- **Hard-wall caveat.** Mention that the ionisation-energy alignment is applied to hard-wall box energies.

**9. Fig. 1(a) legend.** The legend in the new upper-left position overlaps the density peak; the "Hybrid" entry crosses the curves near x = 0. Move it to the right-hand tail region or outside the axes.

**10. KS-SCE documentation.** The module still raises `ValueError` ("x must be strictly increasing") on the paper's own atom, because the cumulant is not monotonic where the density underflows. The new test covers only the harmonic well. Since the module is labelled experimental, a sentence in the docstring or the program summary is enough, for example "requires densities that do not underflow in the box".

**11. Program summary.**
- "tested with 3.10–3.13" should say where: CI runs only Python 3.10, and the 3.13 claim presumably comes from local runs.
- The canonical repository is still unresolved. The paper and CITATION.cff give `github.com/JackWetherell/iDEA`, while the README badges, homepage and bug tracker point to `github.com/iDEA-org/iDEA`. This must be settled before the CPC Library deposit.

**12. Eigenpair heuristic.** Disclosure is adequate for publication. I still recommend a runtime error when fewer than k+1 antisymmetric states survive, rather than an `IndexError` or a wrong state.

**13. Origin of the |x|+1 interaction.** It is now attributed to Hodgson et al. (2013). Please check that this is the earliest iDEA use; Ramsden & Godby (2012) may be the appropriate first citation.
