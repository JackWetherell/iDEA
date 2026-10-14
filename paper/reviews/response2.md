# Response to referee report, round 2

We thank the referee for verifying the round-1 changes.

1. **Table 1 caption.** Updated. Part (b) now describes all three dynamic and excitation checks with their parameters. Part (d) names the hardware.
2. **Timings.** Correction to our round-1 letter: Table 1(d) was not re-measured between rounds 1 and 2. The table in the round-1 manuscript had already been measured on an otherwise idle machine; an earlier internal draft, never sent to the referee, had non-monotonic timings from concurrent jobs. The text now says explicitly that Table 1(d) and the Examples timings come from a 4-core Xeon, which is not the machine used for the GPU benchmarks.
3. **"Exact for any δt".** Restricted to the exact propagation. We now say that the approximate methods, which update their Hamiltonian explicitly, are first order in δt.
4. **Midpoint Hamiltonian (optional).** We agree that this would make the propagator second order. We will consider it as a change to the code rather than within this paper.
5. **Restricted HF.** The paper now says the restricted option is intended for closed shells.
6. **TD inversion.** The paper now says that the root finder's convergence is not enforced and that users should check the returned per-step density error.
7. **Listing runtime.** The listing passes `tol=1e-8` to `reverse`.
8. **Fig. 2.**
   - The −1/(|x|+1) asymptote is overlaid for |x| > 3.
   - The text now says the exact v_xc "tends towards" the asymptote, with the residual difference attributed to the slow approach to the asymptotic region and the finite box. Hard walls are already stated in the Methods.
9. **Fig. 1 legend.** Moved to the empty upper-right region with a smaller font; it no longer overlaps the peak.
10. **KS-SCE.** The text notes that it can fail for densities with long low-density tails.
11. **Python versions and repository.**
    - Python versions now read "continuous integration with 3.10, also run with 3.13".
    - The canonical repository URL is left for the authors to confirm before submission.
12. **Eigenpair heuristic.** Disclosed as a heuristic that can be overridden. A runtime check is left as future code work.
13. **Interaction citation.** Now worded as "used in earlier iDEA work (e.g. Ref. [Hodgson2013])".
