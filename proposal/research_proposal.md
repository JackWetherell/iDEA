# Nearsightedness at strong coupling

**Exact self-energies versus *GW* and Kohn–Sham potentials in one-dimensional Wigner molecules and stretched bonds**

*Research proposal and work plan for a theoretical model-systems paper built on the iDEA code.*
*Draft, October 2026. LaTeX version: [`latex/proposal.tex`](latex/proposal.tex) (compiled: [`latex/proposal.pdf`](latex/proposal.pdf)).*

---

## Summary

Kohn's principle of *nearsightedness* says that local electronic properties depend only weakly on distant changes in the external potential. It underlies linear-scaling methods, embedding theories and every local or semilocal approximation in electronic-structure theory. In earlier work with the iDEA code [1], we showed that the many-body self-energy $\Sigma$ of many-body perturbation theory (MBPT) is advantageously nearsighted, whereas the exact Kohn–Sham (KS) potential is not. When two distant subsystems are combined, the exact KS exchange–correlation (xc) potential develops a step between them, while Hartree–Fock and *GW* self-energies stay local.

That study used *approximate* self-energies, spinless electrons (as in the pre-1.0 iDEA code) and weakly correlated systems.

The decisive test is strong coupling: stretched bonds, Mott insulators and Wigner molecules. This is where both density-functional theory (DFT) and *GW* are known to fail, and where few-electron Wigner physics is now imaged in carbon nanotubes [2, 3], quantum dots [4] and moiré artificial atoms [5].

We propose to compute, for the first time, **exact one-body Green's functions and self-energies of interacting electrons in the one-dimensional (1D) continuum across the crossover into the Wigner regime**. We will use them to answer when the exact self-energy is nearsighted and when it is not, how *GW* and the exact KS potential compare, and what this implies for practical theories.

**Central hypothesis.** Whether the exact self-energy is nearsighted is controlled by near-degeneracy and spin.
- For non-degenerate subsystems, and for same-spin electrons, the exact $\Sigma$ remains nearsighted at all frequencies.
- In the strongly correlated singlet (Heitler–London/Mott) regime, $\Sigma$ develops a low-frequency component that couples distant subsystems. Its frequency-integrated weight does not decay with separation: it is the real-space fingerprint of the Mott pole.
- Approximate *GW* stays nearsighted in all cases and therefore misses this physics.
- The exact KS potential is far-sighted in all cases, through steps and peaks.

A closed-form result for the Hubbard dimer, which we have verified numerically (Sec. 2), already shows this structure.

The work uses iDEA's exact many-electron solver, GPU-accelerated so that the three-electron sector needed for electron addition is affordable, together with the space-time *GW* implementation in open pull request #3, plus a new exact-Green's-function module. It targets *Physical Review B*, or *Physical Review Letters* if the spin- and degeneracy-controlled breakdown proves clean and general. Planned duration is about six months part-time, with under 50 GPU-hours on a single RTX 4090.

---

## 1. Background and motivation

### 1.1 Nearsightedness, and what we showed in 2019
Kohn introduced nearsightedness to justify linear-scaling DFT [6]. Prodan and Kohn sharpened it: at fixed chemical potential, the density at a point responds only to changes in the effective potential nearby [7]. Nearsightedness is why semilocal functionals, divide-and-conquer methods and local embedding work at all.

The exact KS potential, however, is *not* nearsighted in this sense. When a system consists of weakly coupled subsystems with different local ionisation energies, the exact xc potential forms spatial steps whose height is set by the global electron distribution [8–10]. In the symmetric stretched two-electron singlet, it forms a peak at the bond midpoint [11, 12].

In Ref. [1] we contrasted this with MBPT. In a 1D model of two electrons in separate wells, the exact KS potential showed a step, whereas Hartree–Fock and *GW* self-energies showed none. When distant subsystems were combined, the self-energy demonstrated Kohn's nearsightedness numerically. The outlook was that self-energy-based approximations should transfer better between environments than KS potentials, and that "nearsighted" self-energy ingredients could inform future KS-like theories. Related nonlocal-potential schemes from the same programme, such as Koopmans-compliant hybrids [13] and generalised KS with a local correlation potential [14], point the same way.

### 1.2 Why strong coupling is the decisive test
The 2019 comparison involved approximate self-energies for weakly correlated, spinless electrons, a regime where *GW* is expected to be reasonable. Its conclusion is therefore only as strong as *GW*'s physics.

In the strongly correlated regime, *GW* is known to fail qualitatively:
- In H₂ dissociation, self-consistent *GW* misses static correlation and suffers delocalisation and fractional-charge errors [15, 16].
- In the atomic (Mott) limit of the Hubbard dimer, *GW* fails because the vertex is neglected [17].
- For two electrons on a sphere, self-consistency can make *GW* worse [18].

The exact self-energy in this regime is expected to contain a "Mott pole" near the chemical potential. That is precisely the sort of feature that might *not* be nearsighted. Whether MBPT's nearsightedness advantage survives strong correlation is therefore open, and it matters for:
- local and embedding self-energy schemes, such as dynamical mean-field theory (DMFT) with its local self-energy [19];
- the connector-theory programme for transferring spectral information from models to materials [20–22];
- any attempt to build practical "nearsighted" MBPT-inspired functionals.

Strongly correlated few-electron systems are also experimentally topical:
- two-electron Wigner molecules in ultraclean nanotubes [2];
- imaging of 1D Wigner crystals [3];
- coherent control of a quantum-dot Wigner molecule [4];
- Wigner molecular crystals in moiré superlattices [5];
- recent *ab initio* treatments of small 1D Wigner crystals and their exchange couplings [23, 24].

### 1.3 What is missing
- **Exact self-energies exist only for lattice models.**
  - The Hubbard dimer [17].
  - The spectral potential of the asymmetric dimer [22].
  - Exact benchmarks for *GW* and vertex corrections in the extended Hubbard model [25].
  - A recent quantitative analysis argues that the celebrated cancellation of vertex corrections, previously inferred from model vertices, needs re-examination [26].
- **No real-space study of nonlocality.** Nobody has mapped the real-space nonlocality of an exact self-energy as a function of separation and correlation strength, nor compared it with *GW* and with the exact KS potential for the same systems.
- **Local frequency-dependent alternatives are untested in the continuum.** The spectral potential [20, 21] and a dynamical xc field in lieu of the self-energy [27] have not been tested against exact continuum references.

### 1.4 Why now, and why iDEA
iDEA (v1.1.0) solves the static and time-dependent many-electron Schrödinger equation exactly on a 1D real-space grid, for any external potential, interaction and spin configuration, with a GPU-accelerated exact solver. It also provides:
- non-interacting, Hartree, Hartree–Fock, LDA and hybrid methods;
- an experimental KS strictly-correlated-electrons (KS-SCE) module [28–30];
- exact KS inversion.

Three recent developments make this project feasible now:
- **Space-time *GW*.** Open PR #3 adds G₀W₀, GW₀ and fully self-consistent *GW* in the space-time formalism [31, 32]. It works on optimally chosen imaginary-time and imaginary-frequency grids in the spirit of the discrete Lehmann representation [33], and provides Galitskii–Migdal energies [34]. It has been validated against Lehmann sum-over-states references and against the legacy iDEA *GW* used in Ref. [1].
- **GPU acceleration.** Exact three-electron Hamiltonians on grids of about 100–150 points are now practical. This is exactly the (N+1) sector needed for the electron-addition part of an exact Green's function of a two-electron system.
- **Consolidated fixes.** A set of bug fixes and many-body observables is ready on feature branches (Sec. 6.1).

---

## 2. Central hypothesis and preliminary analytic result

### 2.1 The Hubbard dimer
For the half-filled Hubbard dimer ($H = -t\sum_\sigma (c^\dagger_{1\sigma}c_{2\sigma} + \mathrm{h.c.}) + U\sum_i n_{i\uparrow}n_{i\downarrow}$), the exact self-energy relative to the non-interacting Hamiltonian is diagonal in the bonding/antibonding basis [17]:

$$
\Sigma_{b/a}(\omega) = \frac{U}{2} + \frac{U^2/4}{\omega - \mu \mp 3t}, \qquad \mu = \frac{U}{2}.
$$

Transforming to the site basis and continuing to imaginary frequency $\omega = \mu + i\nu$:

$$
\Sigma_{12}(\mu + i\nu) = -\frac{3tU^2/4}{\nu^2 + 9t^2}, \qquad
\int_{-\infty}^{\infty}\! d\nu\, \Sigma_{12}(\mu+i\nu) = -\frac{\pi U^2}{4}, \qquad
\Sigma_{12}(\mu) = -\frac{U^2}{12t}.
$$

We verified these expressions by exact diagonalisation in Fock space (Appendix A). The pole position $3t$, residue $U^2/4$ and constant $U/2$ are reproduced to ten digits for all $t/U$ tested, $0.05 \le t/U \le 0.2$. The integrated weight $-\pi U^2/4$ is reproduced to quadrature accuracy for $0.01 \le t/U \le 0.2$.

### 2.2 Interpretation
When the two sites are moved apart, $t$ decays exponentially with separation, and the inter-site self-energy behaves in three ways at once:
- **Finite frequency:** at fixed $\nu \gg t$, $\Sigma_{12} \approx -3tU^2/(4\nu^2) \propto t$. It decays exponentially with separation, so it is *nearsighted*.
- **Chemical potential:** $\Sigma_{12}(\mu) \propto 1/t$ *grows* exponentially, so it is *ultra-far-sighted*.
- **Integrated weight:** $-\pi U^2/4$, *independent of separation*.

So in the strongly correlated limit, the nonlocal part of the exact self-energy does not disappear. It collapses onto a frequency window of width $\sim 3t$ around $\mu$. This is the Mott pole viewed in real space. A static (quasiparticle-like) approximation $\Sigma(\mu)$, as used in many self-consistent schemes, would be maximally far-sighted.

### 2.3 Two controls that restore nearsightedness
- **Asymmetry.** We diagonalised the asymmetric dimer (site energies $\pm\Delta/2$) numerically, with $U=1$ and the chemical potential at midgap. Asymmetry suppresses the inter-site self-energy, but only gradually:

| $t/U$ | $\Delta/U$ | $\Sigma_{12}(\mu)/U$ | $\Sigma_{12}(\mu+0.3iU)/U$ | $\int d\nu\,\Sigma_{12}/U^2$ |
|---|---|---|---|---|
| 0.05 | 0 | −1.667 | −0.333 | −0.785 |
| 0.05 | 0.2 | −1.111 | −0.303 | −0.641 |
| 0.05 | 0.5 | −0.390 | −0.201 | −0.379 |
| 0.01 | 0 | −8.333 | −0.083 | −0.785 |
| 0.01 | 0.05 | −4.908 | −0.082 | −0.603 |
| 0.01 | 0.2 | −0.685 | −0.074 | −0.225 |
| 0.01 | 0.5 | −0.117 | −0.049 | −0.093 |

- **Spin.** For two same-spin electrons on two sites, the ground state is a single determinant. The self-energy then reduces to its static Hartree–Fock form: zero for on-site $U$ only, and static exchange with inter-site interaction. There is no low-frequency nonlocal component.

### 2.4 What the continuum adds
The dimer cannot tell us four things:
- the real-space range and shape of the nonlocal component;
- how long-range Coulomb tails and the internal structure of Wigner molecules modify the picture;
- how the exact KS potential (steps and peaks) compares quantitatively;
- whether *GW* captures any of it.

These are the questions below.

---

## 3. Research questions and hypotheses

| | Question | Hypothesis | What would falsify it |
|---|---|---|---|
| **RQ1** | In the 2019 setting (asymmetric double well, one electron per well), is the **exact** $\Sigma(x,x';i\nu)$ nearsighted? With spin, does the singlet differ from the triplet? | **H1.** For asymmetry $\Delta \gg t_\mathrm{eff}$, the exact $\Sigma$ is nearsighted at all frequencies (it confirms and strengthens Ref. [1]), while the exact $v_\mathrm{xc}$ shows the step. Singlet and triplet behave alike. | Sizeable inter-well block at low $\nu$ for $\Delta \gg t_\mathrm{eff}$. |
| **RQ2** | In the symmetric dissociation limit, how does the inter-subsystem block $\Sigma_{AB}(i\nu)$ scale with separation $d$ and frequency? Does it collapse onto the dimer form with $t_\mathrm{eff}(d)$ and $U_\mathrm{eff}$ extracted from the exact spectra? How does asymmetry restore nearsightedness? | **H2 (central).** In the singlet, $\Vert\Sigma_{AB}\Vert$ at fixed $\nu$ decays like $t_\mathrm{eff}(d)$; $\Sigma_{AB}(\mu)$ grows like $1/t_\mathrm{eff}$; the frequency-integrated weight is $d$-independent ($\approx \pi U_\mathrm{eff}^2/4$ when projected on localised orbitals). The triplet shows no such component. Asymmetry restores nearsightedness over a range $\Delta \sim t_\mathrm{eff}$ to $U_\mathrm{eff}$. | The weight decays with $d$, or the triplet shows the same low-frequency component. |
| **RQ3** | Within a single soft well (a two-electron Wigner molecule as the coupling parameter $\lambda$ grows), does the spatial range of $\Sigma$ track the Wigner spacing, and at which frequencies? | **H3 (exploratory).** The range grows with $\lambda$ and concentrates at low $\nu$, driven by near-degeneracies of the (N+1) Wigner molecule. The *GW* range is set by Hartree–Fock orbitals and does not grow correspondingly. | The range is independent of $\lambda$. |
| **RQ4** | How do Hartree–Fock (restricted and unrestricted), G₀W₀@HF, GW₀ and self-consistent *GW* compare with the exact nonlocality? How does this relate to the far-sighted exact $v_\mathrm{xc}$ and to KS-SCE, which is exact at strong coupling? | **H4.** Spin-restricted *GW* is *spuriously* nearsighted (no Mott pole, so the gap is too small). Unrestricted-HF-based *GW* opens the gap through static symmetry breaking: nearsighted but spin-contaminated, the MBPT face of the symmetry dilemma [35]. Exact $v_\mathrm{xc}$ is far-sighted (step or peak) in every case; KS-SCE reproduces these features at strong coupling [29, 36]. | Restricted *GW* reproduces the low-frequency inter-subsystem weight. |
| **RQ5** | What is lost by *enforcing* nearsightedness, and what is the minimal far-sighted ingredient? | **H5.** Truncating $\Sigma$ to intra-subsystem blocks is harmless for the triplet and asymmetric cases but closes the gap in the symmetric singlet. Adding a rank-2 low-frequency "Mott-pole" coupling term, with parameters from local quantities, restores the ionisation potential (IP), electron affinity (EA) and gap. | Truncation is harmless for the singlet, or the pole term fails to restore the gap. |

---

## 4. Model systems and parameter space

The default interaction is iDEA's softened $v(x,x') = 1/(|x-x'|+1)$ [1, 9], which keeps continuity with the 2019 work and with PR #3's validation. The soft-Coulomb $1/\sqrt{(x-x')^2+1}$ is a check, and a short-range (Yukawa-screened) interaction matrix is a control that separates genuine correlation nonlocality from the Coulomb tail. Atomic units are used throughout. In effective-mass units, for example GaAs with Ha\* ≈ 11 meV and $a_B^*$ ≈ 10 nm, the same models describe quantum-dot and nanotube devices.

| System | External potential | $N$, spin | Scan | Probes |
|---|---|---|---|---|
| **S1** asymmetric double well ("2019 analogue") | Two softened-Coulomb or Gaussian wells, depths $V_A \neq V_B$, separation $d$ | 2, `ud` (singlet) and `uu` (triplet) | $d$ = 2–12; $\Delta = V_A - V_B$ from 0 to O($U_\mathrm{eff}$) | RQ1, RQ2 (asymmetry crossover), RQ4 |
| **S2** symmetric double well (stretched bond) | As S1 with $V_A = V_B$ | 2, `ud` / `uu` | $d$ until $t_\mathrm{eff} \sim 10^{-5}$ | RQ2, RQ4, RQ5 |
| **S3** single soft well (Wigner molecule) | $\tfrac12\omega^2x^2$ | 2, `ud` / `uu` | $\lambda = \omega^{-1/2}$ = 1–6 | RQ3, RQ4 |
| **S4** (stretch goal) | As S1–S3 | 3, removal sector only | — | Three-electron photoemission spectra |

The exact parameters of S1 will be matched to Ref. [1] in WP0, from the published code repository [JackWetherell/advantageous-nearsightedness](https://github.com/JackWetherell/advantageous-nearsightedness). Grids will have $N_x$ ≈ 80–150 points with iDEA's default 13-point stencil. Development will be at $N_x \le 60$ on a CPU; production will be on the GPU.

---

## 5. Methods

### 5.1 Exact ground states
`iDEA.methods.interacting.solve(s, k, GPU=True)` returns $\Psi_0^N$, and $\langle S^2\rangle$ labels the spin states (branch feat/46). For $N=2$ in the `ud` configuration, the spatial wavefunction $\psi(x_1,x_2)$ (electron 1 up, electron 2 down) fully specifies the state.

### 5.2 Exact one-body Green's function
We use the conventions of PR #3: zero temperature, real imaginary-time functions, energies measured from $\mu$, kernels on the grid. For spin $\sigma$:

$$
G_\sigma(x,x';\tau) = -\theta(\tau)\sum_n f_n(x)f_n^*(x')\,e^{-(\varepsilon_n^+-\mu)\tau} + \theta(-\tau)\sum_m g_m(x)g_m^*(x')\,e^{-(\varepsilon_m^- -\mu)\tau},
$$

with $\varepsilon_n^+ = E_n^{N+1}-E_0^N$, $f_n(x) = \langle\Psi_0^N|\hat\psi_\sigma(x)|\Psi_n^{N+1}\rangle$, $\varepsilon_m^- = E_0^N - E_m^{N-1}$ and $g_m(x) = \langle\Psi_m^{N-1}|\hat\psi_\sigma(x)|\Psi_0^N\rangle$.

- **Removal sector, exact and complete for $N=2$.** The (N−1) states are the $N_x$ eigenstates $\phi_m$ of the one-body Hamiltonian. The Dyson amplitudes are $g_m(x) = \int dx_2\,\phi_m^*(x_2)\,\psi(x,x_2)$ when an up electron is removed, which satisfy $\sum_m |g_m(x)|^2 = n_\uparrow(x)$. This is the same algebra as `mbpt.non_interacting_green_function`, with orbitals replaced by Dyson amplitudes.
- **Addition sector, by Krylov methods.** We use
  $$G^+_\sigma(x,x';i\omega) = \langle\chi_x|\,[\,i\omega+\mu-(\hat H^{N+1}-E_0^N)\,]^{-1}|\chi_{x'}\rangle, \qquad |\chi_x\rangle = \hat\psi^\dagger_\sigma(x)|\Psi_0^N\rangle.$$
  - $\hat H^{N+1}$ is iDEA's exact three-electron Hamiltonian (`interacting.hamiltonian` for a `uud` system).
  - The source $|\chi_{x'}\rangle$ is built on the grid with the correct antisymmetry in the like-spin coordinates; since the Hamiltonian commutes with like-spin permutations, Lanczos stays in the right sector.
  - One Lanczos run per source $x'$, with projections onto all $\langle\chi_x|$, yields $G^+$ at every imaginary time (via the exponential of the tridiagonal matrix) and every frequency (via the resolvent) on PR #3's `SpaceTimeGrid`, and on the real axis with broadening $\eta$ for spectral functions $A(x,\omega)$.
  - Block or band Lanczos over all sources is an optimisation.
- **Information that needs no analytic continuation.** The IP and EA follow from the large-$|\tau|$ decay of $G(\tau)$; the density from $G(x,x;0^-)$; the total energy from the Galitskii–Migdal formula.

### 5.3 Exact self-energy by Dyson inversion
With the exact density $n$ and one-body density matrix $\gamma$:

$$
\Sigma_c^{\mathrm{exact}}(i\omega) = i\omega + \mu - \hat h_0 - v_H[n] - \Sigma_x[\gamma] - G^{-1}(i\omega),
$$

so that $\Sigma_c \to 0$ as $\omega \to \infty$, since the static limit of the exact $\Sigma$ is the Hartree–Fock operator of the exact $\gamma$. On the grid this inverts PR #3's `dyson_equation` (kernel × $\delta x$ convention):

$$\Sigma_c = [(i\omega+\mu)\mathbb 1 - H - (G\,\delta x)^{-1}]/\delta x,$$

where $H$ contains the kinetic, external, Hartree and exchange parts (`mbpt.exchange_self_energy`). We report $\Sigma_{xc} = \Sigma_x + \Sigma_c$ and its parts. If needed, a reduced natural-orbital basis controls round-off at high frequency.

### 5.4 Exact Kohn–Sham potentials
- **Two-electron singlet (analytic).** $v_\mathrm{KS}(x) = \varepsilon_H + \tfrac12\,\partial_x^2\sqrt{n}/\sqrt{n}$, with $\varepsilon_H = -I$ (exact IP), and $v_\mathrm{xc} = v_\mathrm{KS} - v_\mathrm{ext} - v_H$.
- **Triplet and other cases.** `iDEA.reverse_engineering.reverse(s, n, iDEA.methods.non_interacting)`, with the constant fixed by the IP.

### 5.5 Approximations compared
- Hartree–Fock, restricted and unrestricted.
- G₀W₀@HF, GW₀ and self-consistent *GW* (PR #3, `mbpt.solve(mode=…)`), with restricted and unrestricted starting points.
- LDA and hybrid, as local and semilocal KS references.
- KS-SCE, as a far-sighted functional that is exact at strong coupling.

### 5.6 Nearsightedness diagnostics
Subsystems are defined by smooth partitions $P_A(x) + P_B(x) = 1$ placed at the barrier, and by localised (Wannier-like) orbitals $w_{A,B}$ from the exact natural orbitals.

| | Diagnostic | Definition | KS counterpart |
|---|---|---|---|
| **M1** | Subsystem transferability | $\Vert P_A(\Sigma^{AB}-\Sigma^{A})P_A\Vert / \Vert P_A\Sigma^{A}P_A\Vert$, with $\Sigma^A$ the exact self-energy of subsystem $A$ alone | $\Vert P_A(v_\mathrm{xc}^{AB}-v_\mathrm{xc}^{A})\Vert$ in the physical gauge |
| **M2** | Range | Hilbert–Schmidt norm $\Vert\Sigma_{AB}(i\nu)\Vert$ versus $d$ and $\nu$; decay length $\xi_\Sigma(\nu)$ | Step or peak height versus $d$ |
| **M3** | Response | $\delta\Sigma(x,x';i\nu)/\delta v_\mathrm{ext}(y)$ for $y$ in the distant subsystem (finite differences) | $\delta v_\mathrm{xc}(x)/\delta v_\mathrm{ext}(y)$, alongside $\delta n(x)/\delta v_\mathrm{ext}(y)$ |
| **M4** | Integrated weight | $\int d\nu\,\langle w_A \vert \Sigma(i\nu) \vert w_B\rangle$, compared with $-\pi U_\mathrm{eff}^2/4$ | — |

### 5.7 Consequence tests (RQ5)
- **Truncation.** $\Sigma^{\mathrm{near}} = P_A\Sigma P_A + P_B\Sigma P_B$; solve Dyson (`mbpt.dyson_equation`); compare $n$, IP, EA, gap and the Galitskii–Migdal energy with the exact values.
- **Pole correction.** Add the rank-2 term $\Sigma^{\mathrm{pole}}_{AB}(i\nu) = -\tfrac{3t_\mathrm{eff}U_\mathrm{eff}^2/4}{\nu^2+9t_\mathrm{eff}^2}\,\big(|w_A\rangle\langle w_B| + |w_B\rangle\langle w_A|\big)$.
- **Parameters for the pole term.** Both are local spectral quantities: $t_\mathrm{eff}$ from the (N−1) bonding–antibonding splitting, $U_\mathrm{eff}$ from the subsystem IP−EA.

### 5.8 Validation (automated tests)
1. **Zero interaction:** $\Sigma^{\mathrm{exact}} \equiv 0$ to round-off.
2. **Small grids ($N_x \approx 20$–25):** full diagonalisation of the $N\pm1$ sectors gives a Lehmann reference, which the Lanczos $G$ must match to about $10^{-12}$.
3. **Density matrix:** $G(x,x';0^-) = \gamma(x,x')$ from `observables.density_matrix`.
4. **Jump condition:** $G(0^+)-G(0^-) = -\delta(x-x')/\delta x$.
5. **Static limit:** $\Sigma(i\omega\to\infty) \to v_H + \Sigma_x[\gamma]$.
6. **Energy:** the Galitskii–Migdal energy from the exact $G$ (`mbpt.total_energy`) equals the exact eigenvalue.
7. **One electron:** the exact $\Sigma$ is checked against brute-force two-electron diagonalisation, which connects to self-interaction studies of *GW* [37, 38].
8. **Dimer limit:** for deep, narrow, well-separated wells, $\langle w_A|\Sigma|w_B\rangle$ follows the Sec. 2 form with fitted $t$ and $U$.

---

## 6. Implementation in iDEA

### 6.1 Prerequisites (WP0)

| Item | Where | Why it is needed |
|---|---|---|
| Space-time *GW*, `SpaceTimeGrid`, `dyson_equation`, Galitskii–Migdal energy, `QuasiParticleState` | PR #3 (`iDEA/methods/mbpt.py`) | *GW* comparisons, and common grids and conventions for exact $G$ and $\Sigma$ |
| Restricted HF and hybrid exchange was doubled | branch `fix/40-rhf-exchange` | Restricted HF and *GW* starting points |
| `antisymmetrize` misaligned `space`/`spin` with `full` | branch `fix/42-antisymmetrize-alignment` | Correct spatial parts of excited states (used to build sources) |
| Many-body kinetic and interaction energies (pair density) | branch `feat/45-manybody-observables` | Energy decompositions, Galitskii–Migdal validation |
| $\langle S_z\rangle$ and $\langle S^2\rangle$ | branch `feat/46-spin-observables` | Spin labels of the $N\pm1$ states |
| Indexing into `ManyBodyStates` | branch `feat/51-manybodystates-index` | Brute-force Lehmann validation |
| Hartree total energy and KS-SCE gauge/energy fixes | branch `ccr-6064c011-6imjt7` | Correct mean-field and KS-SCE energies |
| KS-SCE checks: default `interp_vsce`, parity of `ddinteraction`, $1/n$ in dilute tails | new | KS-SCE comparisons (RQ4); validate against Ref. [29] |

Crank–Nicolson propagation (PR #5) is not needed, because nothing here is propagated in real time.

### 6.2 New code (following existing iDEA patterns)
- **`iDEA/methods/exact_green.py`**
  - `green_function(s, state, grid=None, krylov_dim=…, tol=…, GPU=False)` returns an object compatible with `QuasiParticleState` (`G`, `G_iw` per spin, `mu`, `grid`, `energy`), so `observables.density`, `observables.density_matrix` and `mbpt.total_energy` work unchanged.
  - Helpers build Dyson amplitudes and $\hat\psi^\dagger(x)|\Psi\rangle$ sources consistent with iDEA's spin-string ordering.
  - Also `self_energy(...)` and `spectral_function(s, ..., omega, eta)`.
- **`iDEA/reverse_engineering.py`:** add an analytic two-electron singlet KS potential alongside `reverse`.
- **`tests/test_exact_green.py`:** the validation list of Sec. 5.8.
- **`tutorial/example_exact_green.ipynb`:** a worked example.
- **Research scripts in `paper/scripts/`,** following the CPC draft on branch `ccr-6064c011-6imjt7`: nearsightedness diagnostics (M1–M4), the truncation and Mott-pole tests, and `make_figures.py`, which regenerates every figure and number. Project-specific analysis stays out of the library.

### 6.3 Computational cost

| Step | Size at $N_x = 130$ | Cost |
|---|---|---|
| $N=2$ ground state | dim 1.7 × 10⁴ | seconds |
| $N+1 = 3$ sector Hamiltonian | dim 2.2 × 10⁶, ≈ 8 × 10⁷ non-zeros (13-point stencil), ≈ 1 GB CSR | built once per system |
| Addition-sector Lanczos | $N_x$ sources × 200–400 steps | minutes on the RTX 4090; about an hour on a 4-core CPU |
| Dyson inversions | $N_\omega \approx 80$ dense $N_x\times N_x$ inversions | negligible |
| *GW* (PR #3) | — | about a minute per system on a 4-core CPU (PR #3 timing for a 121-point system) |

A full campaign of about 300 exact-$G$ runs, including convergence checks in grid, box, Krylov dimension and grid tolerance, fits in **under 50 GPU-hours**.

---

## 7. Work packages and timeline

| WP | Content | Deliverable / milestone |
|---|---|---|
| **WP0** Consolidation | Merge PR #3 and the branches of Sec. 6.1 into a research branch. Validate KS-SCE. Recover the 2019 parameters and reproduce Ref. [1] with PR #3 *GW* and exact KS. | **MS1:** 2019 figures reproduced with spinful electrons |
| **WP1** Exact Green's function | `exact_green.py`; removal Lehmann sum and addition Lanczos; tests 1–8 | **MS2:** validation suite passes; Galitskii–Migdal energy = exact energy to ≤ 10⁻⁸ Ha |
| **WP2** Diagnostics | M1–M4; partitions and localised orbitals; truncation and pole-correction tests | **MS3 (go/no-go):** first S1/S2 maps. Does $\Sigma_{AB}$ collapse onto the dimer form? Yes → letter-style framing; no → detailed PRB mapping |
| **WP3** Production | S1–S3 scans (singlet and triplet, $d$, $\Delta$, $\lambda$), *GW* variants, KS and KS-SCE, convergence | **MS4:** converged dataset |
| **WP4** Consequences and extensions | RQ5. Optional: exact imaginary-axis spectral potential [20–22]; thesis chapter 6 *GW* orbital-swapping and derivative-discontinuity material | **MS5:** results for RQ5 |
| **WP5** Paper | Manuscript, `paper/` with `make_figures.py`, data release, README "Papers You Can Reproduce" entry | **MS6:** submission-ready manuscript |

| | M1 | M2 | M3 | M4 | M5 | M6 |
|---|---|---|---|---|---|---|
| WP0 | ■ | | | | | |
| WP1 | ■ | ■ | | | | |
| WP2 | | ■ | ■ | | | |
| WP3 | | | ■ | ■ | | |
| WP4 | | | | ■ | ■ | |
| WP5 | | | | | ■ | ■ |

---

## 8. Expected results and planned figures

1. **Systems and exact KS potentials.** Densities, pair densities and exact $v_\mathrm{xc}$ for S1–S3: the step in asymmetric cases, the midpoint peak in the symmetric singlet, and its absence for the triplet.
2. **Self-energy maps.** $\Sigma_{xc}(x,x';i\nu)$ at three frequencies, exact versus self-consistent *GW* versus Hartree–Fock, for S1, S2 and S3.
3. **Range versus separation.** $\Vert\Sigma_{AB}(i\nu)\Vert$ against $d$ for the singlet, the triplet and asymmetric cases at fixed $\nu$; the integrated weight (M4) against $d$; overlaid step or peak heights of $v_\mathrm{xc}$.
4. **Mott-pole collapse.** $\langle w_A|\Sigma(i\nu)|w_B\rangle$ rescaled by $t_\mathrm{eff}$ and $U_\mathrm{eff}$, collapsing onto $-(3/4)/(\tilde\nu^2+9)$, with the asymmetry crossover shown as an inset.
5. **Consequences.** Spectral functions $A(x,\omega)$ and IP, EA and gap from the exact, *GW*, truncated and pole-corrected self-energies.

A positive outcome gives a sharp, quotable statement: *nearsightedness of the self-energy is protected by non-degeneracy and spin, and fails in a characteristic low-frequency, separation-independent way in the Mott/Wigner regime.* A negative outcome, with nearsightedness surviving, would be equally publishable. It would strengthen the case of Ref. [1] for MBPT-based local approximations at all correlation strengths.

---

## 9. Novelty and positioning

| Prior work | What it established | What is new here |
|---|---|---|
| Wetherell *et al.* 2019 [1] | Approximate $\Sigma$ (HF, *GW*) nearsighted; exact KS step | **Exact** $\Sigma$; spin; strong coupling; frequency-resolved nonlocality |
| Romaniello, Guyot & Reining 2009 [17]; Vanzini, Reining & Gatti 2018 [22] | Exact $\Sigma$ and spectral potential of the Hubbard dimer | Real-space continuum; separation and coupling scans; Coulomb tails and Wigner structure |
| Exact *GW* and vertex benchmarks, extended Hubbard model (2026) [25]; Förster & Bruneval 2024 [26] | Lattice benchmarks; vertex cancellation questioned | Exact continuum self-energies against the same *GW* hierarchy |
| Caruso *et al.* 2013 [15]; Hellgren *et al.* 2015 [16]; Loos, Romaniello & Berger 2018 [18] | Failures of *GW* in energy and density for stretched or strongly correlated systems | *Nonlocality* of the exact versus approximate $\Sigma$, and its consequences |
| Godby, Schlüter & Sham 1988 [39] | Real-space nonlocality of the *GW* $\Sigma$ in semiconductors; xc potentials from $\Sigma$ | Nonlocality of the *exact* $\Sigma$ at strong coupling |
| Kohn 1996 [6]; Prodan & Kohn 2005 [7] | Nearsightedness of the density | Nearsightedness of the self-energy at strong coupling |
| Buijse *et al.* 1989 [11]; Helbig *et al.* 2009 [12]; Hodgson *et al.* 2016–17 [9, 10]; Kraisler *et al.* 2021 [8] | Steps and peaks of the exact KS potential | Side-by-side with the exact $\Sigma$ in the same systems |
| Malet & Gori-Giorgi 2012 [28]; Mirtschink *et al.* 2013 [36]; Giarrusso & Agostini 2024 [40] | Strong-coupling KS functionals reproduce steps and peaks | Far-sighted KS-SCE versus the exact $\Sigma$ |

We found no exact real-space self-energy of a strongly correlated continuum system in the literature.

---

## 10. Risks and mitigations

| Risk | Mitigation |
|---|---|
| Round-off in the Dyson inversion at high frequency | $G$ to near machine precision; checks of the Galitskii–Migdal energy, density and static limit; reduced natural-orbital basis; Krylov-dimension convergence |
| The N+1 sector limits $N_x$ to about 150 | Compact model parameters; 13-point stencil; GPU; block Lanczos |
| Long-range Coulomb tails blur "nearsightedness" | Analyse $\Sigma_{xc}$ (Hartree removed); short-range interaction control runs |
| Self-consistent *GW* fails to converge or has multiple solutions at strong coupling | Damping and G₀W₀/GW₀ fallbacks; report as a result |
| KS-SCE module defects | WP0 validation; otherwise evaluate the SCE functional non-self-consistently on exact densities |
| Scope creep (spectral potential, three electrons) | Kept as optional WP4 extensions after MS3 |

---

## 11. Outputs, dissemination and follow-ups

**Outputs**
- **Paper:** *Physical Review B*, or *Physical Review Letters* plus a long PRB if H2 holds cleanly.
- **Code:** an exact-Green's-function module with tests and a tutorial in iDEA.
- **Reproducibility:** a `paper/` directory with `make_figures.py`, data, and a README "Papers You Can Reproduce" entry.
- **Software paper:** the draft CPC paper on branch `ccr-6064c011-6imjt7` documents the code base; the new module would feature in the next release.

**Follow-ups**
1. **Exact continuum spectral potentials.** Connector theory [20–22] is known exactly only for the dimer. A second paper would ask whether the local, frequency-dependent spectral potential inherits KS-like far-sightedness or $\Sigma$-like nearsightedness, and compare with the dynamical xc field formalism [27].
2. **Machine learning.** Nearsighted targets are learnable by local models. This connects to the hypersensitivity of density-based learning [41]: learning $\Sigma$, or a nearsighted correlation potential, may be better conditioned than learning $v_\mathrm{xc}$.
3. **Nearsighted generalised-KS functionals.** Use the exact $\Sigma$ to design local correlation potentials on top of unrestricted Hartree–Fock, in the framework of Hodgson & Wetherell [14].
4. **Three-electron removal spectra and real-time extensions** (S4).

**Suggested collaborators**
- M. J. P. Hodgson and R. W. Godby (York): Ref. [1], exact KS, *GW*.
- M. Gatti and L. Reining (LSI, École Polytechnique): spectral potential, connector theory, vertex.
- P. Romaniello and J. A. Berger (Toulouse): exact model self-energies, 1D Wigner crystals [18, 23].
- D. P. Kooi and P. Gori-Giorgi: authors of iDEA's KS-SCE module and the SCE theory.

---

## 12. Alternatives considered
- **"The exchange blind spot."** Exact singlet–triplet exchange $J$ of 1D Wigner molecules (spin qubits, moiré) against DFT, KS-SCE and *GW*, including exact time-dependent spin-KS potentials for an exchange gate. Spin effects enter the strong-coupling functional only at order $e^{-\sqrt\lambda}$ [42]. A strong candidate for a follow-up.
- **Machine-learned non-adiabatic xc potentials** from exact 2e dynamics. Crowded: a 2026 preprint already learns memory-dependent correlation potentials (arXiv:2609.04527).
- **Other frontier ideas:** real-time adiabatic KS-SCE against exact time-dependent KS potentials; exact ensemble DFT in the Wigner regime; exact thermal (Mermin) DFT in the continuum; KS versus many-body localisation tensor ("quantum geometry").

---

## Appendix A. Hubbard-dimer verification
We built the 16-dimensional Fock space of the two-site Hubbard model with Jordan–Wigner operators and took the $N=2$ ground state. We formed the exact spin-up $G_{ij}(z)$ from the Lehmann sums over the $N\pm1$ sectors and computed $\Sigma = G_0^{-1}-G^{-1}$ with $G_0$ non-interacting.

- **Bonding-channel pole.** Fitting $\Sigma_b(\omega) = A + R/(\omega-\mu-p)$ at three real frequencies gives $A = U/2$, $p = 3t$ and $R = U^2/4$ to ten digits for $t/U$ = 0.05, 0.1 and 0.2.
- **Inter-site element.** $\Sigma_{12}(\mu+i\nu)$ agrees with $-(3tU^2/4)/(\nu^2+9t^2)$ at $\nu = 0$ and $\nu = 10t$ to all printed digits for $0.01 \le t/U \le 0.2$, and is real to about $10^{-15}$.
- **Integrated weight.** It equals $-0.78541U^2$ for every $t$, against $-\pi U^2/4 = -0.78540U^2$ (quadrature error about $2\times10^{-5}$).

The asymmetric-dimer table in Sec. 2.3 was produced the same way, with $\mu = [E_0^{N+1}-E_0^{N-1}]/2$. The script will be added to `paper/scripts/` in WP2.

## Appendix B. Map onto the iDEA code base
- **Exact solver:** `iDEA/methods/interacting.py` (`hamiltonian`, `solve`, `GPU=True`).
- **Observables:** `iDEA/observables.py` (`density`, `density_matrix`; `kinetic_energy` and `interaction_energy` on feat/45; `total_spin_squared` on feat/46).
- **Mean-field methods:** `iDEA/methods/hartree_fock.py`, `hybrid.py`, `lda.py`, `kssce.py`.
- ***GW*:** `iDEA/methods/mbpt.py` (PR #3): `SpaceTimeGrid`, `non_interacting_green_function`, `exchange_self_energy`, `correlation_self_energy`, `dyson_equation`, `quasiparticle_states`, `total_energy`, `solve`.
- **Inversion:** `iDEA/reverse_engineering.py` (`reverse`).
- **New:** `iDEA/methods/exact_green.py`, `tests/test_exact_green.py`, `tutorial/example_exact_green.ipynb`, `paper/`.

---

## References
Entries marked † were checked only against abstracts or listings; their bibliographic details are to be confirmed.

1. J. Wetherell, M. J. P. Hodgson, L. Talirz, R. W. Godby, *Advantageous nearsightedness of many-body perturbation theory contrasted with Kohn–Sham density functional theory*, Phys. Rev. B **99**, 045129 (2019). arXiv:1812.02661.
2. S. Pecker, F. Kuemmeth, A. Secchi, M. Rontani, D. C. Ralph, P. L. McEuen, S. Ilani, *Observation and spectroscopy of a two-electron Wigner molecule in an ultraclean carbon nanotube*, Nat. Phys. **9**, 576 (2013). doi:10.1038/nphys2692.
3. I. Shapir, A. Hamo, S. Pecker, C. P. Moca, Ö. Legeza, G. Zarand, S. Ilani, *Imaging the electronic Wigner crystal in one dimension*, Science **364**, 870 (2019). arXiv:1803.08523.
4. J. Corrigan *et al.*, *Coherent control and spectroscopy of a semiconductor quantum dot Wigner molecule*, Phys. Rev. Lett. **127**, 127701 (2021). arXiv:2009.13572.
5. H. Li *et al.*, *Wigner molecular crystals from multielectron moiré artificial atoms*, Science **385**, 86 (2024). doi:10.1126/science.adk1348.
6. W. Kohn, *Density functional and density matrix method scaling linearly with the number of atoms*, Phys. Rev. Lett. **76**, 3168 (1996).
7. E. Prodan, W. Kohn, *Nearsightedness of electronic matter*, Proc. Natl. Acad. Sci. USA **102**, 11635 (2005). arXiv:cond-mat/0503124.
8. E. Kraisler, M. J. P. Hodgson, E. K. U. Gross, *From Kohn–Sham to many-electron energies via step structures in the exchange–correlation potential*, J. Chem. Theory Comput. **17**, 1390 (2021). doi:10.1021/acs.jctc.0c01093.
9. M. J. P. Hodgson, J. D. Ramsden, R. W. Godby, *Origin of static and dynamic steps in exact Kohn–Sham potentials*, Phys. Rev. B **93**, 155146 (2016). arXiv:1603.04816.
10. M. J. P. Hodgson, E. Kraisler, A. Schild, E. K. U. Gross, *How interatomic steps in the exact Kohn–Sham potential relate to derivative discontinuities of the energy*, J. Phys. Chem. Lett. **8**, 5974 (2017). arXiv:1706.00586.
11. M. A. Buijse, E. J. Baerends, J. G. Snijders, *Analysis of correlation in terms of exact local potentials: applications to two-electron systems*, Phys. Rev. A **40**, 4190 (1989).
12. N. Helbig, I. V. Tokatly, A. Rubio, *Exact Kohn–Sham potential of strongly correlated finite systems*, J. Chem. Phys. **131**, 224105 (2009). arXiv:0908.0710.
13. A. R. Elmaslmane, J. Wetherell, M. J. P. Hodgson, K. P. McKenna, R. W. Godby, *Accuracy of electron densities obtained via Koopmans-compliant hybrid functionals*, Phys. Rev. Materials **2**, 040801(R) (2018). arXiv:1803.11076.
14. M. J. P. Hodgson, J. Wetherell, *Accurate real-time evolution of electron densities and ground-state properties from generalized Kohn–Sham theory*, Phys. Rev. A **101**, 032502 (2020). arXiv:1909.02510.
15. F. Caruso, D. R. Rohr, M. Hellgren, X. Ren, P. Rinke, A. Rubio, M. Scheffler, *Bond breaking and bond formation: how electron correlation is captured in many-body perturbation theory and density-functional theory*, Phys. Rev. Lett. **110**, 146403 (2013). arXiv:1210.8300.
16. M. Hellgren, F. Caruso, D. R. Rohr, X. Ren, A. Rubio, M. Scheffler, P. Rinke, *Static correlation and electron localization in molecular dimers from the self-consistent RPA and GW approximation*, Phys. Rev. B **91**, 165110 (2015). arXiv:1412.7507.
17. P. Romaniello, S. Guyot, L. Reining, *The self-energy beyond GW: local and nonlocal vertex corrections*, J. Chem. Phys. **131**, 154111 (2009).
18. P.-F. Loos, P. Romaniello, J. A. Berger, *Green functions and self-consistency: insights from the spherium model*, J. Chem. Theory Comput. **14** (2018)†. arXiv:1803.04234.
19. A. Georges, G. Kotliar, W. Krauth, M. J. Rozenberg, *Dynamical mean-field theory of strongly correlated fermion systems and the limit of infinite dimensions*, Rev. Mod. Phys. **68**, 13 (1996).
20. M. Gatti, V. Olevano, L. Reining, I. V. Tokatly, *Transforming nonlocality into a frequency dependence: a shortcut to spectroscopy*, Phys. Rev. Lett. **99**, 057401 (2007). arXiv:0705.1887.
21. M. Vanzini, A. Aouina, M. Panholzer, M. Gatti, L. Reining, *Connector theory for reusing model results to determine materials properties*, npj Comput. Mater. **8**, 98 (2022). doi:10.1038/s41524-022-00762-2. See also M. Vanzini, L. Reining, M. Gatti, *Dynamical local connector approximation for electron addition and removal spectra*, arXiv:1708.02450.
22. M. Vanzini, L. Reining, M. Gatti, *Spectroscopy of the Hubbard dimer: the spectral potential*, Eur. Phys. J. B **91**, 192 (2018). doi:10.1140/epjb/e2018-90277-3.
23. D. Lagasco, J. A. Berger, *Electronic and magnetic properties of small one-dimensional Wigner crystals from an ab initio approach*, arXiv:2607.26858 (2026).
24. D. Lagasco, Z. Ristivojevic, *The exchange coupling of a Wigner dimer*, arXiv:2512.02678 (2025).
25. *Accidental accuracy and vertex corrections in GW: exact benchmarks for the extended Hubbard model*, arXiv:2608.24838 (2026)† (author list to be confirmed).
26. A. Förster, F. Bruneval, *Why does the GW approximation give accurate quasiparticle energies? The cancellation of vertex corrections quantified*, arXiv:2410.17843 (2024)† (journal to be confirmed).
27. F. Aryasetiawan, *Density-functional Green function theory: dynamical exchange-correlation field in lieu of self-energy*, arXiv:2505.19852 (2025).
28. F. Malet, P. Gori-Giorgi, *Strong correlation in Kohn–Sham density functional theory*, Phys. Rev. Lett. **109**, 246402 (2012).
29. F. Malet, A. Mirtschink, J. C. Cremon, S. M. Reimann, P. Gori-Giorgi, *Kohn–Sham density functional theory for quantum wires in arbitrary correlation regimes*, Phys. Rev. B **87**, 115146 (2013).
30. A. Marie, D. P. Kooi, J. Grossi, M. Seidl, Z. Musslimani, K. J. H. Giesbertz, P. Gori-Giorgi, *Real space Mott–Anderson electron localization with long-range interactions*, Phys. Rev. Research **4**, 043192 (2022).
31. H. N. Rojas, R. W. Godby, R. J. Needs, *Space-time method for ab initio calculations of self-energies and dielectric response functions of solids*, Phys. Rev. Lett. **74**, 1827 (1995).
32. M. M. Rieger, L. Steinbeck, I. D. White, H. N. Rojas, R. W. Godby, *The GW space-time method for the self-energy of large systems*, Comput. Phys. Commun. **117**, 211 (1999).
33. J. Kaye, K. Chen, O. Parcollet, *Discrete Lehmann representation of imaginary time Green's functions*, Phys. Rev. B **105**, 235115 (2022). arXiv:2107.13094.
34. V. M. Galitskii, A. B. Migdal, *Application of quantum field theory methods to the many body problem*, Sov. Phys. JETP **7**, 96 (1958).
35. J. P. Perdew, A. Savin, K. Burke, *Escaping the symmetry dilemma through a pair-density interpretation of spin-density functional theory*, Phys. Rev. A **51**, 4531 (1995).
36. A. Mirtschink, M. Seidl, P. Gori-Giorgi, *Derivative discontinuity in the strong-interaction limit of density-functional theory*, Phys. Rev. Lett. **111**, 126402 (2013).
37. W. Nelson, P. Bokes, P. Rinke, R. W. Godby, *Self-interaction in Green's-function theory of the hydrogen atom*, Phys. Rev. A **75**, 032505 (2007). arXiv:cond-mat/0701592.
38. J. Wetherell, M. J. P. Hodgson, R. W. Godby, *GW self-screening error and its correction using a local density functional*, Phys. Rev. B **97**, 121102(R) (2018). arXiv:1802.06826.
39. R. W. Godby, M. Schlüter, L. J. Sham, *Self-energy operators and exchange-correlation potentials in semiconductors*, Phys. Rev. B **37**, 10159 (1988).
40. S. Giarrusso, F. Agostini, *Modelling the Kohn–Sham potential for molecular dissociation with orbital-independent functionals: a proof of principle*, arXiv:2408.06038 (2024).
41. L. Arnstein, J. Wetherell, … Lawrence, P. J. Hasnip, M. J. P. Hodgson (initials to be confirmed), *Addressing hypersensitivity in DFT for reliable machine learning*, npj Comput. Mater. **12**, 338 (2026)†. arXiv:2602.16618.
42. J. Grossi, D. P. Kooi, K. J. H. Giesbertz, M. Seidl, A. J. Cohen, P. Mori-Sánchez, P. Gori-Giorgi, *Fermionic statistics in the strongly correlated limit of density functional theory*, J. Chem. Theory Comput. **13**, 6089 (2017). doi:10.1021/acs.jctc.7b00998.
