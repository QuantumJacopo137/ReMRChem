# Last activity — SWORD development diary

Last updated: **12 September 2026**

This is my reconstruction of how the four-electron SWORD code evolved, why I made each recent change, what the verbose outputs actually showed, and what is still only a hypothesis. I am writing it in a slightly informal way on purpose: this should be the file I can read after some weeks away and understand not only *what* is in the code, but also *why it ended up there*.

## 12 September — common Dirac/SWORD test driver and handoff guide

Added [`main_Dirac_vs_SWORD.py`](main_Dirac_vs_SWORD.py) as the common launch program for Dirac or SWORD with 1, 2 or 4 electrons. The six routes call the existing one-electron, two-electron and `Lazy_4_el` solvers. Input parsing, validation, MRA setup, nuclear potentials, initial states, solver dispatch and output are separate commented functions. Importing the file starts no calculation; `--help` and `--dry-run` work with the Python standard library alone.

The driver carries forward the active controls from `test_SWORD.py` and `test_DIRAC.py`: nuclear charge and position, light speed, precision/order/box/depth, derivatives, all five nuclear-potential models, potential read/compute/save, full-orbital restarts, legacy four-electron Weyl seeds, saving guesses and final orbitals, and optional coarse Dirac warm-up. The four-electron screened 2s/1s guess ordering is preserved. Generated Weyl seeds are converted in memory without compulsory disk round trips. A full restart uses its loaded state and defaults to no warm-up, fixing the old one-electron branch's uninitialized `init_guess` problem.

Each run gets a directory with the launch command, resolved parameters, Python output log and separate setup/warm-up/SCF/save/total wall times. Four-electron runs also save the returned complex Fock matrix. Tree saving is optional, and existing run logs are not overwritten. The default is a fresh Xe four-electron SWORD calculation with a computed Fermi potential; native potential loading is explicit. Automatic boxes include the nuclear displacement. Gaussian projection handles the finite nuclear-centre limit, and the homogeneous-sphere input is correctly named `--mass-number` rather than confusing it with a radius.

A small supporting change to `two_electron.coulomb_gs_2e()` adds the optional `niter=100` argument needed for bounded Dirac runs and warm-up. Its propagation formula is unchanged. The loop now evaluates the final state before returning, stops on the original orbital threshold or the update limit, and no longer retains the defective `compute_last_energy` flag. Existing calls that supply the original seven arguments remain valid.

The driver makes existing API limits visible: four-electron `thr` is internal (`50*prec` for maximum spinor change and `prec/10` for relative trace), so a user-supplied `--threshold` is rejected for 4e. Four-electron derivatives remain internal apart from guess construction; 1e/2e SWORD keeps the shared helpers' BS propagation and ABGV balance. These conventions are recorded in the run settings. Solver return and process success are not treated as proof of convergence.

[`MAIN_DRIVER_GUIDE.md`](MAIN_DRIVER_GUIDE.md) gives commands for all six combinations, potential options and units, checkpoint naming, old-script setting mappings, output interpretation, restart examples, environment requirements and instructions for handing a run to someone else. It also explains that the old commented plotting experiments are outside this launch interface and that the open four-electron numerical issues below remain unvalidated.

Validation: 13 standard-library tests in [`tests/test_main_driver.py`](tests/test_main_driver.py) pass on Python 3.7.3. They exercise all six orchestration routes with stand-ins, transfer of the warm-up result, all potential models, fresh and loaded guesses, complete/incomplete checkpoints, output files, invalid settings, dry-run isolation and missing-runtime errors. Syntax and CLI help checks pass. The actual two-electron Dirac routine also passed isolated converged, iteration-limited and zero-update checks in `/tmp/check_two_electron_dirac.py`, including final energies and KTRS. Scientific dependencies are unavailable in this environment, so these are control-flow checks, not physical convergence or performance benchmarks.

## 12 September — convert the two-electron ground-state routine to SWORD

`two_electron.coulomb_gs_2e_SWORD()` was still a copy of the Dirac routine: it convolved a full four-component potential action and then applied the shifted Dirac Hamiltonian. It now follows the propagation/balance/energy order in `one_electron.gs_SWORD_1e()` and reuses `one_electron.SWORD_propagator()` and `one_electron.balance_Dirac_spinor()`.

The call is now:

```python
spinorb1, spinorb2 = twoel.coulomb_gs_2e_SWORD(
    spinorb1, potential, mra, prec, thr, derivative,
    output_file="output", niter=100,
)
```

The original six positional arguments still work. The two-component class receives the supplied MRA and the starting spinor's speed of light before entering the shared helpers.

The steps are commented in the routine:

1. Build the initial partner Coulomb field from the full spinor density. The existing two-electron Kramers-pair reduction is retained: after self direct/exchange cancellation, the effective local potential is the nuclear attraction plus the partner's Coulomb field. The orbital energy includes rest energy, and the total remains `2*eps - JmK.real`.
2. Propagate only the independent left-handed Weyl component through the one-electron SWORD helper, supplying the full nuclear-plus-Coulomb potential action and the current orbital energy.
3. Normalize the propagated spinor, apply the nuclear potential and the **same frozen Coulomb field** to this new spinor, and reconstruct the right-handed component through the shared balance helper. This uses the new potential action, rather than an action on the old spinor. The field is not rebuilt from the intermediate state that contains only L. Crop and normalize the complete balanced spinor as in the one-electron routine.
4. Rebuild the Coulomb field from the balanced full density and recompute the Hamiltonian, potential actions and energies. These values are used both for the next propagation and for the final report, so the reported energy belongs to the returned state.
5. Check the full-spinor change against the existing two-electron `thr`, then generate the final partner using KTRS. This conversion retains the two-electron orbital threshold; it does not add the one-electron relative-energy stopping condition.

Wall times use `time.time()` and the same printed labels as `gs_SWORD_1e`: `Propagation time`, `Balancing time`, `Energy calculation time`, and the `TOT TIME` breakdown. The initial energy calculation is also timed. Field construction is included in the energy stage; applying the frozen field to the propagated spinor is included in balancing.

Two copied control-flow/reporting problems were fixed at the same time. `output_file` is now an explicit optional argument instead of an undefined name, and final energies are labelled SWORD. The persistent `compute_last_energy` flag is gone: it could keep the old loop running indefinitely after the orbital error fell below `prec`. A bounded loop now reports convergence or exhaustion of `niter`; even `niter=0` returns the input state and its calculated energies. The existing Kutzelnigg diagnostic is retained and uses the refreshed final-state actions. The unused classical-energy intermediates were removed.

The one-electron helpers themselves were not changed. Their derivative choices therefore remain in effect: propagation uses the helper's `sigma_p` default (BS), balance passes ABGV to `apply_R`, and the supplied `derivative` is used for the energy Hamiltonian. This matches the current one-electron reference but is not a new validation of those derivative choices.

Validation: Python syntax checks passed. Five isolated regression checks in `/tmp/check_two_electron_sword.py` passed, exercising the actual solver body with finite-dimensional stand-ins: updated Coulomb actions and energies, application of the frozen field during balance, final-state energy bookkeeping, convergence below `prec`, the iteration limit, zero iterations, the original six-argument call, the `thr` boundary, KTRS return and timing labels. These checks validate the SCF bookkeeping, **not the numerical SWORD kernels**. The available Python lacks VAMPyR, NumPy and SciPy, so no physical two-electron convergence run or wall-time benchmark was performed. A comparison with `coulomb_gs_2e()` in the scientific runtime is still required before making accuracy or speed claims.

## Four-electron checkpoint recorded on 4 September

The four-electron history and conclusions below are retained from the earlier diary update; the two-electron conversion above does not validate or resolve those open four-electron experiments.

The present reference calculation is Xe with four occupied spinors arranged as two Kramers pairs, a Fermi–Dirac nuclear model, B-spline derivatives, `prec = 1e-7`, polynomial order 11 and an automatically selected box of 1 a.u. The driver first runs two deliberately coarse four-component iterations at `1e-3`, then gives that smoothed state to SWORD at `1e-7`:

```python
Dirac_array, Fock_matrix = lazy4el.scf_4e_4c(
    Dirac_array, V_tree, mra, 1e-3, auto_save=False, max_iter=2
)
Dirac_array, Fock_matrix = lazy4el.scf_4el(
    Dirac_array, V_tree, mra, prec, auto_save=False
)
```

This two-step warm-up was introduced because the analytic/nonrelativistic starting guess contains high-frequency noise. SWORD contains derivatives in places where this noise can be amplified immediately, so the cheap coarse Dirac iterations act as a numerical low-pass filter before the accurate calculation.

## A warning about reconstructing the history

The output files in `outputs/` are not committed together with a frozen version of the source. Therefore the history below comes from four things together:

1. the Git commits and their messages;
2. the output timestamps and filenames;
3. the diagnostic signatures printed inside each output;
4. the current uncommitted code and the discussions in the previous chats.

When an association is certain I state it normally. When a filename strongly suggests a change but the exact source snapshot no longer exists, I call it **inferred**. This distinction is important for the article: a filename is useful evidence, but it is not a reproducible code revision.

## The main idea I was trying to implement

The “lazy” SWORD strategy stores a complete four-component orbital, but propagates only the independent left-handed Weyl component. The opposite chirality is reconstructed using the Dirac–Fock equation. Schematically, for orbital (i),

\[
R_i = \frac{F_{ii}}{c^2}L_i
      -\frac{1}{c^2}V_i L_i
      -\frac{1}{c}\,\boldsymbol{\sigma}\cdot\mathbf p\,L_i
      +\frac{1}{c^2}\sum_{j\ne i}F_{ji}L_j,
\]

up to the sign convention used for the Pauli matrices in this repository. The attraction of the method is obvious: the expensive Helmholtz propagation should act on two independent Weyl components instead of four Dirac components. The danger is also now clear: the reconstructed (R) is an algebraic sum containing a derivative and the nuclear-potential cusp. It does not receive the same final smoothing as the fully propagated four-component orbital.

The implementation is therefore a competition between two facts:

- SWORD really does save work in the propagation;
- an unnecessarily refined reconstructed (R), additional SCF iterations, or grid unions during complex arithmetic can consume the whole gain.

That is the thread connecting almost every output below.

---

## January–July 2026: how the present algorithm was reached

### 27 January — one electron first worked

Commit `2b11386` was the first working one-electron result. Kramers time-reversal symmetry (KTRS) still needed work, so this was not yet a four-electron algorithm. The important result was that the Weyl propagation itself could reproduce the one-electron solution.

### 3 February — the two-component Weyl spinor worked up to Hg

Commit `ae064d9` extended the one-electron result to heavy atoms. This was an early indication that the formulation was not limited to the weakly relativistic regime.

### 26 February — first four-electron attempt

Commit `695c4dd` introduced an SCF shaped like the four-electron calculation while trying to propagate only two components. It was not yet stable. At this point the missing part was not just a better convergence threshold: the relation between the propagated and reconstructed components had to be handled consistently through propagation, damping, orthonormalization and KTRS restoration.

### 19 March — the “lazy” 4c container

Commit `286d27d` changed the approach. Instead of maintaining a completely separate two-component state, I used the four-component orbital as a container for (L) and for (R) reconstructed from the Dirac–Fock equation. This made it possible to reuse the existing four-component Coulomb, exchange, overlap and Fock machinery.

This was the conceptual start of the current `Lazy_4_el.py`: propagate (L), balance to regenerate (R), then use the normal 4c infrastructure for the rest.

### 20–23 March — damping was not optional

Commit `1a2c2e3` added damping alternatives. The residual initially decreased and then spiralled out of control, accompanied by a rapidly growing kinetic contribution. The first suspicion was the helicity coupling between (L) and (R), which was reasonable: a derivative of a noisy adaptive function is exactly where high-frequency errors become expensive and dynamically dangerous.

Commit `755ff6d` produced the first converged four-electron calculation. The lesson at that time was: the basic equations were capable of convergence, but the damping factor controlled whether the update was usable.

### 26 March — exact propagator and an indexing bug in Löwdin

Commit `5a888bc` corrected the exact propagator and an index call in the Löwdin orthonormalization. After this, the calculation converged to the correct value. This is worth remembering because it established a pattern repeated later: an apparently “physical” instability can actually be an index/order problem in a basis transformation.

### 7 and 28 July — reference 4c method, layout and derivative choice

Commit `90419a9` established the working four-component Be calculation as the reference algorithm. Commit `4c7acc3` made the RKB layout closer to the paper and changed the default derivative to B-spline, because it appeared less noisy than the alternatives.

Commit `9ebb345` is the last committed checkpoint. It contains both SWORD and Dirac machinery and already describes the remaining problem accurately: the initial guess carries high-frequency noise, SWORD amplifies it, and two ultra-low-precision convolution steps can clean the input before the accurate iterations.

Everything from the Xe verbose campaign below is currently **uncommitted experimental work** relative to that commit.

---

## 22 August — the long baseline run showed a residual floor and late divergence

Output: [`outputs/Xe_SWORD_MW7.out`](outputs/Xe_SWORD_MW7.out)

This was the first long Xe baseline. It ran all 100 SWORD iterations. The maximum spinor change reached its best value, about

\[
1.13\times10^{-6},
\]

near iteration 30, but then it did not remain there. It grew again and reached roughly (7.3\times10^{-2}) at iteration 100. The final reported total energy was approximately

\[
E=-3672.681811398\ E_h,
\]

which is not the physical converged Xe result. Total elapsed time was about 32900 s.

The key conclusion was not “I need more iterations”. It was the opposite: after iteration 30 the algorithm had already reached its useful numerical floor and continuing the same map drove it away. This is why later work concentrated on the convergence criterion, the update representation and the unwanted tree refinement instead of simply increasing `max_iter`.

## 25 August — propagator cropping experiment, but no completed SWORD result

Output: [`outputs/Xe_SWORD_MW7_Changed_Prop_crop.out`](outputs/Xe_SWORD_MW7_Changed_Prop_crop.out)

This output contains the two coarse warm-up iterations and only the beginning of the accurate calculation. It therefore cannot demonstrate convergence or an energy improvement. It records the start of testing crops around the propagation path, but it must not be quoted as a successful run.

This distinction became important later: a smaller tree printed during one stage does not prove a smaller peak-memory calculation, and an output that stops before the first full SWORD iteration cannot validate a change.

## 26 August — verbose instrumentation located the growth around balance and Löwdin

Output: [`outputs/Xe_SWORD_MW7_verbose.out`](outputs/Xe_SWORD_MW7_verbose.out)

The verbose version followed the run into iteration 61. Numerically it reproduced the same story as the baseline: the residual reached about (1.13\times10^{-6}) around iteration 30, then had already grown to about (7.94\times10^{-5}) by iteration 60.

The node prints made the representation problem visible. A sampled independent spinor after the old Löwdin path could carry about 21888 total nodes with a component maximum around 4632. At this point I suspected that orthonormalization was forming a union of all component grids and returning that union without enough recompression.

The relevant code was the four-term Löwdin transformation:

```python
orthonormal_array[i] = (
    S_inv_sqrt[0, i] * spinor_array[0]
    + S_inv_sqrt[1, i] * spinor_array[1]
    + S_inv_sqrt[2, i] * spinor_array[2]
    + S_inv_sqrt[3, i] * spinor_array[3]
)
```

This expression is mathematically normal, but for adaptive `FunctionTree`s every addition can import the union of the participating grids. A tiny or even exactly zero algebraic contribution was still capable of carrying another tree's structure in the old complex scalar implementation.

## 26 August — cropping after Löwdin did reduce the stored state

Output: [`outputs/Xe_SWORD_MW7_verbose_crop_after_LOWDIN.out`](outputs/Xe_SWORD_MW7_verbose_crop_after_LOWDIN.out)

I added a crop on the two independent orbitals after Löwdin and regenerated their Kramers partners. The present helper contains the essential part:

```python
orthonormal_array[0].crop(prec)
orthonormal_array[2].crop(prec)
```

This experiment was incomplete, stopping near the fifth accurate iteration, so again it did not prove final convergence. It did prove that the persistent post-orthonormalization state could be reduced. In comparable prints the sampled spinor fell to about 13504 nodes, and in the later stabilized implementation to about 8704 before accounting for the logging artefact described below.

So this change **worked for stored-node reduction**. It did not by itself solve the SCF floor or the largest transient allocations.

## 27 August — removing crop from damping did not cure the instability

Output: [`outputs/Xe_SWORD_MW7_verbose_crop_after_LOWDIN_No_crop_Damp.out`](outputs/Xe_SWORD_MW7_verbose_crop_after_LOWDIN_No_crop_Damp.out)

This run completed 100 iterations. It again reached a best residual near (1.14\times10^{-6}) at iteration 30 and then diverged to about (6.51\times10^{-2}). Its final energy, (-3668.916689\ E_h), was wrong. The elapsed time dropped to roughly 26100 s, but a faster divergent run is not an algorithmic improvement.

The lesson was that repeated cropping inside damping was not the cause of the late instability. Damping still creates an old/new linear combination, so its additions can create a transient union even if the result is cropped afterwards. Removing the crop may save some work, but it does not prevent the union and does not fix the nonlinear SCF behaviour.

## 27 August — establish a fair four-component reference

Outputs:

- [`outputs/Xe_DIRAC_MW7_verbose.out`](outputs/Xe_DIRAC_MW7_verbose.out)
- [`outputs/Xe_DIRAC_MW7_verbose_cropped.out`](outputs/Xe_DIRAC_MW7_verbose_cropped.out)

The uncropped four-component reference converged in 24 loop passes to approximately

\[
E=-3716.456347554\ E_h,
\]

with a final maximum spinor difference (6.59\times10^{-7}) and elapsed time about 9798 s.

Adding sensible cropping to the reference gave

\[
E_{4c}=-3716.4563534683098\ E_h,
\]

with residual (6.34\times10^{-7}) and elapsed time about 7999 s. This cropped run is the present numerical and performance reference.

This comparison changed the memory diagnosis. The fully four-component method was not small because it represented fewer physical components; it was small because every component passed through the smoothing/convolution/update path and then was cropped. SWORD's (R) was reconstructed from a derivative and (V\Psi), so it retained finer structure even though only (L) was propagated.

## 27–28 August — bundled SWORD fixes reached the correct energy but missed convergence

Output: [`outputs/Xe_SWORD_MW7_verbose_crop_after_LOWDIN_rescale_Helmholtz_actual_zero_R_in_balance_correct_F_filter.out`](outputs/Xe_SWORD_MW7_verbose_crop_after_LOWDIN_rescale_Helmholtz_actual_zero_R_in_balance_correct_F_filter.out)

This filename records a bundle of changes:

- crop/recompress around the expensive Helmholtz source;
- rescale the convolved (L) instead of carrying unnecessary arithmetic;
- construct a genuinely empty (R) after propagation instead of multiplying an old (R) by zero;
- screen Fock couplings before adding them to the propagated/balanced state;
- keep only the two independent spinors, then regenerate Kramers partners.

After 30 iterations the result was

\[
E=-3716.456343712525\ E_h,
\]

with maximum residual (1.109\times10^{-6}). The energy was now only about (9.76\times10^{-6}\ E_h) above the cropped Dirac reference. This was the first strong evidence that the SWORD equations were giving the correct four-electron solution and that the remaining “non-convergence” was partly a criterion problem.

The run still took about 9032 s and was labelled non-converged at 30 iterations. It therefore did not yet provide a real wall-clock advantage over the 4c calculation.

### Why “actual zero R” mattered

The old pattern could be conceptually equivalent to

```python
zero_R = 0.0 * old_R
```

but in the adaptive representation this was not a zero-cost zero. The numerical coefficients became zero while the old tree topology could remain. Constructing a new `orbital2c`, calling `setZero()`, and inserting that object prevented the old (R) grid from being carried into the next stage.

## 28 August — using the same threshold inside balance changed essentially nothing

Output: [`outputs/Xe_SWORD_MW7_verbose_crop_after_LOWDIN_rescale_Helmholtz_actual_zero_R_in_balance_correct_F_filter_same_thr_balance.out`](outputs/Xe_SWORD_MW7_verbose_crop_after_LOWDIN_rescale_Helmholtz_actual_zero_R_in_balance_correct_F_filter_same_thr_balance.out)

This run was numerically almost identical:

\[
E=-3716.456343712541\ E_h,\qquad
r_{\max}=1.10922\times10^{-6}.
\]

Elapsed time was about 9016 s. The conclusion was useful even though the result did not improve: the small off-diagonal balance contributions being changed by that threshold were already below the effective resolution or removed by later cropping. Threshold consistency there was not the source of either the residual floor or the large (R) grids.

## 28 August — raw Fock-trace convergence criterion was too strict

Output: [`outputs/Xe_SWORD_MW7_verbose_F_trace_criteria.out`](outputs/Xe_SWORD_MW7_verbose_F_trace_criteria.out)

The previous stopping rule was effectively of the form

```python
if max_norm_diff < ... or max_energy_change < ...:
    converged
```

The `or` was dangerous. Orbital energies can appear stationary while the occupied subspace is still moving, or individual orbital differences can remain nonzero because the occupied orbitals rotate within an already converged subspace.

I replaced the orbital-energy check with the change in the occupied Fock trace and required it together with the residual. The first version used the raw quantity

\[
\Delta\operatorname{Tr}F =
2\Delta F_{00}+2\Delta F_{22}.
\]

At iteration 30, the run had the same correct energy and residual as above, but the raw trace change was (3.59\times10^{-6}), larger than the selected (10^{-8}) threshold. The calculation was therefore declared non-converged.

This criterion was too strict and not dimensionless. More importantly, applying the same raw formula to the final cropped Dirac run gives about (2.78\times10^{-5}), which is *larger* than the SWORD value. It would have rejected the reference calculation too. That made it an unfair criterion for claiming that one algorithm converges and the other does not.

## 3 September — relative trace and real diagonal energy in balance

Output: [`outputs/Xe_SWORD_MW7_verbose_F_trace_criteria_relative_and_real_E_balance.out`](outputs/Xe_SWORD_MW7_verbose_F_trace_criteria_relative_and_real_E_balance.out)

Two changes were tested together:

1. normalize the Fock-trace change;
2. use only `F_ii.real` as the energy entering the balance/propagator.

The current relative expression is

\[
\delta_{\mathrm{tr}}=
\frac{2(F_{00}^{n}-F_{00}^{n-1})+2(F_{22}^{n}-F_{22}^{n-1})}
{(F_{00}^{n}+F_{00}^{n-1})+(F_{22}^{n}+F_{22}^{n-1})}.
\]

With the combined condition

```python
if max_norm_diff < prec*50 and abs(rel_trace_diff) < prec/10:
    converged
```

the run stopped at iteration 25 with

\[
\delta_{\mathrm{tr}}=-8.65\times10^{-9},\qquad
r_{\max}=1.27\times10^{-6},
\]

and

\[
E_{\mathrm{SWORD}}=-3716.456343691445\ E_h.
\]

The difference from the cropped 4c reference is

\[
E_{\mathrm{SWORD}}-E_{4c}\approx+9.78\times10^{-6}\ E_h,
\]

or about (2.6\times10^{-9}) relative to the total energy. Elapsed time was about 7779 s, slightly below the 7999 s reference.

The important interpretation is that the successful early termination came mainly from the **relative trace criterion**, not from taking `.real`. At the same iteration in the earlier run, the relevant Fock elements and total energy were already almost the same. The imaginary part of (F_{ii}/c^2) is so small that its contribution normally lies below later crop thresholds.

This is the best completed SWORD result so far, but the speed claim needs care. SWORD and Dirac were not stopped using an identical subspace-residual criterion, so “SWORD is faster overall” is still preliminary rather than publication-ready.

## 4 September — changing the coupling threshold again had no measurable effect

Output: [`outputs/Xe_SWORD_MW7_verbose_F_trace_criteria_relative_and_real_E_balance_NEW_THR.out`](outputs/Xe_SWORD_MW7_verbose_F_trace_criteria_relative_and_real_E_balance_NEW_THR.out)

The outer Fock filter was tightened to roughly `prec/c**2` instead of the previous empirical scaling. The result was essentially byte-for-byte the same in the important observables: same convergence iteration, energy, residual and relative trace. Elapsed time was about 7798 s.

This was a negative but valuable result. The screened terms are physically and numerically too small to determine the current error. Spending time tuning this threshold further will not solve the node problem.

## 4 September — adaptive-R experiment hyper-refined instead of coarsening

Output: [`outputs/Xe_SWORD_MW7_verbose_F_trace_criteria_relative_and_real_E_balance_NEW_THR_ADAPTIVE_R.out`](outputs/Xe_SWORD_MW7_verbose_F_trace_criteria_relative_and_real_E_balance_NEW_THR_ADAPTIVE_R.out)

This output is incomplete. It stops during the first accurate SWORD iteration/Fock work after the coarse warm-up. It contains reconstructed (R) components with roughly 19144 nodes each and scale 14. This is exactly the signature that led to the out-of-memory concern: the supposedly adaptive balance was refining much more deeply than the Dirac reference.

The precise code snapshot is unavailable, so the mapping is **inferred from the filename and diagnostic prints**. The present experimental helper `apply_R_adaptive()` forms one adaptive three-term sum:

```python
return add_vector(
    [self, V_Psi, sigma_p_L],
    np.asarray([energy/c2, -1.0/c2, derivative_coefficient]),
    prec,
)
```

The strategy is reasonable: combine the already scaled contributions once instead of building several raw intermediate unions. But “adaptive addition” cannot undo a derivative or a potential term that has already been evaluated at excessive resolution. It prevents some avoidable unioning; it does not guarantee a coarse physical result.

The latest source also contains a more specific trap. `balance_Dirac_spinor_new()` sets

```python
work_prec = prec / 10.0
base_R = L[i].apply_R(..., work_prec)
```

while the old `apply_R()` internally evaluates

```python
sigma_p_weyl = self.sigma_p(prec / 10, derivative)
```

If these two are combined, the derivative is effectively requested at `prec/100`. For a production `prec=1e-7`, that is (10^{-9}), and a B-spline derivative near the Xe nuclear cusp can easily reach scale 14. This is a concrete explanation for how a change intended to reduce trees can instead exhaust memory.

## 4 September — `balance_AI` and the zero-aware multiplication checkpoint

Output: [`outputs/Xe_SWORD_MW7_verbose_balance_AI.out`](outputs/Xe_SWORD_MW7_verbose_balance_AI.out)

This latest output is also incomplete. It finishes the two coarse 4c warm-up iterations, whose reported energy is (-3714.215390400822\ E_h), then stops during the first accurate SWORD Coulomb/exchange/Fock work. That warm-up number is **not a final SWORD energy** and must not be compared directly with the converged values.

The output still shows (R) trees around 18056 nodes each at scale 14. There may be some improvement relative to the previous 19144-node experiment, but there is no completed iteration or peak resident-memory measurement, so it does not yet validate the new balance.

At this point I also corrected complex scalar multiplication. The old generic formula always computed all four products:

\[
(a+ib)(u+iv)=(au-bv)+i(av+bu).
\]

That is correct algebra, but disastrous for sparse adaptive grids when (a=0), (b=0), or the coefficient is purely real. Tests showed the pathology directly: multiplying trees with about 217 and 3985 nodes could leave roughly 4177 nodes in **both** real and imaginary outputs; even `0*b` could retain the 3985-node topology.

The current `complex_fcn.__mul__` dispatches the special cases:

```python
if real_coefficient == 0.0 and imag_coefficient == 0.0:
    output = complex_fcn()
    output.setZero()
    return output

if imag_coefficient == 0.0:
    return self.real_mul(real_coefficient)

if real_coefficient == 0.0:
    return self.imag_mul(imag_coefficient)
```

This is the correct representation-level fix. A real scalar now preserves the separate real/imaginary grids, a purely imaginary scalar swaps them without unioning, and an exact zero produces empty trees.

However, this only removes **pointless grid equalization**. A genuinely complex coefficient has both real and imaginary cross-terms and therefore genuinely requires both supports. Also, it cannot coarsen a `sigma_p(L)` or (V L) tree that was already created at scale 14. This is why zero-aware multiplication can be right and still not solve the whole memory problem.

---

## What the node counts finally taught me

The early printed totals were slightly misleading because the code printed all four Löwdin outputs before overwriting spinors 1 and 3 with Kramers transforms of the cropped independent spinors. The final live state was therefore smaller than that print implied, although the four raw outputs still existed transiently and could contribute to peak memory.

After correcting for this logging order, representative steady-state totals were approximately:

| state | SWORD nodes | cropped Dirac nodes | interpretation |
|---|---:|---:|---|
| all four live spinors | 33920 | 26752 | SWORD about 27% larger |
| independent (L) parts | 13184 | 13376 | SWORD is slightly smaller here |
| reconstructed (R) parts | 20736 | 13376 | SWORD is about 55% larger here |

The entire steady excess is therefore in (R), not in the propagated (L).

The first SWORD iteration was much worse: roughly 102144 live nodes versus about 53248 for Dirac, almost a factor of two. The SWORD stage totals then relaxed approximately as follows:

| iteration | after propagation, mostly (L) | after balance | after Löwdin/KTRS |
|---:|---:|---:|---:|
| 1 | 23808 | 92416 | 102144 |
| 2 | 13184 | 67776 | 87168 |
| 3 | 13184 | 46400 | 58752 |
| 4 | 13184 | 34560 | 38208 |
| 5 | 13184 | 33792 | 34176 |
| 8 onward | 13184 | 33536 | 33920 |

This is the cleanest answer to “why is SWORD using a ton of nodes while 4c uses few?”: propagation itself is not the problem. The jump happens in balance. At steady state balance creates almost all the persistent excess, and on the first iterations it creates a very large transient.

Within balance, the diagnostic sequence was also revealing. A typical independent (L) had about 3296 nodes. Its `sigma_p` temporary could reach about 10272 nodes. The algebraic RKB object before the derivative term had about 4000, while the final balanced spinor retained roughly 8256–8512 total nodes. The nuclear-potential term reached scale 14; (L) and `sigma_p(L)` more often stopped at scale 13. This points to the cusp-bearing (V\Psi) contribution as the deepest-grid importer, with the derivative adding a second expensive structure.

Cropping is not “failing” in the simple sense. `crop(prec)` removes coefficients below the requested error criterion; it does not promise a target node count. If scale-14 coefficients are above the tolerance, or if a preceding operation requests (10^{-9}) accuracy, cropping correctly keeps them. Repeatedly calling crop cannot compensate for constructing the source too accurately.

## Why the 4c reference remains smoother

The old SWORD propagator differentiates a rough potential product before convolution:

```python
Big_V_Psi_L = (
    -c2 * V_R
    - F_matrix[i, i].real * V_L
    - light_speed * V_L.sigma_p(prec/10)
)
```

The Helmholtz convolution smooths the resulting new (L), but the next balance reconstructs (R) directly from (L), (V\Psi), and `sigma_p(L)`. There is no equivalent final convolution on that reconstructed (R).

The four-component algorithm, by contrast, convolves the full update source and then applies its Hamiltonian/crop path to the full spinor. That operation ordering, not merely “4 components versus 2”, explains why its small components stop around scale 12 while SWORD's reconstructed ones can reach scale 14.

The next useful memory experiment is therefore not another arbitrary crop. It is one of these controlled alternatives:

1. evaluate every balance contribution at a common, justified working precision—no accidental `prec/100` derivative;
2. screen contributions using the norm of the **already scaled** term;
3. perform one adaptive sum, then one relative complex-norm crop;
4. test whether the cusp term can be represented/smoothed in a way consistent with the SWORD derivation before it enters (R);
5. measure peak RSS and native tree nodes stage by stage, not `pympler` object sizes.

`pympler.asizeof` only sees the Python wrapper reliably. Most MRCPP/VAMPyR storage lives in native objects, so a tiny Python size does not mean the function tree uses little memory.

---

## Fock matrix, “non-Hermiticity”, and what the logs really say

The best SWORD output contains values of the approximate size

\[
\operatorname{Im}F_{00}\approx3.0\times10^{-7},\qquad
\operatorname{Im}F_{22}\approx2.0\times10^{-6},
\]

and

\[
F_{03}\approx-5.4\times10^{-7}+4.77\times10^{-5}i.
\]

The final Dirac output has diagonal imaginary parts around (10^{-13}) and (F_{03}) around (10^{-12}).

The literal non-Hermiticity visible in the printed SWORD matrix is the imaginary diagonal: a Hermitian matrix must have real diagonal elements. A nonzero (F_{03}) by itself is **not** proof of non-Hermiticity. The code explicitly manufactures the opposite triangle by conjugation:

```python
F_matrix[0, 3] = ...
F_matrix[3, 0] = np.conj(F_matrix[0, 3])
F_matrix[1, 2] = -np.conj(F_matrix[0, 3])
F_matrix[2, 1] = -F_matrix[0, 3]
```

Therefore the printed off-diagonal matrix is Hermitian by construction; it does not test whether independently evaluated (\langle\psi_i|F|\psi_j\rangle) and (\langle\psi_j|F|\psi_i\rangle^*) agree.

Also, a nonzero (F_{03}) is not automatically a violation of Kramers symmetry. A Kramers-compatible occupied block has quaternion structure and may acquire such an element under a unitary rotation of the occupied pair. For an ideally spherical, consistently aligned (m_j) basis I would expect it to be close to zero, but the raw element is gauge dependent.

The SWORD diagnostics separate (F_{03}) into a kinetic part around

\[
-6.28\times10^{-6}+2.24\times10^{-4}i

\]

and a potential part around

\[
+5.74\times10^{-6}-1.76\times10^{-4}i.
\]

The reported residual is the cancellation between two larger quantities. This makes it very sensitive to different adaptive grids and derivative precision. It also explains why taking only the real diagonal energy inside balance did not remove (F_{03}): the off-diagonal element is generated by the full propagated orbital and by cancellation errors, not just by the tiny imaginary part of (F_{ii}).

### The more serious Fock indexing problem still present in the code

The repository defines

\[
F_{ij}=\langle\psi_i|\hat F|\psi_j\rangle.
\]

For an update of ket (\psi_i), the occupied-space expansion uses the **column** coefficient (F_{ji}), because

\[
\hat F|\psi_i\rangle
=\sum_j|\psi_j\rangle F_{ji}+|r_i\rangle.
\]

The current propagator and balance instead use the row:

```python
F_matrix[i, j] * L_spinor_array[j]
```

and the quadratic term uses

```python
F_matrix[i, j] * F_matrix[j, k] * L_spinor_array[k]
```

These are indistinguishable if the matrix is real symmetric, which is why the bug can remain hidden in the Dirac-like early iterations. Once complex off-diagonal elements appear, row and column differ by conjugation. For the final SWORD values, the row/column difference in each of the (F_{02}) and (F_{03}) coefficients is about (9.53\times10^{-5}i); the combined projected mismatch is of order (1.35\times10^{-4}), well above `prec` and the observed residual floor.

This is now the highest-priority correctness fix, but it is **not yet validated**. It must be changed consistently in:

- exact SWORD propagation;
- the quadratic Fock coupling;
- the balance correction;
- the corresponding occupied-space subtraction in the Dirac reference.

Changing only one routine would make the comparison less meaningful.

The direct test after the fix should be:

```python
S_ij = <psi_i | psi_j>
c = F[:, i]
projected_error = F[:, i] - S @ c
```

For an orthonormal occupied set, the column convention should make the occupied projection vanish up to numerical precision. I should also compute every (F_{ij}) independently and print

\[
\|F-F^\dagger\|_F,
\]

instead of using the conjugate-filled matrix to diagnose Hermiticity.

---

## Performance: where SWORD has an edge and where it loses it

Using steady iterations from the comparable verbose runs, the approximate timings were:

| stage | SWORD | cropped Dirac | SWORD difference |
|---|---:|---:|---:|
| complete steady iteration | 244 s | 295 s | about 17% faster |
| propagation | 93 s | 138 s | about 32% faster |
| potential/J–K work | 79 s | 105 s | about 25% faster |
| Fock construction | 49 s | 39 s | about 25% slower |
| explicit balance | 7.4 s | none | extra SWORD stage |

So the basic edge is real: SWORD's propagation is substantially cheaper. But the first SWORD iteration was around 789 s versus about 495 s for Dirac because of tree explosion, and the older SWORD convergence rule required 30 iterations versus roughly 24 for Dirac. Those two costs erased the propagation advantage.

With the relative trace criterion SWORD stopped after 25 iterations and total elapsed time became about 7779 s versus 7999 s for the cropped Dirac run. This is promising, but for an article I should not call it a decisive speed-up until both methods use the same physically meaningful convergence checks and peak memory is measured externally.

Another practical issue is that `F_matrix()` prints several expensive diagnostics even when `verbose=False`, including additional `sigma_p` evaluations. Those lines are useful during diagnosis, but they add work and temporary trees to every iteration. They should be guarded by `if verbose` for production benchmarks.

---

## What definitely worked, what did not, and what is still open

### Changes supported by completed outputs

- The two coarse 4c warm-up iterations suppress enough starting-guess noise for Xe SWORD to approach the physical solution.
- Cropping the independent orbitals after Löwdin materially reduced the persistent stored state.
- Creating a genuinely empty (R) avoided carrying the old (R) topology through a nominal zero.
- The bundled propagation/cropping/filter changes brought SWORD to within about (9.8\times10^{-6}\ E_h) of the cropped Dirac total energy.
- A relative occupied-trace check combined with the orbital-change threshold avoided continuing into the known late divergent regime.
- Zero-aware complex scalar multiplication fixes the demonstrated, representation-level real/imaginary grid-union bug.

### Changes that did not solve the target problem

- Running to 100 iterations did not push through the (10^{-6}) floor; it led to divergence.
- Removing the damping crop did not remove the late instability.
- Making the balance threshold equal to the surrounding threshold made essentially no numerical difference.
- Tightening the Fock coupling filter to `prec/c**2` made essentially no numerical difference.
- Using only `F_ii.real` in balance did not remove (F_{03}) or the diagonal imaginary residue.
- Post-operation cropping did not prevent peak transient memory when the inputs had already been refined or unioned.

### Changes not yet validated

- `exact_propagator_new()` and `balance_Dirac_spinor_new()` exist, but the active path currently does not call them.
- `apply_R_adaptive()` is conceptually better for addition, but has no completed Xe convergence result.
- The latest adaptive-balance logs stop in the first accurate iteration and therefore cannot support energy, convergence, or speed claims.
- The row-to-column Fock indexing correction is identified but not implemented/tested consistently.
- A gauge-invariant occupied-projector residual has not yet replaced the orbital-by-orbital norm as the main comparison criterion.

---

## Exact current checkpoint in the source

This is the part I must read first when I come back.

As of this diary update, the active calls in `scf_4el()` are:

```python
spinorb_array_new = exact_propagator(
    spinorb_array, V_Psi_array, F_ij, prec, True
)

# balance_Dirac_spinor_new(...) is commented out
spinorb_array_new = balance_Dirac_spinor(
    spinorb_array_new, V_Psi_array, F_ij, prec, False
)
```

Therefore the active calculation uses the **old propagator and old balance**. The new helpers are prototypes located later in the same file. In particular, the current active old balance already uses `F_ij[i,i].real` and the old `apply_R()`.

The active convergence rule is the relative trace plus maximum spinor change:

```python
if max_norm_diff < prec*50 and abs(rel_trace_diff) < prec/10:
```

The nominal input variable `thr = prec*10` printed by `test_SWORD.py` is not passed into `scf_4el`; the actual convergence tolerances are hard-coded from `prec` in `Lazy_4_el.py`. This should be cleaned up before producing final benchmark tables, otherwise the output header suggests a threshold that the SWORD driver does not actually use.

The zero-aware `complex_fcn.__mul__` **is active** in the current code.

The worktree contains many other uncommitted source and generated-file changes. Before the next scientific run, the working configuration should be committed or tagged together with its input and output. Otherwise a future filename such as `NEW_THR_ADAPTIVE_R` will again be impossible to map to an exact implementation.

---

## The next experiments, in the order that now makes sense

### 1. Fix and test the Fock index convention first

Use (F_{ji}) for an update of ket (i), including the quadratic coupling. Apply the same convention to SWORD and Dirac. Then run a cheap low-precision test and verify the occupied projection algebraically before spending hours on Xe.

### 2. Add diagnostics that cannot be satisfied by construction

For each iteration record:

\[
\|F-F^\dagger\|_F,
\qquad
\max_i\|\hat F\psi_i-\sum_j\psi_jF_{ji}\|,
\]

and the occupied-projector change. For old and new occupied orbitals, with

\[
M_{ij}=\langle\psi_i^{\mathrm{old}}|\psi_j^{\mathrm{new}}\rangle,
\]

use

\[
\|P_{\mathrm{new}}-P_{\mathrm{old}}\|_F^2
=2N-2\operatorname{Tr}(M^\dagger M).
\]

This quantity is invariant to rotations inside the occupied space, unlike direct orbital differences and raw (F_{03}).

### 3. Repair the precision plumbing in adaptive balance

Choose one documented `work_prec`. If `balance_Dirac_spinor_new()` passes `work_prec`, the called derivative must not silently divide it by ten again. Test at `prec`, `prec/3` and `prec/10`, record the final energy error and native node counts, and select the loosest value that preserves the required observable accuracy.

### 4. Switch one new routine at a time

Do not enable `exact_propagator_new()` and `balance_Dirac_spinor_new()` together for the first validation. A useful matrix is:

| propagator | balance | purpose |
|---|---|---|
| old | old | present completed reference |
| new | old | isolate propagation algebra and zero-grid changes |
| old | adaptive | isolate the (R)-node problem |
| new | adaptive | final candidate only after both pass separately |

Each run should save the Git hash or diff hash, parameters, completion status, final energy, residuals, peak RSS and stage node totals.

### 5. Stop printing/constructing expensive diagnostics in benchmark mode

Guard kinetic decomposition, extra helicity derivatives and full tree dumps with `verbose`. Keep a compact CSV-like iteration line for article data.

### 6. Check the exchange/Kramers-pair convention

The comments around `K_Psi` say that exchange should sum all four occupied spinors, while the implementation skips `j == i + 1` in the relevant loop. Since SWORD and Dirac share this code, it does not explain their node-count difference, but it can affect the physical four-electron energy and should be settled before publication.

---

## Claims I can safely make now for the article

1. A two-component/Weyl propagation embedded in a four-component SCF reaches the same Xe four-electron total energy as the reference 4c algorithm to approximately (10^{-5}\ E_h) in the current test.
2. The SWORD propagation stage is roughly one third faster in steady iterations for this case.
3. The naive implementation loses much of that benefit because algebraic reconstruction of the opposite chirality produces finer adaptive trees, especially in the first iterations.
4. Grid topology is part of the numerical algorithm: algebraically zero cross-terms and the order of derivative, addition, convolution and cropping strongly affect memory even when the formulas are equivalent.
5. A relative, occupied-space-aware convergence measure is more suitable than separate orbital energies, but a projector/residual criterion is still needed for a final fair comparison.

Claims I should **not** make yet:

- that the adaptive (R) implementation is fixed;
- that SWORD has lower peak memory than 4c;
- that the observed (F_{03}) alone proves broken Hermiticity or Kramers symmetry;
- that the latest incomplete `ADAPTIVE_R` or `balance_AI` outputs converged;
- that SWORD is definitively faster end-to-end under identical convergence conditions.

## Minimal restart checklist after a break

1. Read this file, then inspect the active calls near the top of `scf_4el()`; do not assume the `_new` functions are active.
2. Save or commit the current dirty worktree before changing numerical logic.
3. Run the cheap indexing/projection tests before a full Xe job.
4. Record external peak RSS; do not rely on `pympler` for MRCPP tree memory.
5. Print actual live node totals *after* KTRS replacement, plus a separate peak/transient total during Löwdin and balance.
6. Compare against `Xe_DIRAC_MW7_verbose_cropped.out` and the completed relative-trace SWORD output, not against an incomplete launcher log.
7. Change one of propagator, balance, arithmetic or threshold at a time and name the output with the Git hash.

The present scientific picture is encouraging but specific: SWORD's equations are already accurate enough to match the 4c total energy closely, and the steady propagation really is cheaper. The remaining edge is being lost mostly through the representation of reconstructed (R), transient adaptive-grid unions, and convergence/indexing details—not because the basic two-component idea failed.
