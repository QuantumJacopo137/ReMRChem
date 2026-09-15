# Running the Dirac/SWORD comparison driver

`main_Dirac_vs_SWORD.py` is the common entry point for **one atom with 1, 2, or 4 electrons**, using either Dirac propagation of four components or SWORD propagation of the independent Weyl component followed by balancing. Both algorithms store complete four-component spinors. Two-electron and four-electron calculations use Kramers pairs.

## Start here

Use Python 3.7 or newer in the working ReMRChem environment. Calculations need VAMPyR, NumPy, SciPy, Matplotlib, and Pympler, plus the local solver and `orbital4c` files. Matplotlib and Pympler are imported by existing solver modules even when you are not plotting. Use the same scientific environment as the original scripts; this guide does not prescribe a replacement VAMPyR version.

From the repository directory:

```bash
python main_Dirac_vs_SWORD.py --help
python main_Dirac_vs_SWORD.py --method sword --electrons 2 --element He --dry-run
```

`--dry-run` checks options, required tree files and nuclear-model parameters, prints the resolved settings, and exits. It neither imports scientific packages nor writes an output directory. It cannot validate the contents or compatibility of a native tree file.

With no options, the driver selects Xe, four electrons, SWORD, Fermi–Dirac nuclear potential, `prec=1e-7`, BS derivatives, automatic box and polynomial order, and at most two coarse Dirac warm-up iterations at `1e-3`. It computes the potential. Potential, guess and final-orbital tree saving are individually opt-in.

## Select a calculation

All six combinations use the same command-line interface:

```bash
# One electron: hydrogen-like calculations.
python main_Dirac_vs_SWORD.py --method dirac --electrons 1 --element H
python main_Dirac_vs_SWORD.py --method sword --electrons 1 --element H

# Two electrons: one occupied Kramers pair.
python main_Dirac_vs_SWORD.py --method dirac --electrons 2 --element He
python main_Dirac_vs_SWORD.py --method sword --electrons 2 --element He

# Four electrons: two occupied Kramers pairs.
python main_Dirac_vs_SWORD.py --method dirac --electrons 4 --element Be
python main_Dirac_vs_SWORD.py --method sword --electrons 4 --element Be
```

For an initial smoke test, add `--precision 1e-4 --max-iter 3`. Reaching that iteration limit is useful for checking the setup but does not establish convergence. Nuclear charge comes from `--element`; `--charge` explicitly overrides it. Electron count is independent of nuclear charge, so `--element Xe --electrons 2` means a two-electron Xe ion.

A controlled pair of runs can use separate output directories:

```bash
python main_Dirac_vs_SWORD.py --method dirac --electrons 2 --element He \
    --precision 1e-5 --max-iter 100 --save-potential --save-orbitals \
    --output-dir outputs/He_2e_dirac
python main_Dirac_vs_SWORD.py --method sword --electrons 2 --element He \
    --precision 1e-5 --max-iter 100 --read-potential outputs/He_2e_dirac/potential \
    --save-orbitals --output-dir outputs/He_2e_sword
```

The second run reuses only the potential; both start from freshly generated guesses. Use matching numerical controls and inspect convergence before comparing energies or times. Choose fresh directory names when repeating the example.

## Numerical controls

| Option | Meaning and default |
|---|---|
| `--precision`, `--prec` | Multiwavelet precision; `1e-7`. |
| `--threshold`, `--thr` | Orbital-change threshold for 1e/2e; `10*precision`. |
| `--order` | Polynomial order; `int(-log10(precision)) + 4`. |
| `--box` | Half-length of the box in bohr. |
| `--auto-box` | Default when `--box` is omitted: `ceil(50/Z + max(abs(position)))`. |
| `--position X Y Z` | Nuclear position in bohr; the origin. |
| `--max-depth` | MRA refinement-depth limit; 25. |
| `--light-speed` | Speed of light in atomic units; `137.03599913900001`. |
| `--derivative` | `BS`, `ABGV`, or `PH`; default `BS`. Scope is explained below. |
| `--max-iter` | Maximum main SCF updates; 100 for 1e/2e, 30 for 4e. |
| `--warmup-iterations` | Maximum coarse Dirac updates; 2 for fresh guesses, 0 for full restarts. Set 0 to disable. |
| `--warmup-precision` | Precision during warm-up; `1e-3`. |
| `--damping` | Initial mixing coefficient for 4e; 0.5. The 4e solver subsequently adjusts it. |

Warm-up is available for either algorithm and all electron counts. It passes its returned state directly into the main solver. The two-electron Dirac routine now accepts an optional `niter` argument to support this; its Dirac propagation formula is retained, and a final energy-only pass reports the energy of the returned spinor.

The existing solver interfaces differ:

- **1e:** the legacy solver may stop when either the orbital threshold or its relative-energy criterion is satisfied. Its analytical hydrogen-like energy comparison is a point-charge reference, including when you select a finite nuclear model.
- **2e:** the orbital-change threshold controls stopping. Energies are calculated for the returned state. The Kramers reduction retains the partner Coulomb field after self direct/exchange cancellation.
- **4e:** both routines currently require maximum spinor change below `50*precision` and relative occupied-trace change below `precision/10`. They do not accept an external `thr`; the driver rejects `--threshold` for 4e and records the actual limits in `parameters.json`.
- **Derivative choices:** Dirac 1e/2e uses the selected derivative in its Hamiltonian calls. In SWORD 1e/2e, the shared propagation helper uses BS and the balance helper uses ABGV; the selected derivative controls the energy Hamiltonian. For 4e, `--derivative` controls construction of a fresh Weyl guess, while SCF uses the kernels' built-in choices. The driver does not change these numerical conventions.

A successful process exit means the solver returned without an exception. It **does not guarantee SCF convergence**: check the solver's convergence or iteration-limit messages in `run.log`.

## Nuclear potentials

The driver stores the positive nuclear-potential tree used by the existing code; solver calls supply the attractive minus sign.

| `--potential` | Additional input |
|---|---|
| `fermi_dirac` | `--radius` is the half-charge radius in **bohr**. With radius 0, look up `--element` in `Half_Charge_Radius.txt`. |
| `point_charge` | No extra model parameter. |
| `coulomb_HFYGB` | Uses the existing Harrison smoothing with precision `prec/10`. |
| `homogeneus_charge_sphere` | The spelling follows the existing code. Supply `--mass-number A` for the 1973 sphere model. |
| `gaussian` | Supply a positive `--epsilon`, the Gaussian exponent in bohr⁻². |

`--radius-table PATH` selects another two-column symbol/radius table. Elements absent from the table need an explicit `--radius`. The sphere routine's fourth argument is a **mass number**, even though the old scripts passed a variable called `radius`; it is now named explicitly in the driver.

```bash
python main_Dirac_vs_SWORD.py --electrons 2 --element He \
    --potential point_charge --dry-run
python main_Dirac_vs_SWORD.py --electrons 2 --element He \
    --potential gaussian --epsilon 1.167153887e9 --dry-run
python main_Dirac_vs_SWORD.py --electrons 4 --element Xe \
    --potential homogeneus_charge_sphere --mass-number 132 --dry-run
```

A generated Gaussian potential uses the finite value at the nuclear centre, avoiding the legacy helper's `erf(0)/0` expression there.

`--compute-potential` explicitly selects generation, which is also the default. `--read-potential PREFIX` loads `PREFIX.tree`; the `.tree` suffix may be included. These options are mutually exclusive. `--save-potential` writes `potential.tree` inside the new output directory.

Native tree loading does not reconstruct your calculation settings. When reusing a potential, provide the same atom, centre, box, order, nuclear model and appropriate precision as the producing run. Consult its `parameters.json`. The driver verifies file existence but cannot verify that the stored potential matches these settings.

## Guesses, saving and restarting

Fresh 1e/2e calculations use the existing hydrogen-like 1s guess. Fresh 4e calculations retain the old screened guesses and ordering: indices 0/1 are a 2s Kramers pair with effective charge `Z-2.05`; indices 2/3 are a 1s pair with `Z-0.3`. They are converted to full spinors in memory. The four-electron SCF routines perform orthonormalization. Some legacy solver print labels describe these indices as 1s/2s in the opposite order; use the orbital indices when comparing their output.

- `--save-guess` saves the initialized full spinors before warm-up under `output-dir/guess`. For a generated or loaded 4e Weyl guess, it also saves `W_spinor0` through `W_spinor3` two-component seeds.
- `--save-orbitals` saves the final full spinors under `output-dir/spinor`.
- `--read-orbitals PREFIX` loads full spinors for either method. `--continue-run` is an alias. Warm-up defaults to zero on this path, but can be enabled explicitly.
- `--read-guess PREFIX` is the separate 4e path for legacy two-component seeds such as `W_spinor0`. The driver adds indices directly to the prefix and reconstructs the initial L/R components. This retains the normal warm-up default.

Full-spinor prefix conventions are shared by saving and loading:

| Electron count | Prefix passed to `--read-orbitals` | Full spinors loaded |
|---|---|---|
| 1 | `outputs/run/spinor` | `spinor` |
| 2 | `outputs/run/spinor` | `spinor_0`, `spinor_1` |
| 4 | `outputs/run/spinor` | `spinor_0` through `spinor_3` |

Each full spinor consists of eight files, for example `spinor_0_Large_alpha_real.tree`. Supply the **prefix**, not an individual component file. Save or copy the complete set. Multi-electron partners are regenerated from the independent spinors to retain KTRS.

```bash
# Restart a full 2e checkpoint, using its matching potential and settings.
python main_Dirac_vs_SWORD.py --method sword --electrons 2 --element He \
    --precision 1e-5 --read-potential outputs/He_2e_dirac/potential \
    --read-orbitals outputs/He_2e_sword/spinor \
    --save-orbitals --output-dir outputs/He_2e_restart

# Load the existing four-electron W_spinor0..3 seeds in the repository.
python main_Dirac_vs_SWORD.py --method sword --electrons 4 --element Xe \
    --read-guess W_spinor --read-potential potential --dry-run

# Legacy one-electron full-spinor filename, without an added index.
python main_Dirac_vs_SWORD.py --electrons 1 --element H \
    --read-orbitals Last_run_spinor_1el --dry-run
```

## Outputs and timing

Without `--output-dir`, each launch gets a unique directory under the current directory's `outputs/`. An explicit directory containing `run.log` is rejected so that an earlier experiment is not replaced.

| File | Contents |
|---|---|
| `run.log` | Python stdout/stderr, solver iteration output, final reports and any caught traceback. |
| `parameters.json` | Resolved inputs, working directory and actual convergence/derivative controls. |
| `command.txt` | Shell-quoted launch command. |
| `timings.json` | Setup, coarse warm-up, main SCF, saving, and total wall time. |
| `energies.txt` | Main 1e/2e solver energy report. Four-electron energy reports are in `run.log`. |
| `warmup_energies.txt` | Separate 1e/2e warm-up report, when enabled. |
| `fock_matrix.txt` | Final complex 4e Fock matrix, including rest energy. |
| Native `.tree` files | Potential, guesses and final orbitals when requested. |

For solver comparisons, use `scf_seconds` to exclude setup and warm-up, and inspect `total_seconds` for the broader cost. Existing stage timers remain visible in the log. A nonconverged run may still produce timings and checkpoints.

The Python log mirror cannot capture output written directly by native code to operating-system stdout/stderr. For a complete terminal transcript, redirect the launch with the shell as well:

```bash
python main_Dirac_vs_SWORD.py --method sword --electrons 4 --element Xe \
    --output-dir outputs/Xe_full_run > Xe_full_console.log 2>&1
```

## Handing the calculation to someone else

Share the repository revision **and its uncommitted source changes**, this guide, the run's `parameters.json` and `command.txt`, and any complete native trees used as inputs. `Half_Charge_Radius.txt` must accompany runs that rely on table lookup. Record the working scientific environment, for example with `python --version` and your environment's package export. Relative input paths in the recorded command are interpreted from the recorded working directory.

Configuration does not execute `input.txt` as Python code. Translate the settings from an old script using the options above. The experimental plotting blocks commented out in the old scripts remain in those scripts; they are not part of this SCF launch interface.

For code maintenance, the new driver's functions follow the run itself: `parse_arguments`/`resolve_settings`, `initialize_mra`, `build_potential`, `initial_spinors`, `dispatch_solver`, and `run_calculation`. Numerical SCF equations remain in `one_electron.py`, `two_electron.py`, and `Lazy_4_el.py`.

Run the lightweight driver tests without VAMPyR:

```bash
python -B -m unittest discover -s tests -p test_main_driver.py -v
```

These checks cover dispatch, warm-up, checkpoint paths, potential selection, validation and output using stand-ins. They do not validate the native kernels. A physical convergence and timing comparison is still required in the scientific runtime; the existing four-electron experimental issues recorded in `last_activity.md` remain open.
