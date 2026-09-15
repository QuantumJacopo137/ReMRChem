#!/usr/bin/env python3
"""Launch atomic Dirac or SWORD calculations with 1, 2, or 4 electrons.

Run ``python main_Dirac_vs_SWORD.py --help`` for all controls. The companion
MAIN_DRIVER_GUIDE.md explains checkpoints, examples, and solver limitations.
Only the standard library is imported here, so help and --dry-run work on a
machine without the scientific runtime. Importing this file starts no SCF.
"""

from __future__ import annotations

import argparse
from contextlib import redirect_stderr, redirect_stdout
from datetime import datetime
import json
import math
from pathlib import Path
import shlex
import sys
import time
from types import SimpleNamespace


# Resolve bundled data relative to this file, not the launch directory.
REPO_DIR = Path(__file__).resolve().parent
POTENTIAL_MODELS = (
    "fermi_dirac", "point_charge", "coulomb_HFYGB",
    "homogeneus_charge_sphere", "gaussian",
)
# Atomic numbers come from the order of the symbols, keeping --element and Z
# consistent without requiring an extra chemistry package or an input script.
ELEMENTS = (
    "H He Li Be B C N O F Ne Na Mg Al Si P S Cl Ar K Ca Sc Ti V Cr Mn Fe Co Ni "
    "Cu Zn Ga Ge As Se Br Kr Rb Sr Y Zr Nb Mo Tc Ru Rh Pd Ag Cd In Sn Sb Te I Xe "
    "Cs Ba La Ce Pr Nd Pm Sm Eu Gd Tb Dy Ho Er Tm Yb Lu Hf Ta W Re Os Ir Pt Au Hg "
    "Tl Pb Bi Po At Rn Fr Ra Ac Th Pa U Np Pu Am Cm Bk Cf Es Fm Md No Lr Rf Db "
    "Sg Bh Hs Mt Ds Rg Cn Nh Fl Mc Lv Ts Og"
).split()


def parse_arguments(argv=None):
    """Collect options in functional groups and resolve dependent defaults."""
    parser = argparse.ArgumentParser(
        description=__doc__.split("\n\n")[0],
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    system = parser.add_argument_group("Atom and algorithm")
    system.add_argument("--method", choices=("dirac", "sword"), default="sword")
    system.add_argument("--electrons", type=int, choices=(1, 2, 4), default=4)
    system.add_argument("--element", default="Xe", help="chemical symbol")
    system.add_argument("--charge", type=float, help="nuclear Z; default: atomic number")
    system.add_argument("--position", type=float, nargs=3, default=(0.0, 0.0, 0.0),
                        metavar=("X", "Y", "Z"), help="nuclear position in bohr")
    system.add_argument("--light-speed", type=float, default=137.03599913900001)

    numerics = parser.add_argument_group("Numerical controls")
    numerics.add_argument("--precision", "--prec", type=float, default=1e-7)
    numerics.add_argument("--threshold", "--thr", type=float,
                          help="1e/2e orbital threshold; default: 10*precision")
    numerics.add_argument("--derivative", choices=("PH", "ABGV", "BS"), default="BS",
                          help="1e/2e energy and 4e guess; see guide for kernel defaults")
    numerics.add_argument("--order", type=int, help="MRA order; default: int(-log10(prec))+4")
    box_group = numerics.add_mutually_exclusive_group()
    box_group.add_argument("--box", type=float, help="half-box length in bohr")
    box_group.add_argument("--auto-box", action="store_true",
                           help="use ceil(50/Z + max(abs(position))); also the default")
    numerics.add_argument("--max-depth", type=int, default=25)
    numerics.add_argument("--max-iter", type=int, help="default: 100 for 1e/2e, 30 for 4e")
    numerics.add_argument("--damping", type=float, default=0.5,
                          help="initial mixing coefficient for the 4e solvers")
    numerics.add_argument("--warmup-iterations", type=int,
                          help="coarse Dirac updates; default: 2, or 0 on full restart")
    numerics.add_argument("--warmup-precision", type=float, default=1e-3)

    potential = parser.add_argument_group("Nuclear potential")
    potential.add_argument("--potential", choices=POTENTIAL_MODELS, default="fermi_dirac")
    source = potential.add_mutually_exclusive_group()
    source.add_argument("--compute-potential", action="store_true",
                        help="generate the potential; also the default")
    source.add_argument("--read-potential", type=Path, metavar="PREFIX",
                        help="load PREFIX.tree instead of generating a potential")
    potential.add_argument("--save-potential", action="store_true",
                           help="save the potential in the output directory")
    potential.add_argument("--radius", type=float, default=0.0,
                           help="Fermi half-charge radius in bohr; 0: look up the element")
    potential.add_argument("--radius-table", type=Path,
                           default=REPO_DIR / "Half_Charge_Radius.txt")
    potential.add_argument("--mass-number", type=float,
                           help="A in the 1973 homogeneous-sphere model")
    potential.add_argument("--epsilon", type=float,
                           help="positive Gaussian exponent, in bohr^-2")

    state = parser.add_argument_group("Starting guess and checkpoints")
    read_state = state.add_mutually_exclusive_group()
    read_state.add_argument("--read-orbitals", "--continue-run", type=Path, metavar="PREFIX",
                            help="read full spinors; see guide for prefix conventions")
    read_state.add_argument("--read-guess", type=Path, metavar="PREFIX",
                            help="4e only: read legacy PREFIX0..PREFIX3 two-component seeds")
    state.add_argument("--save-orbitals", action="store_true",
                       help="save final full spinors under output-dir/spinor")
    state.add_argument("--save-guess", action="store_true",
                       help="save initial full spinors and, for 4e fresh guesses, Weyl seeds")

    output = parser.add_argument_group("Output and inspection")
    output.add_argument("--output-dir", type=Path,
                        help="default: a unique directory under ./outputs")
    output.add_argument("--dry-run", action="store_true",
                        help="validate input and print resolved settings without importing VAMPyR")
    args = parser.parse_args(argv)
    try:
        resolve_settings(args)
    except (ValueError, OSError) as exc:
        parser.error(str(exc))
    return args


def resolve_settings(args):
    """Validate before allocating native trees or launching an expensive SCF."""
    args.element = args.element.capitalize()
    if args.element not in ELEMENTS:
        raise ValueError(f"Unknown element: {args.element}")
    if args.charge is None:
        args.charge = float(ELEMENTS.index(args.element) + 1)

    # Reject NaN and infinity as well as invalid signs; argparse accepts both.
    positive = ("charge", "light_speed", "precision", "warmup_precision", "damping")
    for name in positive:
        value = getattr(args, name)
        if not math.isfinite(value) or value <= 0:
            raise ValueError(f"--{name.replace('_', '-')} must be finite and positive")
    if args.precision >= 1 or args.warmup_precision >= 1:
        raise ValueError("Precision values must be smaller than 1")
    if args.damping > 1:
        raise ValueError("--damping must be in (0, 1]")
    if not all(math.isfinite(x) for x in args.position):
        raise ValueError("--position must contain finite coordinates")
    if args.threshold is not None and args.electrons == 4:
        raise ValueError("4e solvers set thresholds internally; omit --threshold (see guide)")
    if args.threshold is None and args.electrons != 4:
        args.threshold = 10 * args.precision
    if args.threshold is not None and (not math.isfinite(args.threshold) or args.threshold <= 0):
        raise ValueError("--threshold must be finite and positive")

    # Keep the origin-centred default of the reference scripts while allowing
    # a displaced atom enough space on either side of its nucleus.
    args.auto_box = args.box is None
    if args.auto_box:
        args.box = float(math.ceil(50 / args.charge + max(abs(x) for x in args.position)))
    if not math.isfinite(args.box) or args.box <= max(abs(x) for x in args.position):
        raise ValueError("--box must be positive and contain the nucleus strictly inside")
    if args.order is None:
        args.order = int(-math.log10(args.precision)) + 4
    if args.order < 1 or args.max_depth < 1:
        raise ValueError("--order and --max-depth must be positive")
    if args.max_iter is None:
        args.max_iter = 30 if args.electrons == 4 else 100
    if args.max_iter < 1:
        raise ValueError("--max-iter must be at least 1")
    if args.warmup_iterations is None:
        args.warmup_iterations = 0 if args.read_orbitals else 2
    if args.warmup_iterations < 0:
        raise ValueError("--warmup-iterations cannot be negative")
    if args.read_guess and args.electrons != 4:
        raise ValueError("--read-guess is for 4e Weyl seeds; use --read-orbitals for 1e/2e")
    if args.electrons == 4 and not (args.read_orbitals or args.read_guess) and args.charge <= 2.05:
        raise ValueError("The screened 4e starting guess needs nuclear Z > 2.05")

    # Only a generated potential needs model parameters. A loaded native tree
    # must already match the requested atom, MRA and model (see the guide).
    if not math.isfinite(args.radius) or args.radius < 0:
        raise ValueError("--radius must be finite and nonnegative")
    args.compute_potential = args.read_potential is None
    if args.read_potential:
        args.read_potential = tree_prefix(args.read_potential)
        require_files([Path(f"{args.read_potential}.tree")])
    elif args.potential == "fermi_dirac" and args.radius == 0:
        args.radius = lookup_radius(args.element, args.radius_table)
    elif args.potential == "gaussian":
        if args.epsilon is None or not math.isfinite(args.epsilon) or args.epsilon <= 0:
            raise ValueError("The Gaussian model requires a positive --epsilon")
    elif args.potential == "homogeneus_charge_sphere":
        if args.mass_number is None or not math.isfinite(args.mass_number) or args.mass_number <= 0:
            raise ValueError("The homogeneous sphere requires a positive --mass-number")

    if args.read_orbitals:
        require_files(spinor_files(args.read_orbitals, args.electrons))
    if args.read_guess:
        require_files(weyl_seed_files(args.read_guess))
    if args.output_dir is None:
        stamp = datetime.now().strftime("%Y%m%d_%H%M%S_%f")
        args.output_dir = Path("outputs") / f"{args.element}_{args.method}_{args.electrons}e_{stamp}"
    args.output_dir = args.output_dir.resolve()


def lookup_radius(element, table):
    """Read the same Fermi radius table as the old scripts, with clear errors."""
    for line in table.read_text().splitlines():
        fields = line.split()
        if len(fields) == 2 and fields[0] == element:
            radius = float(fields[1])
            if math.isfinite(radius) and radius > 0:
                return radius
            break
    raise ValueError(f"No positive radius for {element} in {table}; supply --radius in bohr")


def tree_prefix(path):
    """Native loadTree/saveTree append .tree themselves."""
    return path.with_suffix("") if path.suffix == ".tree" else path


def spinor_prefixes(prefix, electrons):
    """Preserve the legacy 1e prefix and the indexed multi-electron layout."""
    return [prefix] if electrons == 1 else [Path(f"{prefix}_{i}") for i in range(electrons)]


def spinor_files(prefix, electrons):
    """A full orbital uses eight native real/imaginary component tree files."""
    return [Path(f"{name}_{component}_{part}.tree")
            for name in spinor_prefixes(prefix, electrons)
            for component in ("Large_alpha", "Large_beta", "Small_alpha", "Small_beta")
            for part in ("real", "imag")]


def weyl_seed_files(prefix):
    """The old W_spinor0 naming has no underscore before the orbital index."""
    return [Path(f"{prefix}{i}_{component}_{part}.tree")
            for i in range(4) for component in ("top", "bottom")
            for part in ("real", "imag")]


def require_files(paths):
    """Catch incomplete checkpoints before native loaders are called."""
    missing = [str(path) for path in paths if not path.is_file()]
    if missing:
        raise ValueError("Missing input tree(s): " + ", ".join(missing))


def load_backend():
    """Keep scientific imports together; help and configuration need none."""
    from vampyr import vampyr3d as vp
    from orbital4c import complex_fcn as cf
    from orbital4c import orbital as orb
    from orbital4c import orbital_2c as orb2c
    from orbital4c import nuclear_potential as nucpot
    import one_electron as oneel
    import two_electron as twoel
    import starting_guess as sg
    import Lazy_4_el as lazy4el
    import numpy as np
    return SimpleNamespace(vp=vp, cf=cf, orb=orb, orb2c=orb2c, nucpot=nucpot,
                           oneel=oneel, twoel=twoel, sg=sg, lazy4el=lazy4el, np=np)


def initialize_mra(args, backend):
    """All scalar, two-component and four-component objects share one MRA."""
    # VAMPyR's simple endpoint constructor requires integers, even for
    # numerically integral floats. Use its scaled box API for fractional boxes.
    if float(args.box).is_integer():
        box = [-int(args.box), int(args.box)]
    else:
        box = backend.vp.BoundingBox(
            corner=[-1, -1, -1], nboxes=[2, 2, 2], scaling=[args.box] * 3)
    mra = backend.vp.MultiResolutionAnalysis(
        box=box, order=args.order, max_depth=args.max_depth)
    backend.cf.complex_fcn.mra = mra
    for orbital_class in (backend.orb.orbital4c, backend.orb2c.orbital2c):
        orbital_class.mra = mra
        orbital_class.light_speed = args.light_speed
    return mra


def build_potential(args, backend, mra):
    """Read a native potential, or construct one of the five existing models."""
    if args.read_potential:
        potential = backend.vp.FunctionTree(mra)
        potential.loadTree(str(args.read_potential))
    elif args.potential == "fermi_dirac":
        potential = backend.nucpot.Fermi_Dirac(
            args.position, args.charge, args.box, mra,
            args.order, args.precision, args.radius)
    else:
        # Project scalar potentials at the same precision used by the scripts.
        projector = backend.vp.ScalingProjector(mra, args.precision / 10)
        models = {
            "point_charge": lambda x: backend.nucpot.point_charge(x, args.position, args.charge),
            "coulomb_HFYGB": lambda x: backend.nucpot.coulomb_HFYGB(
                x, args.position, args.charge, args.precision / 10),
            "homogeneus_charge_sphere": lambda x: backend.nucpot.homogeneus_charge_sphere_1973(
                x, args.position, args.charge, args.mass_number),
            "gaussian": lambda x: gaussian_value(x, args, backend),
        }
        potential = projector(models[args.potential])
    if args.save_potential:
        potential.saveTree(str(args.output_dir / "potential"))
    return potential


def gaussian_value(position, args, backend):
    """Use the finite nuclear-centre limit to avoid erf(0)/0 in projection."""
    if tuple(position) == tuple(args.position):
        return 2 * args.charge * math.sqrt(args.epsilon / math.pi)
    return backend.nucpot.gaussian_potential(
        position, args.position, args.charge, args.epsilon)


def initial_spinors(args, backend, mra):
    """Prepare full spinors; the solvers handle their own orthonormalization."""
    if args.read_orbitals:
        # Full checkpoints can restart either algorithm. Read the actual saved
        # orbitals directly, avoiding the old uninitialized init_guess branch.
        spinors = []
        for name in spinor_prefixes(args.read_orbitals, args.electrons):
            spinor = backend.orb.orbital4c()
            spinor.read(str(name))
            spinor.normalize()
            spinors.append(spinor)
    elif args.electrons in (1, 2):
        spinor = backend.sg.make_NR_starting_guess(
            args.position, args.charge, mra, args.precision)
        spinor.normalize()
        spinors = [spinor]
        if args.electrons == 2:
            spinors.append(spinor.ktrs(args.precision))
    else:
        spinors = initial_four_spinors(args, backend, mra)

    # Independent spinors define their partners throughout these atomic solvers.
    # Regenerate the pairs after reading as well, preserving the KTRS assumption.
    if args.electrons > 1:
        for i in range(0, args.electrons, 2):
            spinors[i + 1] = spinors[i].ktrs(args.precision)
    if args.save_guess:
        save_spinors(spinors, args.output_dir / "guess")
    return spinors


def initial_four_spinors(args, backend, mra):
    """Use the screened 2s/1s guesses and Weyl balancing of the reference files."""
    if args.read_guess:
        seeds = [backend.orb2c.orbital2c() for _ in range(4)]
        for i, seed in enumerate(seeds):
            seed.read(f"{args.read_guess}{i}")
            seed.normalize()
    else:
        # Preserve the legacy ordering: (2s, KTRS-2s, 1s, KTRS-1s).
        seed_2s = backend.sg.make_NR_starting_guess(
            args.position, args.charge - 2.05, mra, args.precision, comp=2, n=2, l=0)
        seed_1s = backend.sg.make_NR_starting_guess(
            args.position, args.charge - 0.3, mra, args.precision, comp=2, n=1, l=0)
        seeds = [seed_2s, seed_2s.ktrs(args.precision),
                 seed_1s, seed_1s.ktrs(args.precision)]
    if args.save_guess:
        for i, seed in enumerate(seeds):
            seed.save(str(args.output_dir / f"W_spinor{i}"))

    # Construct L and R entirely in memory; saving/reloading the generated
    # seeds is optional and no longer a required step of the calculation.
    spinors = []
    for seed in seeds:
        correction = (1 / (2 * args.light_speed)) * seed.sigma_p(args.precision, args.derivative)
        left = seed + correction
        right = seed - correction
        spinor = backend.orb2c.Weyl_to_Dirac(left, right)
        spinor.normalize()
        spinors.append(spinor)
    return spinors


def save_spinors(spinors, prefix):
    """Use one layout for guesses, final checkpoints, and full restarts."""
    for spinor, name in zip(spinors, spinor_prefixes(prefix, len(spinors))):
        spinor.save(str(name))


def dispatch_solver(args, backend, mra, potential, spinors, *, method,
                    precision, threshold, iterations, output_file):
    """Map the six supported cases to existing solvers without duplicating SCF."""
    if args.electrons == 1:
        solver = backend.oneel.gs_D_1e if method == "dirac" else backend.oneel.gs_SWORD_1e
        spinor = solver(spinors[0], potential, mra, precision, threshold,
                        args.derivative, args.charge, str(output_file), niter=iterations)
        return [spinor], None
    if args.electrons == 2:
        solver = (backend.twoel.coulomb_gs_2e if method == "dirac"
                  else backend.twoel.coulomb_gs_2e_SWORD)
        pair = solver(spinors[0], potential, mra, precision, threshold,
                      args.derivative, str(output_file), niter=iterations)
        return list(pair), None

    solver = backend.lazy4el.scf_4e_4c if method == "dirac" else backend.lazy4el.scf_4el
    # The 4e API has internal derivative/convergence choices, not thr/derivative
    # parameters. Print these limits in the header rather than implying support.
    return solver(spinors, potential, mra, precision, max_iter=iterations,
                  auto_save=False, Dampen_alpha=args.damping)


def calculation_settings(args):
    """Record resolved values, including controls fixed inside legacy solvers."""
    settings = vars(args).copy()
    settings["working_directory"] = str(Path.cwd())
    if args.electrons == 4:
        settings["convergence"] = {
            "max_spinor_change": 50 * args.precision,
            "relative_trace_change": args.precision / 10,
        }
        settings["derivative_scope"] = "--derivative controls the guess; 4e kernels use built-in choices"
    else:
        settings["convergence"] = {"orbital_threshold": args.threshold}
        settings["derivative_scope"] = "--derivative controls the energy Hamiltonian"
        if args.electrons == 1:
            settings["convergence"]["relative_energy_threshold"] = args.precision / 10
            settings["convergence"]["rule"] = "legacy 1e solver may stop on either criterion"
        if args.method == "sword":
            settings["sword_derivatives"] = {"propagation": "BS", "balance": "ABGV"}
    return settings


class Tee:
    """Mirror Python output to the terminal and the run log, flushing promptly."""

    def __init__(self, terminal, logfile):
        self.terminal = terminal
        self.logfile = logfile

    def write(self, text):
        self.terminal.write(text)
        self.logfile.write(text)
        self.flush()
        return len(text)

    def flush(self):
        self.terminal.flush()
        self.logfile.flush()


def run_calculation(args, backend):
    """Set up, warm up, solve, and save, with separate wall times for each."""
    total_start = time.time()
    settings = calculation_settings(args)
    print(f"{args.element}: {args.electrons} electrons, {args.method.upper()}")
    print(json.dumps(settings, indent=2, default=str))
    (args.output_dir / "parameters.json").write_text(json.dumps(settings, indent=2, default=str) + "\n")

    # Potential generation and guess creation are outside the measured SCF time.
    mra = initialize_mra(args, backend)
    potential = build_potential(args, backend, mra)
    spinors = initial_spinors(args, backend, mra)
    timings = {"setup_seconds": time.time() - total_start}

    # Both methods get the same optional coarse Dirac preparation. A full
    # restart skips this by default; an explicit count can enable it again.
    start = time.time()
    if args.warmup_iterations:
        print("\n-> Coarse Dirac warm-up...")
        spinors, _ = dispatch_solver(
            args, backend, mra, potential, spinors, method="dirac",
            precision=args.warmup_precision, threshold=10 * args.warmup_precision,
            iterations=args.warmup_iterations, output_file=args.output_dir / "warmup_energies.txt")
    timings["warmup_seconds"] = time.time() - start

    print(f"\n-> Main {args.method.upper()} calculation...")
    start = time.time()
    spinors, fock = dispatch_solver(
        args, backend, mra, potential, spinors, method=args.method,
        precision=args.precision, threshold=args.threshold,
        iterations=args.max_iter, output_file=args.output_dir / "energies.txt")
    timings["scf_seconds"] = time.time() - start

    # Save full spinors for either method: SWORD's reconstructed R is part of
    # the checkpoint, so restart does not have to guess or reconstruct it again.
    start = time.time()
    if args.save_orbitals:
        save_spinors(spinors, args.output_dir / "spinor")
    if fock is not None:
        backend.np.savetxt(str(args.output_dir / "fock_matrix.txt"), fock, fmt="%.16e")
        print("\nFinal Fock matrix (including rest energy):")
        print(fock)
    timings["save_seconds"] = time.time() - start
    timings["total_seconds"] = time.time() - total_start
    (args.output_dir / "timings.json").write_text(json.dumps(timings, indent=2) + "\n")

    # A returned state is not necessarily converged. Keep the solver's explicit
    # convergence messages in run.log; this footer reports timings and paths.
    centered = all(x == 0 for x in args.position)
    print(f"\nCalculation ID: {args.element} {args.electrons}e {args.method} "
          f"centered={int(centered)} prec={args.precision:g} {args.derivative} box={args.box:g}")
    for stage, seconds in timings.items():
        print(f"{stage}: {seconds:.4f} seconds")
    print(f"Results: {args.output_dir}")
    return spinors, fock


def main(argv=None):
    """Keep the command-line boundary small and the calculation functions reusable."""
    args = parse_arguments(argv)
    if args.dry_run:
        print(json.dumps(calculation_settings(args), indent=2, default=str))
        return 0
    try:
        backend = load_backend()
    except ImportError as exc:
        print(f"Cannot load the scientific runtime: {exc}\n"
              "Activate your ReMRChem environment; see MAIN_DRIVER_GUIDE.md.", file=sys.stderr)
        return 1

    args.output_dir.mkdir(parents=True, exist_ok=True)
    # Exclusive log creation avoids silently replacing an earlier experiment.
    try:
        logfile = (args.output_dir / "run.log").open("x", buffering=1)
    except FileExistsError:
        print(f"A run already exists in {args.output_dir}; choose another --output-dir.", file=sys.stderr)
        return 1
    with logfile:
        with redirect_stdout(Tee(sys.stdout, logfile)), redirect_stderr(Tee(sys.stderr, logfile)):
            command_args = sys.argv[1:] if argv is None else list(argv)
            command = " ".join(shlex.quote(part) for part in
                               [sys.executable, str(Path(__file__).resolve()), *command_args])
            (args.output_dir / "command.txt").write_text(command + "\n")
            print(f"Command: {command}")
            # Keep tracebacks beside the parameters of a failed scientific run.
            try:
                run_calculation(args, backend)
            except Exception:
                import traceback
                traceback.print_exc()
                return 1
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
