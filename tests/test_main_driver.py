"""Driver checks using a lightweight backend; no VAMPyR installation needed.

These test option semantics, actual orchestration, and files. They deliberately
make no claims about the numerical convergence of the scientific solvers.
"""

import contextlib
import io
import json
from pathlib import Path
import tempfile
from types import SimpleNamespace
import unittest
from unittest.mock import Mock, patch

import main_Dirac_vs_SWORD as driver


class FakeSpinor:
    components = ("Large_alpha", "Large_beta", "Small_alpha", "Small_beta")

    def __init__(self, value=1.0):
        self.value = value
        self.loaded = None

    def __add__(self, other):
        return type(self)(self.value + other.value)

    def __sub__(self, other):
        return type(self)(self.value - other.value)

    def __rmul__(self, scale):
        return type(self)(scale * self.value)

    def normalize(self):
        pass

    def ktrs(self, precision):
        return type(self)(-self.value)

    def sigma_p(self, precision, derivative):
        return type(self)(0.1)

    def read(self, prefix):
        self.loaded = prefix

    def save(self, prefix):
        for component in self.components:
            for part in ("real", "imag"):
                Path(f"{prefix}_{component}_{part}.tree").touch()


class FakeWeyl(FakeSpinor):
    components = ("top", "bottom")


class FakeTree:
    def __init__(self, mra=None):
        self.loaded = None

    def loadTree(self, prefix):
        self.loaded = prefix

    def saveTree(self, prefix):
        Path(f"{prefix}.tree").touch()


def fake_backend():
    """Six distinct mocks expose method selection and warm-up state transfer."""
    def solver(electrons):
        def call(*args, **kwargs):
            result = [FakeSpinor(i + 10) for i in range(electrons)]
            if electrons == 1:
                return result[0]
            if electrons == 2:
                return tuple(result)
            return result, [[1.0, 0.0], [0.0, 2.0]]
        return Mock(side_effect=call)

    oneel = SimpleNamespace(gs_D_1e=solver(1), gs_SWORD_1e=solver(1))
    twoel = SimpleNamespace(coulomb_gs_2e=solver(2), coulomb_gs_2e_SWORD=solver(2))
    lazy4el = SimpleNamespace(scf_4e_4c=solver(4), scf_4el=solver(4))
    # Evaluating projected scalar models also checks that their required
    # arguments have been passed through, rather than only selecting a name.
    def projector(mra, precision):
        def project(function):
            function([0.1, 0.2, 0.3])
            return FakeTree()
        return project
    nucpot = SimpleNamespace(**{name: Mock(return_value=1.0) for name in (
        "point_charge", "coulomb_HFYGB", "gaussian_potential", "homogeneus_charge_sphere_1973")})
    nucpot.Fermi_Dirac = Mock(return_value=FakeTree())
    return SimpleNamespace(
        oneel=oneel, twoel=twoel, lazy4el=lazy4el, nucpot=nucpot,
        orb=SimpleNamespace(orbital4c=FakeSpinor),
        orb2c=SimpleNamespace(orbital2c=FakeWeyl,
                              Weyl_to_Dirac=lambda left, right: FakeSpinor(left.value + right.value)),
        cf=SimpleNamespace(complex_fcn=type("Scalar", (), {})),
        vp=SimpleNamespace(FunctionTree=FakeTree, ScalingProjector=projector,
                           MultiResolutionAnalysis=Mock(return_value="test-mra")),
        sg=SimpleNamespace(make_NR_starting_guess=Mock(side_effect=lambda *a, **kw:
            FakeWeyl() if kw.get("comp") == 2 else FakeSpinor())),
        np=SimpleNamespace(savetxt=lambda path, matrix, **kw: Path(path).write_text(str(matrix))),
    )


class MainDriverTests(unittest.TestCase):
    def setUp(self):
        self.tmp = tempfile.TemporaryDirectory(prefix="main-driver-test-")
        self.addCleanup(self.tmp.cleanup)
        self.root = Path(self.tmp.name)

    def parse(self, *options):
        return driver.parse_arguments(["--output-dir", str(self.root / "run"), *options])

    def assert_bad_input(self, *options):
        with contextlib.redirect_stderr(io.StringIO()), self.assertRaises(SystemExit) as exc:
            self.parse(*options)
        self.assertEqual(exc.exception.code, 2)

    def test_element_defaults_and_displaced_automatic_box(self):
        args = self.parse("--element", "he", "--electrons", "2", "--position", "0.1", "0.2", "0.3")
        self.assertEqual(args.charge, 2)
        self.assertEqual(args.box, 26)
        self.assertEqual(args.order, 11)
        self.assertEqual(args.threshold, 1e-6)
        self.assertTrue(args.compute_potential)
        self.assertGreater(args.radius, 0)

    def test_mra_box_matches_native_constructor_types(self):
        backend = fake_backend()
        args = self.parse("--box", "5")
        driver.initialize_mra(args, backend)
        box = backend.vp.MultiResolutionAnalysis.call_args[1]["box"]
        self.assertEqual(box, [-5, 5])
        self.assertTrue(all(type(endpoint) is int for endpoint in box))
        backend.vp.BoundingBox = Mock(return_value="scaled-native-box")
        args = self.parse("--box", "1.5")
        driver.initialize_mra(args, backend)
        backend.vp.BoundingBox.assert_called_once_with(
            corner=[-1, -1, -1], nboxes=[2, 2, 2], scaling=[1.5, 1.5, 1.5])
        self.assertEqual(backend.vp.MultiResolutionAnalysis.call_args[1]["box"], "scaled-native-box")

    def test_invalid_inputs_fail_before_runtime(self):
        cases = [
            ("--precision", "0"), ("--precision", "nan"), ("--precision", "1"),
            ("--light-speed", "inf"), ("--box", "0"), ("--max-iter", "0"),
            ("--warmup-iterations", "-1"), ("--damping", "1.1"),
            ("--potential", "gaussian"), ("--potential", "homogeneus_charge_sphere"),
            ("--electrons", "4", "--threshold", "1e-5"),
            ("--electrons", "2", "--read-guess", "missing"),
            ("--box", "1", "--position", "1", "0", "0"),
            ("--read-potential", "missing"), ("--read-orbitals", "missing"),
            ("--element", "Li", "--electrons", "1"),
        ]
        for case in cases:
            with self.subTest(case=case):
                self.assert_bad_input(*case)

    def test_radius_override_handles_elements_absent_from_table(self):
        args = self.parse("--element", "Li", "--electrons", "1", "--radius", "3e-5")
        self.assertEqual(args.radius, 3e-5)
        self.assertEqual(args.charge, 3)

    def test_full_restart_defaults_to_no_warmup(self):
        prefix = self.root / "old"
        for file in driver.spinor_files(prefix, 4):
            file.touch()
        args = self.parse("--continue-run", str(prefix))
        self.assertEqual(args.warmup_iterations, 0)
        explicit = self.parse("--read-orbitals", str(prefix), "--warmup-iterations", "3")
        self.assertEqual(explicit.warmup_iterations, 3)
        # A single absent component is enough to reject an incomplete restart.
        driver.spinor_files(prefix, 4)[-1].unlink()
        self.assert_bad_input("--read-orbitals", str(prefix))

    def test_dry_run_neither_imports_backend_nor_creates_output(self):
        with patch.object(driver, "load_backend") as load, contextlib.redirect_stdout(io.StringIO()) as output:
            code = driver.main(["--dry-run", "--output-dir", str(self.root / "dry")])
        self.assertEqual(code, 0)
        load.assert_not_called()
        self.assertFalse((self.root / "dry").exists())
        self.assertAlmostEqual(json.loads(output.getvalue())["convergence"]["max_spinor_change"], 5e-6)

    def test_all_six_modes_run_and_save_results(self):
        for electrons in (1, 2, 4):
            for method in ("dirac", "sword"):
                with self.subTest(electrons=electrons, method=method):
                    backend = fake_backend()
                    outdir = self.root / f"{method}_{electrons}"
                    options = ["--method", method, "--electrons", str(electrons),
                               "--max-iter", "3", "--warmup-iterations", "0",
                               "--save-potential", "--save-guess", "--save-orbitals",
                               "--output-dir", str(outdir)]
                    with patch.object(driver, "load_backend", return_value=backend), contextlib.redirect_stdout(io.StringIO()):
                        self.assertEqual(driver.main(options), 0)
                    for name in ("run.log", "parameters.json", "command.txt", "timings.json", "potential.tree"):
                        self.assertTrue((outdir / name).is_file(), name)
                    for prefix in ("spinor", "guess"):
                        self.assertTrue(all(path.is_file() for path in driver.spinor_files(outdir / prefix, electrons)))
                    if electrons == 4:
                        self.assertTrue(all(path.is_file() for path in driver.weyl_seed_files(outdir / "W_spinor")))
                        self.assertTrue((outdir / "fock_matrix.txt").is_file())
                    self.assertIn("Main " + method.upper(), (outdir / "run.log").read_text())
                    timings = json.loads((outdir / "timings.json").read_text())
                    self.assertGreaterEqual(timings["total_seconds"], timings["scf_seconds"])
                    dirac, sword = self.solvers(backend, electrons)
                    chosen, unused = (dirac, sword) if method == "dirac" else (sword, dirac)
                    chosen.assert_called_once()
                    unused.assert_not_called()
                    keyword = "max_iter" if electrons == 4 else "niter"
                    self.assertEqual(chosen.call_args[1][keyword], 3)

    @staticmethod
    def solvers(backend, electrons):
        return {
            1: (backend.oneel.gs_D_1e, backend.oneel.gs_SWORD_1e),
            2: (backend.twoel.coulomb_gs_2e, backend.twoel.coulomb_gs_2e_SWORD),
            4: (backend.lazy4el.scf_4e_4c, backend.lazy4el.scf_4el),
        }[electrons]

    def test_warmup_state_is_used_by_main_solver_for_each_electron_count(self):
        for electrons in (1, 2, 4):
            backend = fake_backend()
            outdir = self.root / f"warmup_{electrons}"
            outdir.mkdir()
            args = driver.parse_arguments(["--electrons", str(electrons), "--output-dir", str(outdir)])
            dirac, sword = self.solvers(backend, electrons)
            prepared = [FakeSpinor(42) for _ in range(electrons)]
            dirac.side_effect = None
            dirac.return_value = (prepared, None) if electrons == 4 else (prepared[0] if electrons == 1 else tuple(prepared))
            with contextlib.redirect_stdout(io.StringIO()):
                driver.run_calculation(args, backend)
            self.assertIs(sword.call_args[0][0], prepared if electrons == 4 else prepared[0])
            self.assertEqual(dirac.call_args[0][3], 1e-3)
            self.assertEqual(sword.call_args[0][3], 1e-7)

    def test_loaded_single_spinor_reaches_solver_without_guess_generation(self):
        backend = fake_backend()
        prefix = self.root / "Last_run_spinor_1el"
        for file in driver.spinor_files(prefix, 1):
            file.touch()
        outdir = self.root / "restart"
        outdir.mkdir()
        args = driver.parse_arguments(["--electrons", "1", "--read-orbitals", str(prefix),
                                       "--output-dir", str(outdir)])
        with contextlib.redirect_stdout(io.StringIO()):
            driver.run_calculation(args, backend)
        backend.sg.make_NR_starting_guess.assert_not_called()
        backend.oneel.gs_D_1e.assert_not_called()
        loaded = backend.oneel.gs_SWORD_1e.call_args[0][0]
        self.assertEqual(loaded.loaded, str(prefix))

    def test_legacy_weyl_seeds_can_be_loaded_without_writing_them(self):
        backend = fake_backend()
        prefix = self.root / "W_spinor"
        for file in driver.weyl_seed_files(prefix):
            file.touch()
        args = self.parse("--read-guess", str(prefix))
        spinors = driver.initial_spinors(args, backend, "test-mra")
        self.assertEqual(len(spinors), 4)
        backend.sg.make_NR_starting_guess.assert_not_called()
        self.assertFalse(args.output_dir.exists())

    def test_all_potential_models_and_gaussian_center(self):
        backend = fake_backend()
        cases = [
            ("fermi_dirac", [], "Fermi_Dirac"), ("point_charge", [], "point_charge"),
            ("coulomb_HFYGB", [], "coulomb_HFYGB"),
            ("gaussian", ["--epsilon", "12"], "gaussian_potential"),
            ("homogeneus_charge_sphere", ["--mass-number", "132"], "homogeneus_charge_sphere_1973"),
        ]
        for model, options, called in cases:
            args = self.parse("--potential", model, *options)
            self.assertIsInstance(driver.build_potential(args, backend, "mra"), FakeTree)
            getattr(backend.nucpot, called).assert_called_once()
        args = self.parse("--potential", "gaussian", "--epsilon", "12")
        self.assertGreater(driver.gaussian_value(args.position, args, backend), 0)
        # The centre limit must not call the singular legacy implementation.
        backend.nucpot.gaussian_potential.assert_called_once()
        self.assertEqual(backend.nucpot.homogeneus_charge_sphere_1973.call_args[0][-1], 132)

    def test_read_potential_accepts_native_tree_extension(self):
        path = self.root / "potential.tree"
        path.touch()
        args = self.parse("--read-potential", str(path), "--potential", "gaussian")
        self.assertFalse(args.compute_potential)
        tree = driver.build_potential(args, fake_backend(), "mra")
        self.assertEqual(tree.loaded, str(path.with_suffix("")))

    def test_existing_run_is_not_overwritten(self):
        outdir = self.root / "existing"
        outdir.mkdir()
        (outdir / "run.log").write_text("previous run")
        with patch.object(driver, "load_backend", return_value=fake_backend()), contextlib.redirect_stderr(io.StringIO()):
            self.assertEqual(driver.main(["--output-dir", str(outdir)]), 1)
        self.assertEqual((outdir / "run.log").read_text(), "previous run")

    def test_missing_runtime_produces_actionable_error(self):
        with patch.object(driver, "load_backend", side_effect=ImportError("No module named vampyr")), contextlib.redirect_stderr(io.StringIO()) as output:
            self.assertEqual(driver.main(["--output-dir", str(self.root / "missing")]), 1)
        self.assertIn("MAIN_DRIVER_GUIDE.md", output.getvalue())
        self.assertFalse((self.root / "missing").exists())


if __name__ == "__main__":
    unittest.main()
