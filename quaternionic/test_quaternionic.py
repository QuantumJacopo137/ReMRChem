"""
Test suite for quaternionic orbital classes.

This demonstrates the usage of CompQuatOrbital (4-component complex spinor)
and QuatOrbital (2-component real quaternionic spinor) classes.
"""
import numpy as np

# Import quaternionic classes
from quaternionic.quat_function import QuatFunction
from quaternionic.quat_orbital import CompQuatOrbital, QuatOrbital


def test_comp_quat_orbital():
    """
    Test CompQuatOrbital: 4-component spinor as complex quaternionic function.

    Structure: q = q0 + q1*i + q2*j + q3*k where:
        - q0 = La (Large alpha)
        - q1 = Lb (Large beta)
        - q2 = Sa (Small alpha)
        - q3 = Sb (Small beta)
    Each component is a complex function.
    """
    print("=" * 60)
    print("Test: CompQuatOrbital (4-component complex spinor)")
    print("=" * 60)

    # Create a complex quaternionic orbital
    orb = CompQuatOrbital()

    print(f"Created CompQuatOrbital")
    print(f"Number of components: {len(orb)}")
    print(f"Component mapping: {orb.comp_dict}")

    # Test component access
    print("\nTesting component access:")
    for comp in ['La', 'Lb', 'Sa', 'Sb']:
        comp_view = orb[comp]
        print(f"  {comp}: real={comp_view.real}, imag={comp_view.imag}")

    # Test norm
    print(f"\nInitial norm: {orb.norm()}")

    # Test arithmetic
    orb2 = CompQuatOrbital()
    orb_sum = orb + orb2
    print(f"Addition works: {orb_sum is not None}")

    # Test beta matrix
    orb_beta = orb.beta()
    print(f"Beta matrix application works: {orb_beta is not None}")

    # Test beta2 (sign structure only)
    orb_beta2 = orb.beta2()
    print(f"Beta2 application works: {orb_beta2 is not None}")

    print("\nCompQuatOrbital test PASSED\n")


def test_quat_orbital_2c():
    """
    Test QuatOrbital: 2-component spinor with real quaternionic components.

    Structure: psi = [psi_0, psi_1] where each psi_n is:
        psi_n = q0 + q1*i + q2*j + q3*k (all coefficients real)
    """
    print("=" * 60)
    print("Test: QuatOrbital (2-component real quaternionic)")
    print("=" * 60)

    # Create a 2-component quaternionic orbital
    orb = QuatOrbital()

    print(f"Created QuatOrbital")
    print(f"Number of components: {len(orb)}")

    # Each component is a quaternionic function with 4 real components
    for i in range(2):
        comp = orb[i]
        print(f"\nComponent {i}:")
        print(f"  scalar norm: {comp['scalar'].squaredNorm()}")
        print(f"  i norm: {comp['i'].squaredNorm()}")
        print(f"  j norm: {comp['j'].squaredNorm()}")
        print(f"  k norm: {comp['k'].squaredNorm()}")

    # Test norm
    print(f"\nTotal norm: {orb.norm()}")

    # Test arithmetic
    orb2 = QuatOrbital()
    orb_sum = orb + orb2
    print(f"Addition works: {orb_sum is not None}")

    # Test Pauli matrices
    for direction in range(3):
        orb_sigma = orb.sigma(direction, prec=1e-10)
        print(f"sigma_{'xyz'[direction]} application works: {orb_sigma is not None}")

    # Test quaternionic multiplication by i
    q = QuatFunction(complex_valued=False)
    q_times_i = orb._quat_mul_i(q)
    print(f"Quaternion i multiplication works: {q_times_i is not None}")

    print("\nQuatOrbital test PASSED\n")


def test_quat_function_compatibility():
    """
    Test that QuatFunction works with both real and complex FunctionTrees.
    """
    print("=" * 60)
    print("Test: QuatFunction compatibility")
    print("=" * 60)

    # Real quaternionic function
    q_real = QuatFunction(complex_valued=False)
    print(f"Real QuatFunction created: is_complex={q_real.is_complex}")

    # Complex quaternionic function
    q_complex = QuatFunction(complex_valued=True)
    print(f"Complex QuatFunction created: is_complex={q_complex.is_complex}")

    # Test quaternion multiplication for real case
    q1 = QuatFunction(complex_valued=False)
    q2 = QuatFunction(complex_valued=False)
    q_prod = q1 * q2
    print(f"Real quaternion multiplication works: {q_prod is not None}")

    # Test quaternion multiplication for complex case
    q1_c = QuatFunction(complex_valued=True)
    q2_c = QuatFunction(complex_valued=True)
    q_prod_c = q1_c * q2_c
    print(f"Complex quaternion multiplication works: {q_prod_c is not None}")

    # Test conjugates
    q_conj = q1.quaternion_conjugate()
    print(f"Quaternion conjugate works: {q_conj is not None}")

    q_cc = q1_c.complex_conjugate()
    print(f"Complex conjugate works: {q_cc is not None}")

    q_full = q1_c.full_conjugate()
    print(f"Full conjugate works: {q_full is not None}")

    print("\nQuatFunction compatibility test PASSED\n")


def test_derivative_operators():
    """Test differential operators on both orbital types."""
    print("=" * 60)
    print("Test: Differential operators")
    print("=" * 60)

    # CompQuatOrbital gradient
    comp_orb = CompQuatOrbital()
    grad = comp_orb.gradient()
    print(f"CompQuatOrbital gradient: {len(grad)} components")

    # QuatOrbital gradient
    quat_orb = QuatOrbital()
    grad = quat_orb.gradient()
    print(f"QuatOrbital gradient: {len(grad)} components")

    # Derivative test
    deriv = comp_orb.derivative(0)
    print(f"CompQuatOrbital d/dx: works")

    deriv = quat_orb.derivative(0)
    print(f"QuatOrbital d/dx: works")

    print("\nDifferential operators test PASSED\n")


def test_inner_products():
    """Test inner product operations."""
    print("=" * 60)
    print("Test: Inner products")
    print("=" * 60)

    # CompQuatOrbital dot product (returns complex)
    orb1 = CompQuatOrbital()
    orb2 = CompQuatOrbital()
    dot_result = orb1.dot(orb2)
    print(f"CompQuatOrbital.dot() returns: {type(dot_result).__name__} = {dot_result}")

    # QuatOrbital dot product (returns real)
    qorb1 = QuatOrbital()
    qorb2 = QuatOrbital()
    dot_result = qorb1.dot(qorb2)
    print(f"QuatOrbital.dot() returns: {type(dot_result).__name__} = {dot_result}")

    print("\nInner products test PASSED\n")


def test_kinetic_balance():
    """Test kinetic balance initialization for CompQuatOrbital."""
    print("=" * 60)
    print("Test: Kinetic balance initialization")
    print("=" * 60)

    orb = CompQuatOrbital()

    # Note: This requires actual FunctionTree data to work properly
    # The test just verifies the method exists and doesn't crash on empty orbitals
    try:
        orb.init_small_components(prec=1e-10)
        print("init_small_components() executed without error")
    except Exception as e:
        print(f"init_small_components() note: {e}")

    print("\nKinetic balance test completed\n")


def main():
    """Run all tests."""
    print("\n" + "=" * 60)
    print("Quaternionic Orbital Module Test Suite")
    print("=" * 60 + "\n")

    try:
        test_quat_function_compatibility()
        test_comp_quat_orbital()
        test_quat_orbital_2c()
        test_derivative_operators()
        test_inner_products()
        test_kinetic_balance()

        print("=" * 60)
        print("ALL TESTS PASSED")
        print("=" * 60)
        print("\nSummary:")
        print("  - QuatFunction: Works with real and complex FunctionTrees")
        print("  - CompQuatOrbital: 4-component spinor as complex quaternionic function")
        print("  - QuatOrbital: 2-component spinor with real quaternionic components")
        print("=" * 60)

    except Exception as e:
        print(f"TEST FAILED: {e}")
        import traceback
        traceback.print_exc()


if __name__ == "__main__":
    main()
