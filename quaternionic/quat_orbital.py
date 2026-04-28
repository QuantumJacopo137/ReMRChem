"""
Quaternionic orbital classes for relativistic quantum chemistry.

This module provides two distinct orbital classes:

1. CompQuatOrbital: A complex quaternionic orbital representing a 4-component
   spinor as a single quaternionic function with 4 complex components.
   Structure: q = q0 + q1*i + q2*j + q3*k where each qn is complex-valued.

2. QuatOrbital: A 2-component spinor where each component is a real
   quaternionic function.
   Structure: psi = [psi_0, psi_1] where each psi_n is a real quaternionic function.

Both classes provide the full suite of orbital operations (density, dot product,
Dirac matrices, etc.) optimized for quaternionic algebra.
"""
from vampyr import vampyr3d as vp
import numpy as np
from typing import Optional, List, Union

from .quat_function import QuatFunction as QuatFunction


# =============================================================================
# CompQuatOrbital: Complex Quaternionic Orbital (4-component spinor equivalent)
# =============================================================================

class CompQuatOrbital:
    """
    Complex quaternionic orbital for 4-component relativistic spinors.

    Represents a Dirac spinor as a single complex quaternionic function:
        q(x) = q0(x) + q1(x)*i + q2(x)*j + q3(x)*k

    where each component qn is a complex function (real + imaginary parts).

    This provides a compact representation of 4-component spinors:
        - q0 encodes the Large-alpha component
        - q1 encodes the Large-beta component
        - q2 encodes the Small-alpha component
        - q3 encodes the Small-beta component

    Attributes:
        quat: Complex quaternionic function containing all 4 spinor components
        mra: Shared MultiResolutionAnalysis object
        light_speed: Speed of light in atomic units
    """

    mra = None  # Shared MultiResolutionAnalysis object
    light_speed = -1.0  # Speed of light (atomic units, -1 = use default)

    # Component mapping: quaternion component -> spinor component
    comp_dict = {'scalar': 0, 'i': 1, 'j': 2, 'k': 3}

    def __init__(self):
        """Initialize a complex quaternionic orbital."""
        self.quat = QuatFunction(complex_valued=True)

    def __getitem__(self, key: str) -> 'ComponentView':
        """
        Access a spinor component as a complex function view.

        Args:
            key: Component name ('scalar', 'i', 'j', 'k')

        Returns:
            ComponentView object providing access to the complex function
        """
        return ComponentView(self.quat, self.comp_dict[key])

    def __len__(self):
        return 4

    def __str__(self):
        return (f"CompQuatOrbital (4-component complex spinor):\n"
                f"  (scalar): {self.quat['scalar']}\n"
                f"  (i):      {self.quat['i']}\n"
                f"  (j):      {self.quat['j']}\n"
                f"  (k):      {self.quat['k']}")

    def setZero(self):
        """Set all components to zero."""
        self.quat.setZero()

    def copy(self, other: 'CompQuatOrbital'):
        """Copy all components from another orbital."""
        self.quat.copy(other.quat)

    # =========================================================================
    # Norm operations
    # =========================================================================

    def squaredNorm(self) -> float:
        """Compute the squared L2 norm: sum of all 4 component norms."""
        return self.quat.squaredNorm()

    def norm(self) -> float:
        """Compute the L2 norm."""
        return np.sqrt(self.squaredNorm())

    def normalize(self):
        """Normalize the orbital to unit norm."""
        norm = self.norm()
        if norm > 1e-14:
            self.quat.rescale(1.0 / norm)

    def rescale(self, factor: Union[float, complex]):
        """Rescale all components by a factor."""
        self.quat.rescale(factor)

    def squaredLargeNorm(self) -> float:
        """Compute squared norm of large components (La, Lb)."""
        return (self.quat['scalar'].squaredNorm() +
                self.quat['i'].squaredNorm())

    def squaredSmallNorm(self) -> float:
        """Compute squared norm of small components (Sa, Sb)."""
        return (self.quat['j'].squaredNorm() +
                self.quat['k'].squaredNorm())

    # =========================================================================
    # Arithmetic operations
    # =========================================================================

    def __add__(self, other: 'CompQuatOrbital') -> 'CompQuatOrbital':
        """Add two complex quaternionic orbitals."""
        if not isinstance(other, CompQuatOrbital):
            return NotImplemented
        output = CompQuatOrbital()
        output.quat = self.quat + other.quat
        return output

    def __sub__(self, other: 'CompQuatOrbital') -> 'CompQuatOrbital':
        """Subtract two complex quaternionic orbitals."""
        if not isinstance(other, CompQuatOrbital):
            return NotImplemented
        output = CompQuatOrbital()
        output.quat = self.quat - other.quat
        return output

    def __neg__(self) -> 'CompQuatOrbital':
        """Negate the orbital."""
        output = CompQuatOrbital()
        output.quat = -self.quat
        return output

    def __rmul__(self, scalar: Union[float, complex]) -> 'CompQuatOrbital':
        """Multiply by a scalar from the left."""
        return self.__mul__(scalar)

    def __mul__(self, other: Union['CompQuatOrbital', float, complex]) -> 'CompQuatOrbital':
        """
        Multiply by a scalar or another orbital.

        For orbital * orbital, performs component-wise multiplication.
        """
        output = CompQuatOrbital()

        if isinstance(other, (int, float, complex)):
            output.quat = other * self.quat
            return output

        if isinstance(other, CompQuatOrbital):
            # Component-wise quaternion multiplication
            output.quat = self.quat * other.quat
            return output

        return NotImplemented

    # =========================================================================
    # Component operations
    # =========================================================================

    def copy_component(self, func, component: str = 'La'):
        """
        Copy a complex function into a spinor component.

        Args:
            func: Source function (complex_fcn or compatible)
            component: Target component name
        """
        idx = self.comp_dict[component]
        if idx == 0:
            self.quat['scalar'].copy_fcns(func.real, func.imag)
        elif idx == 1:
            self.quat['i'].copy_fcns(func.real, func.imag)
        elif idx == 2:
            self.quat['j'].copy_fcns(func.real, func.imag)
        elif idx == 3:
            self.quat['k'].copy_fcns(func.real, func.imag)

    def copy_components(self, La=None, Lb=None, Sa=None, Sb=None):
        """Copy multiple components at once."""
        if La is not None:
            self.copy_component(La, 'La')
        if Lb is not None:
            self.copy_component(Lb, 'Lb')
        if Sa is not None:
            self.copy_component(Sa, 'Sa')
        if Sb is not None:
            self.copy_component(Sb, 'Sb')

    def init_small_components(self, prec: float):
        """
        Initialize small components from large components using kinetic balance.

        For the Dirac equation: psi_S = (sigma·p) / (2mc) * psi_L

        In quaternionic form, this becomes:
            q_small = -i/(2c) * (sigma·grad) q_large

        where q_large = q0 + q1*i and q_small = q2 + q3*i.
        """
        c = CompQuatOrbital.light_speed
        if c < 0:
            print("Warning: light_speed not set. Using default value.")
            c = 137.035999084

        from orbital4c.complex_fcn import complex_fcn

        # Get large component gradients
        q0 = self.quat['scalar']  # La
        q1 = self.quat['i']       # Lb

        grad_q0 = q0.gradient()
        grad_q1 = q1.gradient()

        # sigma·p acting on [q0, q1]:
        # (sigma·p) [q0] = [dz*q0 + (dx-idy)*q1] = [dz*q0 + dx*q1 - i*dy*q1]
        # (sigma·p) [q1] = [(dx+idy)*q0 - dz*q1] = [dx*q0 + i*dy*q0 - dz*q1]

        # Sa (q2) = -i/(2c) * (dz*q0 + dx*q1 - i*dy*q1)
        # Sb (q3) = -i/(2c) * (dx*q0 + i*dy*q0 - dz*q1)

        factor = -1.0 / (2 * c)

        # Compute Sa component (stored in quat['j'])
        sa_result = complex_fcn()

        # Real part of Sa: factor * (dz*q0.imag + dx*q1.imag + dy*q1.real)
        # Imag part of Sa: factor * (dz*q0.real + dx*q1.real - dy*q1.imag)

        temp_real = vp.FunctionTree(self.mra)
        temp_imag = vp.FunctionTree(self.mra)
        temp_real.setZero()
        temp_imag.setZero()

        add_terms_r = []
        add_terms_i = []

        # dz * q0
        if grad_q0[2].real.squaredNorm() > 0:
            add_terms_i.append((factor, grad_q0[2].real))
        if grad_q0[2].imag.squaredNorm() > 0:
            add_terms_r.append((factor, grad_q0[2].imag))

        # dx * q1
        if grad_q1[0].real.squaredNorm() > 0:
            add_terms_i.append((factor, grad_q1[0].real))
        if grad_q1[0].imag.squaredNorm() > 0:
            add_terms_r.append((factor, grad_q1[0].imag))

        # dy * q1 (note: -i*dy*q1 contributes +dy*q1.imag to real, -dy*q1.real to imag)
        if grad_q1[1].imag.squaredNorm() > 0:
            add_terms_r.append((factor, grad_q1[1].imag))
        if grad_q1[1].real.squaredNorm() > 0:
            add_terms_i.append((-factor, grad_q1[1].real))

        if add_terms_r:
            vp.advanced.add(prec, temp_real, add_terms_r)
        if add_terms_i:
            vp.advanced.add(prec, temp_imag, add_terms_i)

        sa_result.real = temp_real
        sa_result.imag = temp_imag
        self.quat['j'].copy_fcns(sa_result.real, sa_result.imag)

        # Compute Sb component (stored in quat['k'])
        sb_result = complex_fcn()

        temp_real = vp.FunctionTree(self.mra)
        temp_imag = vp.FunctionTree(self.mra)
        temp_real.setZero()
        temp_imag.setZero()

        add_terms_r = []
        add_terms_i = []

        # dx * q0
        if grad_q0[0].real.squaredNorm() > 0:
            add_terms_i.append((factor, grad_q0[0].real))
        if grad_q0[0].imag.squaredNorm() > 0:
            add_terms_r.append((factor, grad_q0[0].imag))

        # dy * q0 (i*dy*q0 contributes -dy*q0.imag to real, +dy*q0.real to imag)
        if grad_q0[1].imag.squaredNorm() > 0:
            add_terms_r.append((-factor, grad_q0[1].imag))
        if grad_q0[1].real.squaredNorm() > 0:
            add_terms_i.append((factor, grad_q0[1].real))

        # -dz * q1
        if grad_q1[2].real.squaredNorm() > 0:
            add_terms_i.append((-factor, grad_q1[2].real))
        if grad_q1[2].imag.squaredNorm() > 0:
            add_terms_r.append((-factor, grad_q1[2].imag))

        if add_terms_r:
            vp.advanced.add(prec, temp_real, add_terms_r)
        if add_terms_i:
            vp.advanced.add(prec, temp_imag, add_terms_i)

        sb_result.real = temp_real
        sb_result.imag = temp_imag
        self.quat['k'].copy_fcns(sb_result.real, sb_result.imag)

    # =========================================================================
    # Differential operators
    # =========================================================================

    def gradient(self, der: str = 'ABGV') -> List['CompQuatOrbital']:
        """
        Compute the gradient of the orbital.

        Returns a list of 3 CompQuatOrbitals representing [d/dx, d/dy, d/dz].
        """
        result = []
        for d in range(3):
            grad_orb = CompQuatOrbital()
            grad_orb.quat = self.quat.derivative(d, der)
            result.append(grad_orb)
        return result

    def derivative(self, direction: int = 0, der: str = 'ABGV') -> 'CompQuatOrbital':
        """Compute partial derivative with respect to a coordinate."""
        output = CompQuatOrbital()
        output.quat = self.quat.derivative(direction, der)
        return output

    def complex_conj(self) -> 'CompQuatOrbital':
        """Compute the complex conjugate of the orbital."""
        output = CompQuatOrbital()
        output.quat = self.quat.complex_conjugate()
        return output

    # =========================================================================
    # Density and overlap operations
    # =========================================================================

    def density(self, prec: float) -> vp.FunctionTree:
        """
        Compute the electron density: rho = |psi|^2 = sum_i |psi_i|^2

        Returns a real-valued FunctionTree.
        """
        density = vp.FunctionTree(self.mra)
        density.setZero()

        add_vec = []
        for comp_name in ['La', 'Lb', 'Sa', 'Sb']:
            comp = self[comp_name]
            temp = comp.density(prec)
            if temp.squaredNorm() > 0:
                add_vec.append((1.0, temp))

        if add_vec:
            vp.advanced.add(prec, density, add_vec)

        return density

    def overlap_density(self, other: 'CompQuatOrbital', prec: float) -> vp.FunctionTree:
        """
        Compute the overlap density: rho_ij = psi_i^dagger * psi_j

        Returns a real-valued FunctionTree.
        """
        density = vp.FunctionTree(self.mra)
        density.setZero()

        add_vec = []
        for comp_name in ['La', 'Lb', 'Sa', 'Sb']:
            psi_i = self[comp_name]
            psi_j = other[comp_name]
            temp = psi_i.complex_conj() * psi_j
            temp_crop = temp.density(prec)
            if temp_crop.squaredNorm() > 0:
                add_vec.append((1.0, temp_crop))

        if add_vec:
            vp.advanced.add(prec, density, add_vec)

        return density

    # =========================================================================
    # Dirac matrix operations
    # =========================================================================

    # def alpha(self, direction: int, prec: float) -> 'CompQuatOrbital':
    #     """
    #     Apply the alpha (Dirac) matrix in a given direction.

    #     In the quaternionic representation, alpha matrices act as:
    #     - alpha_x: swaps (La,Sb) and (Lb,Sa)
    #     - alpha_y: swaps with imaginary phase factors
    #     - alpha_z: swaps (La,Sa) and (Lb,-Sb)
    #     """
    #     output = CompQuatOrbital()

    #     q = self.quat.components

    #     if direction == 0:  # alpha_x
    #         # alpha_x = [[0,0,0,1], [0,0,1,0], [0,1,0,0], [1,0,0,0]]
    #         # La -> Sb, Lb -> Sa, Sa -> Lb, Sb -> La
    #         output.quat['scalar'] = q[3]  # La_new = Sb
    #         output.quat['i'] = q[2]       # Lb_new = Sa
    #         output.quat['j'] = q[1]       # Sa_new = Lb
    #         output.quat['k'] = q[0]       # Sb_new = La

    #     elif direction == 1:  # alpha_y
    #         # alpha_y = [[0,0,0,-i], [0,0,i,0], [0,-i,0,0], [i,0,0,0]]
    #         # La -> -i*Sb, Lb -> i*Sa, Sa -> -i*Lb, Sb -> i*La
    #         output.quat['scalar'] = -1j * q[3]
    #         output.quat['i'] = 1j * q[2]
    #         output.quat['j'] = -1j * q[1]
    #         output.quat['k'] = 1j * q[0]

    #     elif direction == 2:  # alpha_z
    #         # alpha_z = [[0,0,1,0], [0,0,0,-1], [1,0,0,0], [0,-1,0,0]]
    #         # La -> Sa, Lb -> -Sb, Sa -> La, Sb -> -Lb
    #         output.quat['scalar'] = q[2]
    #         output.quat['i'] = -q[3]
    #         output.quat['j'] = q[0]
    #         output.quat['k'] = -q[1]

    #     output.quat.crop(prec)
    #     return output

    # def alpha_p(self, prec: float, der: str = "ABGV") -> 'CompQuatOrbital':
    #     """
    #     Apply the alpha·p operator: -i * alpha · gradient

    #     This is the kinetic term in the Dirac Hamiltonian.
    #     """
    #     grad = self.gradient(der)

    #     # alpha·p = -i * (alpha_x * d/dx + alpha_y * d/dy + alpha_z * d/dz)
    #     result = CompQuatOrbital()

    #     for d in range(3):
    #         alpha_d = grad[d].alpha(d, prec)
    #         result = result + (-1j) * alpha_d

    #     result.quat.crop(prec)
    #     return result

    # def beta(self, shift: float = 0.0) -> 'CompQuatOrbital':
    #     """
    #     Apply the beta matrix (with optional energy shift).

    #     Beta = diag(1, 1, -1, -1) * c^2 + shift
    #     In quaternionic form: large components get +c^2, small get -c^2
    #     """
    #     output = CompQuatOrbital()

    #     c = CompQuatOrbital.light_speed
    #     if c < 0:
    #         c = 137.035999084

    #     c2 = c * c

    #     # Large components (La, Lb) -> +c^2 + shift
    #     output.quat['scalar'] = (c2 + shift) * self.quat['scalar']
    #     output.quat['i'] = (c2 + shift) * self.quat['i']

    #     # Small components (Sa, Sb) -> -c^2 + shift
    #     output.quat['j'] = (-c2 + shift) * self.quat['j']
    #     output.quat['k'] = (-c2 + shift) * self.quat['k']

    #     return output

    # def beta2(self) -> 'CompQuatOrbital':
    #     """Apply beta matrix without c^2 factor (just sign structure)."""
    #     output = CompQuatOrbital()

    #     # Large components: +1, Small components: -1
    #     output.quat['scalar'] = self.quat['scalar']
    #     output.quat['i'] = self.quat['i']
    #     output.quat['j'] = -self.quat['j']
    #     output.quat['k'] = -self.quat['k']

    #     return output

    # =========================================================================
    # Kramers' Time-Reversal Symmetry
    # =========================================================================

    def ktrs(self, prec: float) -> 'CompQuatOrbital':
        """
        Apply Kramers' Time-Reversal Symmetry operation.

        For a 4-component spinor psi = [La, Lb, Sa, Sb]:
        T psi = [-Lb*, La*, -Sb*, Sa*]

        In quaternionic form:
        q' = -q1* + q0* * i - q3* * j + q2* * k
        """
        output = CompQuatOrbital()

        # Complex conjugate of all components
        q_conj = self.quat.complex_conjugate()

        # Apply KTRS: [-Lb*, La*, -Sb*, Sa*]
        output.quat['scalar'] = -q_conj['i']  # -Lb*
        output.quat['i'] = q_conj['scalar']   # La*
        output.quat['j'] = -q_conj['k']       # -Sb*
        output.quat['k'] = q_conj['j']        # Sa*

        output.quat.crop(prec)
        return output

    # =========================================================================
    # Inner product
    # =========================================================================

    def dot(self, other: 'CompQuatOrbital') -> complex:
        """
        Compute the inner product: <psi | phi> = sum_i <psi_i | phi_i>

        Returns a complex number.
        """
        result = 0j
        for comp_name in ['La', 'Lb', 'Sa', 'Sb']:
            psi_i = self[comp_name]
            phi_i = other[comp_name]
            result += psi_i.dot(phi_i)
        return result

    # =========================================================================
    # Crop and precision operations
    # =========================================================================

    def crop(self, prec: float):
        """Crop all components to the given precision."""
        self.quat.crop(prec)

    def cropLargeSmall(self, prec: float):
        """Crop large and small components with different precisions."""
        largeNorm = np.sqrt(self.squaredLargeNorm())
        smallNorm = np.sqrt(self.squaredSmallNorm())

        precLarge = prec * largeNorm / 10
        precSmall = prec * smallNorm / 10

        self.quat['scalar'].crop(precLarge, True)
        self.quat['i'].crop(precLarge, True)
        self.quat['j'].crop(precSmall, True)
        self.quat['k'].crop(precSmall, True)

    # =========================================================================
    # Save/Load operations
    # =========================================================================

    def save(self, name: str):
        """Save orbital to disk."""
        self.quat.save(f"{name}_quat")

    def load(self, name: str):
        """Load orbital from disk."""
        self.quat.load(f"{name}_quat")

    # =========================================================================
    # Pointwise evaluation
    # =========================================================================

    def __call__(self, position: np.ndarray) -> np.ndarray:
        """
        Evaluate the orbital at a position.

        Returns: numpy array [La, Lb, Sa, Sb] as complex values
        """
        vals = self.quat(position)
        return np.array([vals[0], vals[1], vals[2], vals[3]])


# =============================================================================
# ComponentView: View into a quaternionic component as a complex function
# =============================================================================

class ComponentView:
    """
    Provides a complex function view into a quaternionic component.

    This allows accessing quaternion components as if they were
    complex_fcn objects, enabling compatibility with existing code.
    """

    def __init__(self, quat: QuatFunction, idx: int):
        self.quat = quat
        self.idx = idx

        # Map index to quaternion component name
        self._names = ['scalar', 'i', 'j', 'k']
        self._name = self._names[idx]

    @property
    def real(self):
        """Return the real part of this component."""
        return self.quat[self._name].real

    @property
    def imag(self):
        """Return the imaginary part of this component."""
        return self.quat[self._name].imag

    def squaredNorm(self) -> float:
        """Return squared L2 norm."""
        return self.quat[self._name].squaredNorm()

    def density(self, prec: float) -> vp.FunctionTree:
        """Compute |psi|^2 for this component."""
        return self.quat[self._name].density(prec)

    def complex_conj(self) -> 'ComponentView':
        """Return complex conjugate (creates new view-compatible object)."""
        from orbital4c.complex_fcn import complex_fcn
        result = complex_fcn()
        result.real = self.real
        result.imag = -self.imag
        return result

    def dot(self, other: Union['ComponentView', 'complex_fcn']) -> complex:
        """Compute inner product with another complex function."""
        if hasattr(other, 'real') and hasattr(other, 'imag'):
            return self.quat[self._name].dot(other)
        return self.quat[self._name].dot(other)

    def __mul__(self, other):
        """Multiply by scalar."""
        from orbital4c.complex_fcn import complex_fcn
        result = complex_fcn()
        result.real = self.real * np.real(other) - self.imag * np.imag(other)
        result.imag = self.real * np.imag(other) + self.imag * np.real(other)
        return result

    def __rmul__(self, scalar):
        return self.__mul__(scalar)

    def __neg__(self):
        from orbital4c.complex_fcn import complex_fcn
        result = complex_fcn()
        result.real = -self.real
        result.imag = -self.imag
        return result


# =============================================================================
# QuatOrbital: 2-Component Real Quaternionic Spinor
# =============================================================================

class QuatOrbital:
    """
    Two-component spinor with real quaternionic function components.

    Represents a Pauli spinor as:
        psi = [psi_0, psi_1]

    where each psi_n is a real quaternionic function:
        psi_n = q0 + q1*i + q2*j + q3*k  (all coefficients real)

    This provides a representation of 2-component spinors using
    quaternionic algebra, useful for non-relativistic and
    Pauli-equation calculations.

    Attributes:
        components: List of 2 real QuatFunction objects
        mra: Shared MultiResolutionAnalysis object
    """

    mra = None  # Shared MultiResolutionAnalysis object

    def __init__(self):
        """Initialize a 2-component real quaternionic orbital."""
        self.components = [QuatFunction(complex_valued=False),
                           QuatFunction(complex_valued=False)]

    def __getitem__(self, idx: int) -> QuatFunction:
        """Access a spinor component (0 or 1)."""
        return self.components[idx]

    def __setitem__(self, idx: int, value: QuatFunction):
        """Set a spinor component (0 or 1)."""
        self.components[idx] = value

    def __len__(self):
        return 2

    def __str__(self):
        return (f"QuatOrbital (2-component real quaternionic):\n"
                f"--> Component 0: {self.components[0]}\n"
                f"--> Component 1: {self.components[1]}")

    def setZero(self):
        """Set all components to zero."""
        for comp in self.components:
            comp.setZero()

    def copy(self, other: 'QuatOrbital'):
        """Copy all components from another orbital."""
        for i in range(2):
            self.components[i].copy(other.components[i])

    # =========================================================================
    # Norm operations
    # =========================================================================

    def squaredNorm(self) -> float:
        """Compute the squared L2 norm: sum of both component norms."""
        return self.components[0].squaredNorm() + self.components[1].squaredNorm()

    def norm(self) -> float:
        """Compute the L2 norm."""
        return np.sqrt(self.squaredNorm())

    def normalize(self):
        """Normalize the orbital to unit norm."""
        norm = self.norm()
        print(f"Normalizing QuatOrbital with norm = {norm}")    
        renorm = 1.0 / norm if norm > 1e-14 else 1.0
        if norm > 1e-14:
            for i in range(2):
                for j in range(4):
                    self.components[i][j] *= renorm

    def rescale(self, factor: float):
        """Rescale all components by a real factor."""
        for i in range(2):
            self.components[i].rescale(factor)

    def print_components_Sqnorms(self):
        """Print norms of each component for debugging."""
        print(f"Component 0 norm: {np.sqrt(self.components[0].squaredNorm())}")
        print(f"Component 1 norm: {np.sqrt(self.components[1].squaredNorm())}")

    def print_all_Sqnorms(self):
        """Print norms of all quaternion components for debugging."""
        print("---- QuatOrbital Component Squared Norms ----")
        print()
        for i in range(2):
                
            comp = self.components[i][0]  # Get the j-th quaternion component of the i-th spinor
            print(f"  Component [{i}][s] squared norm: {comp.squaredNorm()}")
            comp = self.components[i][1]
            print(f"  Component [{i}][i] squared norm: {comp.squaredNorm()}")
            comp = self.components[i][2]
            print(f"  Component [{i}][j] squared norm: {comp.squaredNorm()}")
            comp = self.components[i][3]    
            print(f"  Component [{i}][k] squared norm: {comp.squaredNorm()}")
            print()
            
        print(" Norm = ", self.norm())
        print()
        print("---------------------------------------------")


    # =========================================================================
    # Arithmetic operations
    # =========================================================================

    def __add__(self, other: 'QuatOrbital') -> 'QuatOrbital':
        """Add two quaternionic orbitals component-wise."""
        if not isinstance(other, QuatOrbital):
            return NotImplemented
        output = QuatOrbital()
        for i in range(2):
            output.components[i] = self.components[i] + other.components[i]
        return output

    def __sub__(self, other: 'QuatOrbital') -> 'QuatOrbital':
        """Subtract two quaternionic orbitals component-wise."""
        if not isinstance(other, QuatOrbital):
            return NotImplemented
        output = QuatOrbital()
        for i in range(2):
            output.components[i] = self.components[i] - other.components[i]
        return output

    def __neg__(self) -> 'QuatOrbital':
        """Negate the orbital."""
        output = QuatOrbital()
        for i in range(2):
            output.components[i] = -self.components[i]
        return output

    def __rmul__(self, scalar: float) -> 'QuatOrbital':
        """Multiply by a real scalar from the left."""
        return self.__mul__(scalar)

    def __mul__(self, other: Union['QuatOrbital', float]) -> 'QuatOrbital':
        """
        Multiply by a scalar or another orbital.

        For orbital * orbital, performs quaternionic multiplication
        component-wise.
        """
        output = QuatOrbital()

        if isinstance(other, (int, float)):
            for i in range(2):
                output.components[i] = other * self.components[i]
            return output

        if isinstance(other, QuatOrbital):
            # Component-wise quaternion multiplication
            for i in range(2):
                output.components[i] = self.components[i] * other.components[i]
            return output

        return NotImplemented

    # =========================================================================
    # Differential operators
    # =========================================================================

    def gradient(self, der: str = 'ABGV') -> List['QuatOrbital']:
        """
        Compute the gradient of the orbital.

        Returns a list of 3 QuatOrbitals representing [d/dx, d/dy, d/dz].
        """
        result = []
        for d in range(3):
            grad_orb = QuatOrbital()
            for i in range(2):
                grad_orb.components[i] = self.components[i].derivative(d, der)
            result.append(grad_orb)
        return result

    def derivative(self, direction: int = 0, der: str = 'ABGV') -> 'QuatOrbital':
        """Compute partial derivative with respect to a coordinate."""
        output = QuatOrbital()
        for i in range(2):
            output.components[i] = self.components[i].derivative(direction, der)
        return output
    
    def classicT(self) -> float:
        """Return the classical kinetic energy prefactor."""
        grad = self.gradient()
        k = 0.5 * (grad[0].squaredNorm() + grad[1].squaredNorm() + grad[2].squaredNorm())
        return k

    # =========================================================================
    # Density and overlap operations
    # =========================================================================

    def density(self, prec: float) -> vp.FunctionTree:
        """
        Compute the electron density: rho = |psi_0|^2 + |psi_1|^2

        For quaternionic functions, |q|^2 = q0^2 + q1^2 + q2^2 + q3^2
        Returns a real-valued FunctionTree.
        """
        density = vp.FunctionTree(self.mra)
        density.setZero()

        add_vec = []
        for comp in self.components:
            # Each component is a quaternionic function
            # Its density is sum of squares of all 4 quaternion components
            for j in range(4):
                temp = vp.FunctionTree(self.mra)
                temp.setZero()
                if comp[j].squaredNorm() > 0:
                    vp.advanced.multiply(prec, temp, 1.0, comp[j], comp[j])
                    add_vec.append((1.0, temp))

        if add_vec:
            vp.advanced.add(prec, density, add_vec)

        return density

    def overlap_density(self, other: 'QuatOrbital', prec: float) -> vp.FunctionTree:
        """
        Compute the overlap density: rho = psi_0^* psi_0 + psi_1^* psi_1

        For quaternionic functions, uses quaternionic conjugate.
        Returns a real-valued FunctionTree.
        """
        density = vp.FunctionTree(self.mra)
        density.setZero()

        add_vec = []
        for i in range(2):
            # Quaternionic inner product: q^* * p
            q_conj = self.components[i].quaternion_conjugate()
            prod = q_conj * other.components[i]
            # Density is the scalar part of the product
            for j in range(4):
                temp = vp.FunctionTree(self.mra)
                temp.setZero()
                if prod[j].squaredNorm() > 0:
                    vp.advanced.multiply(prec, temp, 1.0, prod[j], prod[j])
                    add_vec.append((1.0, temp))

        if add_vec:
            vp.advanced.add(prec, density, add_vec)

        return density

    # =========================================================================
    # Pauli matrix operations
    # =========================================================================

    #def sigma(self, direction: int, prec: float) -> 'QuatOrbital':
    #    """
    #    Apply the Pauli matrix in a given direction.
#
    #    For 2-component spinors, Pauli matrices act as:
    #    - sigma_x: swaps components
    #    - sigma_y: swaps with quaternionic i factor
    #    - sigma_z: signs on components

    #    In the quaternionic representation, these become operations
    #    on the quaternionic components.
    #    """
    #    output = QuatOrbital()

    #    if direction == 0:  # sigma_x
    #        # [[0,1], [1,0]]: swaps psi_0 and psi_1
    #        output.components[0] = self.components[1]
    #        output.components[1] = self.components[0]

    #    elif direction == 1:  # sigma_y
    #        # [[0,-i], [i,0]]: swaps with quaternionic rotation
    #        # In quaternionic form: multiply by quaternion unit
    #        output.components[0] = self._quat_mul_i(self.components[1])
    #        output.components[1] = self._quat_mul_minus_i(self.components[0])

    #    elif direction == 2:  # sigma_z
    #        # [[1,0], [0,-1]]: sign on second component
    #        output.components[0] = self.components[0]
    #        output.components[1] = -self.components[1]

    #    output.crop(prec)
    #    return output

    #def _quat_mul_i(self, q: QuatFunction) -> QuatFunction:
    #    """Multiply quaternionic function by quaternion unit i."""
    #    result = QuatFunction(complex_valued=False)
    #    # i * (q0 + q1*i + q2*j + q3*k) = -q1 + q0*i + q3*j - q2*k
    #    result['scalar'] = -q['i']
    #    result['i'] = q['scalar']
    #    result['j'] = q['k']
    #    result['k'] = -q['j']
    #    return result

    #def _quat_mul_minus_i(self, q: QuatFunction) -> QuatFunction:
    #    """Multiply quaternionic function by -i."""
    #    result = QuatFunction(complex_valued=False)
    #    # -i * (q0 + q1*i + q2*j + q3*k) = q1 - q0*i - q3*j + q2*k
    #    result['scalar'] = q['i']
    #    result['i'] = -q['scalar']
    #    result['j'] = -q['k']
    #    result['k'] = q['j']
    #    return result

    #def sigma__p(self, prec: float, der: str = "ABGV") -> 'QuatOrbital':
    #    """
    #    Apply the sigma·p operator: -i * sigma · gradient
#
    #    This is the kinetic term for 2-component spinors.
    #    """
    #    grad = self.gradient(der)
#
    #    # sigma·p = -i * (sigma_x * d/dx + sigma_y * d/dy + sigma_z * d/dz)
    #    result = QuatOrbital()
#
    #    for d in range(3):
    #        sigma_d = grad[d].sigma(d, prec)
    #        result = result + (-1) * sigma_d  # -1 for real quaternionic
#
    #    result.crop(prec)
    #    return result

    # =========================================================================
    # Inner product
    # =========================================================================

    def dot(self, other: 'QuatOrbital') -> float:
        """
        Compute the inner product: <psi | phi> = sum_i <psi_i | phi_i>

        Returns a real number (quaternionic inner product is real-valued).
        """
        result = 0.0
        for i in range(2):
            result += self[i].dot(other[i])
            
        return result

    # =========================================================================
    # Crop and precision operations
    # =========================================================================

    def crop_Top_Bot(self, prec: float):
        """Crop all components to the given precision."""
        Top_norm = np.sqrt(self.components[0].squaredNorm())
        Bot_norm = np.sqrt(self.components[1].squaredNorm())
        Top_crop_prec = prec  / 10
        Bot_crop_prec = prec  / 10
        self.components[0].crop(prec / 10)
        self.components[1].crop(prec / 10)
        

    # =========================================================================
    # Save/Load operations
    # =========================================================================

    def save(self, name: str):
        """Save orbital to disk."""
        self.components[0].save(f"{name}_comp0")
        self.components[1].save(f"{name}_comp1")

    def load(self, name: str):
        """Load orbital from disk."""
        self.components[0].load(f"{name}_comp0")
        self.components[1].load(f"{name}_comp1")

    # =========================================================================
    # Pointwise evaluation
    # =========================================================================

    def __call__(self, position: np.ndarray) -> np.ndarray:
        """
        Evaluate the orbital at a position.

        Returns: numpy array [psi_0, psi_1] as quaternionic values (4-vectors)
        """
        return np.array([self.components[0](position),
                         self.components[1](position)])


# =============================================================================
# Utility functions for CompQuatOrbital
# =============================================================================

    def beta_mc2(orbital: 'QuatOrbital') -> 'QuatOrbital':
        """
        Apply the beta matrix (with optional energy shift).

        Beta = c^2 * (0,1;1,0) + shift * I
        In quaternionic form: large components get +c^2, small get -c^2
        """

        

        result = QuatOrbital()
        c = orbital[0].light_speed
        if c < 0:
            print("Warning: light speed not set, using default 137.035999084")
            c = 137.035999084
        
        c2 = c * c
        result[0] = orbital[1] * c2  
        result[1] = orbital[0] * c2  
        
        return result      

    def c_alpha_p(orbital: 'QuatOrbital', prec: float, der: str = "ABGV") -> 'QuatOrbital':
        """
        Apply the alpha·p operator: -i * alpha · gradient

        This is the kinetic term in the Dirac Hamiltonian.
        """
        

        # alpha·p = -i * (alpha_x * d/dx + alpha_y * d/dy + alpha_z * d/dz)
        result = QuatOrbital()
        c = orbital[0].light_speed
        result[0] = -orbital[0].sigma_p(prec, der)
        result[1] = orbital[1].sigma_p(prec, der)
        
        result[0].rescale(c)
        result[1].rescale(c)


        return result



def apply_dirac_hamiltonian(orbital: QuatOrbital, prec: float, der: str = 'ABGV', shift: float = 0.0) -> QuatOrbital:
    """
    Apply the Dirac Hamiltonian to a complex quaternionic orbital.

    H_D = beta * c^2 + c * alpha · p

    Args:
        orbital: Input orbital
        prec: Precision for cropping
        shift: Energy shift
        der: Derivative type

    Returns:
        H_D psi
    """
    
    c2_beta_phi = orbital.beta_mc2()
    c_alpha_p_phi = orbital.c_alpha_p(prec, der) 
    output = c2_beta_phi + c_alpha_p_phi
    if shift != 0.0:
        output = output + shift * orbital

    output.crop_Top_Bot(prec)
    return output


def apply_potential(factor: float, potential: vp.FunctionTree,
                    orbital: QuatOrbital, prec: float) -> QuatOrbital:
    """
    Apply a scalar potential to a complex quaternionic orbital.

    Args:
        factor: Multiplicative factor
        potential: Potential as a FunctionTree
        orbital: Input orbital
        prec: Precision

    Returns:
        V * psi
    """

    output = QuatOrbital()
    for i in range(2):
        comp = orbital[i]
        result = comp.real_function_times(potential)
        output[i] = factor * result
    return output


def apply_helmholtz(orbital:QuatOrbital, mu: float,
                    prec: float) -> QuatOrbital:
    """
    Apply the Helmholtz operator to an orbital.

    Args:
        orbital: Input orbital
        mu: Helmholtz parameter
        prec: Precision

    Returns:
        (-nabla^2 + mu^2)^{-1} psi
    """
    output = QuatOrbital()
    for i in range(2):
        comp = orbital[i]
        # Apply Helmoltz quat
        output[i] = comp.apply_helmoltz(mu, prec)
        output[i] *= -1.0/(2*np.pi)
        
    return output


def add_vector(orbital_array: List[CompQuatOrbital],
               coeff_array: List[complex],
               prec: float) -> CompQuatOrbital:
    """
    Compute a linear combination of orbitals: sum_i c_i * psi_i

    Args:
        orbital_array: List of orbitals
        coeff_array: List of coefficients
        prec: Precision

    Returns:
        Linear combination of orbitals
    """
    if not orbital_array:
        return CompQuatOrbital()

    output = CompQuatOrbital()
    output.setZero()

    for coeff, orb in zip(coeff_array, orbital_array):
        output = output + coeff * orb

    output.crop(prec)
    return output


# =============================================================================
# Utility functions for QuatOrbital (2-component real)
# =============================================================================

def apply_pauli_hamiltonian(orbital: QuatOrbital, prec: float,
                            potential: vp.FunctionTree = None,
                            magnetic_field: np.ndarray = None) -> QuatOrbital:
    """
    Apply the Pauli Hamiltonian to a 2-component quaternionic orbital.

    H = (1/2) * (sigma·p)^2 + V

    Args:
        orbital: Input orbital
        prec: Precision
        potential: Scalar potential (optional)
        magnetic_field: Magnetic field vector (optional)

    Returns:
        H psi
    """
    # Kinetic term: (1/2) * (sigma·p)^2
    sigma_p_orb = orbital.sigma_p(prec)

    result = QuatOrbital()
    for i in range(2):
        # Apply sigma·p again and multiply by -1/2
        grad = sigma_p_orb.gradient()
        for d in range(3):
            sigma_d = grad[d].sigma(d, prec)
            result = result + (-0.5) * sigma_d

    # Add potential term
    if potential is not None:
        for i in range(2):
            if orbital[i].squaredNorm() > 0:
                temp = vp.FunctionTree(orbital.mra)
                temp.setZero()
                for j in range(4):
                    if orbital[i][j].squaredNorm() > 0:
                        vp.advanced.multiply(prec, temp, 1.0, potential, orbital[i][j])
                result[i]['scalar'] = temp

    return result
