"""
Quaternionic Function Trees using vampyr multiwavelets.

This module provides the QuatFunction class for representing scalar quaternionic
functions on adaptive multiwavelet bases. A quaternionic function is represented as:
    q(x) = q0(x) + q1(x)*i + q2(x)*j + q3(x)*k
where each component q0, q1, q2, q3 is either a real or complex FunctionTree.

For complex quaternionic functions, each component is a complex_fcn (real + imag parts).
For real quaternionic functions, each component is a single real FunctionTree.
"""
from vampyr import vampyr3d as vp
import numpy as np
from typing import Optional, Union, List, Tuple
from orbital4c import complex_fcn as cf  # Import here to avoid circular imports


def dot_product(q1, q2):
    """
    Computes the dot product of two real quaternionic functions (q1, q2).
    This is an optimized version that minimizes function calls.
    <q1|q2> = sum_i <q1_i|q2_i>
    """
    return (vp.dot(q1.components[0], q2.components[0]) +
            vp.dot(q1.components[1], q2.components[1]) +
            vp.dot(q1.components[2], q2.components[2]) +
            vp.dot(q1.components[3], q2.components[3]))


def add_scaled_to(out, q_a, a, q_b, b, prec):
    """
    Computes out += a*q_a + b*q_b for real quaternionic functions.
    """
    for i in range(4):
        vp.advanced.add(prec, out.components[i], [(1.0, out.components[i]), (a, q_a.components[i]), (b, q_b.components[i])])


def add_to(out, q_in, prec):
    """
    Computes out += q_in for real quaternionic functions.
    """
    for i in range(4):
        vp.advanced.add(prec, out.components[i], [(1.0, out.components[i]), (1.0, q_in.components[i])])


def sub_from(out, q_in, prec):
    """
    Computes out -= q_in for real quaternionic functions.
    """
    for i in range(4):
        vp.advanced.add(prec, out.components[i], [(1.0, out.components[i]), (-1.0, q_in.components[i])])


def add_scaled_complex_to(out, q_a, a, q_b, b, prec):
    """
    Computes out += a*q_a + b*q_b for complex quaternionic functions.
    'a' and 'b' are complex scalars.
    """
    a_re, a_im = a.real, a.imag
    b_re, b_im = b.real, b.imag

    for i in range(4):
        # Real part of output component
        vp.advanced.add(prec, out.components[i].real, [
            (1.0, out.components[i].real),
            (a_re, q_a.components[i].real),
            (-a_im, q_a.components[i].imag),
            (b_re, q_b.components[i].real),
            (-b_im, q_b.components[i].imag)
        ])
        # Imaginary part of output component
        vp.advanced.add(prec, out.components[i].imag, [
            (1.0, out.components[i].imag),
            (a_re, q_a.components[i].imag),
            (a_im, q_a.components[i].real),
            (b_re, q_b.components[i].imag),
            (b_im, q_b.components[i].real)
        ])


class QuatFunction:
    """
    Scalar quaternionic function represented on a multiwavelet basis.

    A quaternion q = q0 + q1*i + q2*j + q3*k where:
    - q0 is the scalar (real) part
    - q1, q2, q3 are the vector (imaginary) parts corresponding to i, j, k

    Each component can be:
    - A real FunctionTree (for real quaternionic functions)
    - A complex_fcn (for complex quaternionic functions)

    The class automatically detects whether it's dealing with real or complex
    components based on the type of the first initialized component.
    """

    mra = None  # Shared MultiResolutionAnalysis object

    # Quaternion multiplication table (i*j=k, j*k=i, k*i=j, i^2=j^2=k^2=-1)
    # Encoded as sign and result index for efficient multiplication
    _Q_MULT = np.array([
        # (sign, result_component) when multiplying row * col
        [(0, 0), (0, 1), (0, 2), (0, 3)],  # 1 * [1,i,j,k]
        [(0, 1), (1, 0), (0, 3), (1, 2)],  # i * [1,i,j,k]
        [(0, 2), (1, 3), (1, 0), (0, 1)],  # j * [1,i,j,k]
        [(0, 3), (0, 2), (1, 1), (1, 0)],  # k * [1,i,j,k]
    ])

    def __init__(self, complex_valued: bool = False):
        """
        Initialize a quaternionic function.

        Args:
            complex_valued: If True, use complex_fcn components.
                           If False, use real FunctionTree components.
        """
        self._complex_valued = complex_valued

        if complex_valued:
            # Import complex_fcn here to avoid circular imports
            print("not implemented complex quaternionic functions yet")
        else:
            self._components = [vp.FunctionTree(self.mra) for _ in range(4)]
            for comp in self._components:
                comp.setZero()

    @property
    def is_complex(self) -> bool:
        """Return True if this is a complex quaternionic function."""
        return self._complex_valued

    @property
    def scalar(self):
        """Return the scalar (real) part q0."""
        return self._components[0]

    @property
    def i_component(self):
        """Return the i component q1."""
        return self._components[1]

    @property
    def j_component(self):
        """Return the j component q2."""
        return self._components[2]

    @property
    def k_component(self):
        """Return the k component q3."""
        return self._components[3]

    @property
    def vector_part(self) -> List:
        """Return the vector (imaginary) part [q1, q2, q3]."""
        return self._components[1:4]

    @property
    def components(self) -> List:
        """Return all components [q0, q1, q2, q3]."""
        return self._components

    def __getitem__(self, idx: Union[int, str]):
        """Access components by index (0-3) or name ('scalar', 'i', 'j', 'k')."""
        if isinstance(idx, str):
            mapping = {'scalar': 0, 's': 0, '0': 0,
                       'i': 1, '1': 1,
                       'j': 2, '2': 2,
                       'k': 3, '3': 3}
            idx = mapping.get(idx.lower(), 0)
        return self._components[idx]

    def __setitem__(self, idx: Union[int, str], value):
        """Set a component by index or name."""
        if isinstance(idx, str):
            mapping = {'scalar': 0, 's': 0, '0': 0,
                       'i': 1, '1': 1,
                       'j': 2, '2': 2,
                       'k': 3, '3': 3}
            idx = mapping.get(idx.lower(), 0)

        if self._complex_valued:
            from orbital4c.complex_fcn import complex_fcn
            if isinstance(value, complex_fcn):
                self._components[idx].copy_fcns(value.real, value.imag)
            else:
                # Assume scalar multiplication
                self._components[idx].real *= value
                self._components[idx].imag *= 0
        else:
            if hasattr(value, 'squaredNorm'):
                vp.advanced.copy_grid(self._components[idx], value)
                vp.advanced.copy_func(self._components[idx], value)
            else:
                self._components[idx] *= value

    def __len__(self):
        return 4

    def __str__(self):
        type_str = "complex" if self._complex_valued else "real"
        return (f"QuatFunction ({type_str}):\n"
                f"  scalar: {self._components[0]}\n"
                f"  i:      {self._components[1]}\n"
                f"  j:      {self._components[2]}\n"
                f"  k:      {self._components[3]}")

    def setZero(self):
        """Set all components to zero."""
        if self._complex_valued:
            for comp in self._components:
                comp.setZero()
        else:
            for comp in self._components:
                comp.setZero()

    def copy(self, other: 'QuatFunction'):
        """Copy all components from another QuatFunction."""
        if other._complex_valued != self._complex_valued:
            raise ValueError("Cannot copy between real and complex quaternionic functions")

        for i in range(4):
            if self._complex_valued:
                self._components[i].copy_fcns(other._components[i].real,
                                               other._components[i].imag)
            else:
                vp.advanced.copy_grid(self._components[i], other._components[i])
                vp.advanced.copy_func(self._components[i], other._components[i])

    # =========================================================================
    # Norm operations
    # =========================================================================

    def squaredNorm(self) -> float:
        """
        Compute the squared L2 norm: ||q||^2 = ||q0||^2 + ||q1||^2 + ||q2||^2 + ||q3||^2
        """
        sn = 0.0
        for i in range(4):
            sn += self._components[i].squaredNorm()

        return sn

    def norm(self) -> float:
        """Compute the L2 norm: ||q|| = sqrt(||q0||^2 + ... + ||q3||^2)"""
        return np.sqrt(self.squaredNorm())

    def normalize(self):
        """Normalize the quaternionic function to unit norm."""
        norm = self.norm()
        if norm > 1e-14:
            self.rescale(1.0 / norm)

    def rescale(self, factor: float):
        """Rescale all components by a real factor."""
        if self._complex_valued:
            for i in range(4):
                self.components[i].real *= factor
                self.components[i].imag *= factor
        else:
            for i in range(4):
                
                self.components[i] *= factor
        return self
    # =========================================================================
    # Arithmetic operations
    # =========================================================================

    def __add__(self, other: 'QuatFunction') -> 'QuatFunction':
        """Add two quaternionic functions component-wise."""
        if not isinstance(other, QuatFunction):
            return NotImplemented

        output = QuatFunction(complex_valued=self._complex_valued)
        for i in range(4):
            output._components[i] = self._components[i] + other._components[i]
        return output

    def __sub__(self, other: 'QuatFunction') -> 'QuatFunction':
        """Subtract two quaternionic functions component-wise."""
        if not isinstance(other, QuatFunction):
            return NotImplemented

        output = QuatFunction(complex_valued=self._complex_valued)
        for i in range(4):
            output._components[i] = self._components[i] - other._components[i]
        return output

    def __neg__(self) -> 'QuatFunction':
        """Negate all components."""
        output = QuatFunction(complex_valued=self._complex_valued)
        for i in range(4):
            output._components[i] = -self._components[i]
        return output

    def __rmul__(self, scalar: Union[float, complex]):
        """Multiply by a scalar from the left."""
        return self.__mul__(scalar)

    def __mul__(self, other: Union['QuatFunction', float, complex]):
        """
        Multiply by a scalar or another quaternionic function.

        For quaternion multiplication q * p:
        (q0 + q1*i + q2*j + q3*k) * (p0 + p1*i + p2*j + p3*k)

        Using optimized formula that minimizes FunctionTree operations.
        """
        output = QuatFunction(complex_valued=self._complex_valued)

        if isinstance(other, (int, float, complex)):
            # Scalar multiplication
            if isinstance(other, complex) and self._complex_valued:
                # Complex scalar times complex quaternionic
                # (a + bi) * (q0 + q1*i + q2*j + q3*k)
                a, b = other.real, other.imag
                for i in range(4):
                    q = self._components[i]
                    # (a + bi) * q = a*q + b*i*q
                    # For complex components, this is complex multiplication
                    output._components[i] = other * q
            else:
                # Real scalar multiplication
                for i in range(4):
                    output._components[i] = other * self._components[i]
            return output

        if isinstance(other, QuatFunction):
            # Quaternion multiplication using optimized formula
            # Result components:
            # r0 = q0*p0 - q1*p1 - q2*p2 - q3*p3
            # r1 = q0*p1 + q1*p0 + q2*p3 - q3*p2
            # r2 = q0*p2 - q1*p3 + q2*p0 + q3*p1
            # r3 = q0*p3 + q1*p2 - q2*p1 + q3*p0

            q = self._components
            p = other._components

            if self._complex_valued:
                from orbital4c.complex_fcn import complex_fcn

                # Helper for complex function multiplication
                def cmul(a, b):
                    """Complex function multiplication: (ar+ai*i)(br+bi*i)"""
                    result = complex_fcn()
                    # (ar+ai*i)(br+bi*i) = (ar*br - ai*bi) + (ar*bi + ai*br)*i
                    result.real = a.real * b.real - a.imag * b.imag
                    result.imag = a.real * b.imag + a.imag * b.real
                    return result

                def csub(a, b):
                    result = complex_fcn()
                    result.real = a.real - b.real
                    result.imag = a.imag - b.imag
                    return result

                def cadd(a, b):
                    result = complex_fcn()
                    result.real = a.real + b.real
                    result.imag = a.imag + b.imag
                    return result

                # Compute products needed (optimized to avoid redundant calculations)
                q0p0 = cmul(q[0], p[0])
                q1p1 = cmul(q[1], p[1])
                q2p2 = cmul(q[2], p[2])
                q3p3 = cmul(q[3], p[3])

                q0p1 = cmul(q[0], p[1])
                q1p0 = cmul(q[1], p[0])
                q2p3 = cmul(q[2], p[3])
                q3p2 = cmul(q[3], p[2])

                q0p2 = cmul(q[0], p[2])
                q1p3 = cmul(q[1], p[3])
                q2p0 = cmul(q[2], p[0])
                q3p1 = cmul(q[3], p[1])

                q0p3 = cmul(q[0], p[3])
                q1p2 = cmul(q[1], p[2])
                q2p1 = cmul(q[2], p[1])
                q3p0 = cmul(q[3], p[0])

                # Assemble result components
                output._components[0] = csub(csub(q0p0, q1p1), cadd(q2p2, q3p3))
                output._components[1] = cadd(cadd(q0p1, q1p0), csub(q2p3, q3p2))
                output._components[2] = cadd(csub(q0p2, q1p3), cadd(q2p0, q3p1))
                output._components[3] = cadd(cadd(q0p3, q1p2), csub(q3p0, q2p1))
            else:
                # Real FunctionTree multiplication
                def rmul(a, b, prec=1e-10):
                    """Real FunctionTree multiplication."""
                    result = vp.FunctionTree(self.mra)
                    result.setZero()
                    if a.squaredNorm() > 0 and b.squaredNorm() > 0:
                        vp.advanced.multiply(prec, result, 1.0, a, b)
                    return result

                def radd(a, b):
                    return a + b

                def rsub(a, b):
                    return a - b

                # Compute products
                q0p0 = rmul(q[0], p[0])
                q1p1 = rmul(q[1], p[1])
                q2p2 = rmul(q[2], p[2])
                q3p3 = rmul(q[3], p[3])

                q0p1 = rmul(q[0], p[1])
                q1p0 = rmul(q[1], p[0])
                q2p3 = rmul(q[2], p[3])
                q3p2 = rmul(q[3], p[2])

                q0p2 = rmul(q[0], p[2])
                q1p3 = rmul(q[1], p[3])
                q2p0 = rmul(q[2], p[0])
                q3p1 = rmul(q[3], p[1])

                q0p3 = rmul(q[0], p[3])
                q1p2 = rmul(q[1], p[2])
                q2p1 = rmul(q[2], p[1])
                q3p0 = rmul(q[3], p[0])

                # Assemble result
                output._components[0] = rsub(rsub(q0p0, q1p1), radd(q2p2, q3p3))
                output._components[1] = radd(radd(q0p1, q1p0), rsub(q2p3, q3p2))
                output._components[2] = radd(rsub(q0p2, q1p3), radd(q2p0, q3p1))
                output._components[3] = radd(radd(q0p3, q1p2), rsub(q3p0, q2p1))

            return output

        return NotImplemented
    
    def real_function_times(self, func: vp.FunctionTree) -> 'QuatFunction':
        output = QuatFunction(complex_valued=self._complex_valued)
        for i in range(4):
            if self._complex_valued:
                output._components[i].real = self._components[i].real * func
                output._components[i].imag = self._components[i].imag * func
            else:
                output._components[i] = self._components[i] * func
        return output

    def i_times(self, from_left = True) -> 'QuatFunction':
        """
        Multiply the quaternionic function by the imaginary unit i.

        If from_left is True, compute i*q = -q1 + q0*i - q3*j + q2*k
        If from_left is False, compute q*i = -q1 + q0*i + q3*j - q2*k
        """
        q = self._components
        if from_left:
            self._components = [-q[1], +q[0], -q[3], +q[2]]
        else:
            self._components = [-q[1], +q[0], +q[3], -q[2]]
        return self
    
    def j_times(self, from_left = True) -> 'QuatFunction':
        """
        Multiply the quaternionic function by the imaginary unit j.

        If from_left is True, compute j*q = -q2 + q3*i + q0*j - q1*k
        If from_left is False, compute q*j = -q2 - q3*i + q0*j + q1*k
        """
        q = self._components
        if from_left:
            self._components = [-q[2], +q[3], +q[0], -q[1]]
        else:
            self._components = [-q[2], -q[3], +q[0], +q[1]]
        return self
    
    def k_times(self, from_left=True) -> 'QuatFunction':
        """
        Multiply the quaternionic function by the imaginary unit k.

        If from_left is True, compute k*q = -q3 - q2*i + q1*j + q0*k
        If from_left is False, compute q*k = -q3 + q2*i - q1*j + q0*k
        """
        q = self._components
        if from_left:
            self._components = [-q[3], -q[2], +q[1], +q[0]]
        else:
            self._components = [-q[3], +q[2], -q[1], +q[0]]
        return self

    

    # =========================================================================
    # Quaternion operations
    # =========================================================================

    def quaternion_conjugate(self) -> 'QuatFunction':
        """
        Compute the quaternion conjugate: q* = q0 - q1*i - q2*j - q3*k
        (negates the vector part, keeps scalar part)
        """
        output = QuatFunction(complex_valued=self._complex_valued)
        output._components[0] = self._components[0]
        output._components[1] = -self._components[1]
        output._components[2] = -self._components[2]
        output._components[3] = -self._components[3]
        return output

    def complex_conjugate(self) -> 'QuatFunction':
        """
        Compute the complex conjugate (for complex quaternionic functions).
        Takes complex conjugate of each component.
        """
        if not self._complex_valued:
            # For real functions, return a copy
            output = QuatFunction(complex_valued=False)
            for i in range(4):
                vp.advanced.copy_grid(output._components[i], self._components[i])
                vp.advanced.copy_func(output._components[i], self._components[i])
            return output

        output = QuatFunction(complex_valued=True)
        for i in range(4):
            output._components[i] = self._components[i].complex_conj()
        return output

    def full_conjugate(self) -> 'QuatFunction':
        """
        Compute the full conjugate: complex conjugate of quaternion conjugate.
        Equivalent to Hermitian conjugate for matrix representations.
        """
        if self._complex_valued:
            output = QuatFunction(complex_valued=True)
            output._components[0] = self._components[0].complex_conj()
            output._components[1] = -self._components[1].complex_conj()
            output._components[2] = -self._components[2].complex_conj()
            output._components[3] = -self._components[3].complex_conj()
            return output
        else:
            return self.quaternion_conjugate()

    def norm_squared_pointwise(self, prec: float) -> 'QuatFunction':
        """
        Compute |q(x)|^2 = q0^2 + q1^2 + q2^2 + q3^2 pointwise.
        Returns a real-valued FunctionTree.
        """
        result = vp.FunctionTree(self.mra)
        result.setZero()

        add_vec = []
        for comp in self._components:
            if self._complex_valued:
                # |c|^2 = Re(c)^2 + Im(c)^2
                temp_r = vp.FunctionTree(self.mra)
                temp_i = vp.FunctionTree(self.mra)
                temp_r.setZero()
                temp_i.setZero()
                if comp.real.squaredNorm() > 0:
                    vp.advanced.multiply(prec, temp_r, 1.0, comp.real, comp.real)
                if comp.imag.squaredNorm() > 0:
                    vp.advanced.multiply(prec, temp_i, 1.0, comp.imag, comp.imag)
                add_vec.append((1.0, temp_r))
                add_vec.append((1.0, temp_i))
            else:
                if comp.squaredNorm() > 0:
                    temp = vp.FunctionTree(self.mra)
                    temp.setZero()
                    vp.advanced.multiply(prec, temp, 1.0, comp, comp)
                    add_vec.append((1.0, temp))

        if add_vec:
            vp.advanced.add(prec, result, add_vec)

        return result

    # =========================================================================
    # Differential operators
    # =========================================================================

    def gradient(self, der: str = 'ABGV') -> List['QuatFunction']:
        """
        Calls vp.gradient() once per component (4 calls total) to get all 3
        directions in one C++ pass. Mirrors complex_fcn.gradient().
        """
        if der == 'ABGV':
            D = vp.ABGVDerivative(self.mra, 0.0, 0.0)
        elif der == 'PH':
            D = vp.PHDerivative(self.mra)
        elif der == 'BS':
            D = vp.BSDerivative(self.mra)
        else:
            raise ValueError(f"Unknown derivative type: {der}")

        # 4 calls to vp.gradient, each returns [d/dx, d/dy, d/dz] at once
        all_grads = []
        for i in range(4):
            if self._components[i].squaredNorm() > 0:
                all_grads.append(vp.gradient(D, self._components[i]))
            else:
                all_grads.append(None)

        result = []
        for d in range(3):
            q_grad = QuatFunction(complex_valued=False)
            for i in range(4):
                if all_grads[i] is not None:
                    q_grad._components[i] = all_grads[i][d]
            result.append(q_grad)
        return result

    def derivative(self, direction: int = 0, der: str = 'ABGV') -> 'QuatFunction':
        """
        Compute the partial derivative with respect to a coordinate direction.

        Args:
            direction: 0=x, 1=y, 2=z
            der: Derivative type ('ABGV', 'PH', 'BS')
        """
        output = QuatFunction(complex_valued=self._complex_valued)

        if self._complex_valued:
            for i in range(4):
                output._components[i] = self._components[i].derivative(direction, der)
        else:
            if der == 'ABGV':
                D = vp.ABGVDerivative(self.mra, 0.0, 0.0)
            elif der == 'PH':
                D = vp.PHDerivative(self.mra)
            elif der == 'BS':
                D = vp.BSDerivative(self.mra)
            else:
                raise ValueError(f"Unknown derivative type: {der}")

            for i in range(4):
                if self._components[i].squaredNorm() > 0:
                    output._components[i] = D(self._components[i], direction)

        return output

    def divergence(self, prec: float, der: str = 'BS') -> 'QuatFunction':
        """
        Compute the divergence of the vector part: d/dx(q1) + d/dy(q2) + d/dz(q3)
        This is useful for quaternionic differential operators.
        """
        grad = self.gradient(der)

        output = QuatFunction(complex_valued=self._complex_valued)
        output._components[0] = (
            grad[0].i_component +
            grad[1].j_component +
            grad[2].k_component
        )
        output._components[0].cropRealImag(prec) if self._complex_valued else output._components[0].crop(prec)

        return output
    
    def sigma_p(self, prec: float, der: str = 'ABGV') -> 'QuatFunction':
        """
        Compute sigma.p = i*(d/dx) + j*(d/dy) + k*(d/dz).
        Assembles output directly via vp.advanced.add — zero intermediates.
        """
        grad = self.gradient(der)
        gx = grad[0]._components  # [gx0, gx1, gx2, gx3]
        gy = grad[1]._components
        gz = grad[2]._components

        output = QuatFunction(complex_valued=False)

        # out[0] = -gx1 - gy2 - gz3
        # out[1] = +gx0 + gy3 - gz2  (note: j*q gives +q3 for i-component)
        # out[2] = -gx3 + gy0 + gz1  (note: k*q gives -q2 for j-component)
        # out[3] = +gx2 - gy1 + gz0
        #
        # Full expansion:
        # i*[g0,g1,g2,g3] = [-g1, g0, -g3,  g2]
        # j*[g0,g1,g2,g3] = [-g2, g3,  g0, -g1]
        # k*[g0,g1,g2,g3] = [-g3,-g2,  g1,  g0]
        # Sum for each output slot:
        contrib = [
            [(-1.0, gx[1]), (-1.0, gy[2]), (-1.0, gz[3])],  # out[0]
            [(+1.0, gx[0]), (+1.0, gy[3]), (-1.0, gz[2])],  # out[1]
            [(-1.0, gx[3]), (+1.0, gy[0]), (+1.0, gz[1])],  # out[2]
            [(+1.0, gx[2]), (-1.0, gy[1]), (+1.0, gz[0])],  # out[3]
        ]
        for idx, terms in enumerate(contrib):
            active = [(c, f) for c, f in terms if f.squaredNorm() > 0]
            if active:
                vp.advanced.add(prec, output._components[idx], active)

        return output
    

    # ========================================================================
    # Integral operators (to be implemented as needed)
    # ========================================================================

    def apply_helmoltz(self: 'QuatFunction', mu: float, prec: float) -> 'QuatFunction':
        """
        Apply the Helmholtz operator (Laplacian + k^2) to a quaternionic function.

        H(q) = (∇^2 + k^2) q
        """
        mra = self.mra
        result = QuatFunction(complex_valued=self.is_complex)
        c = self.light_speed
        if not self.is_complex:
            H = vp.HelmholtzOperator(mra, mu, prec)

        for i in range(4):
            func = self[i]
            if func.squaredNorm() > 0:
                if self.is_complex:
                    result[i] = cf.apply_helmholtz(func,mu,c, prec)
                else:
                    vp.advanced.apply(prec, result[i], H, func)
        

        return result


    # =========================================================================
    # Crop and precision operations
    # =========================================================================

    def crop(self, prec: float, abs: bool = False):
        """Crop all components to the given precision."""
        for comp in self._components:
            if self._complex_valued:
                comp.crop(prec, abs)
            else:
                comp.crop(prec, abs)

    def cropRealImag(self, prec: float):
        """Crop complex components with precision scaling."""
        if self._complex_valued:
            norm = self.norm()
            for comp in self._components:
                comp.crop(prec * norm, True)

    # =========================================================================
    # Inner product
    # =========================================================================

    def dot(self, other: 'QuatFunction') -> float:
        """
        Compute L2 inner product with zero-guards on every vp.dot call.
        Mirrors complex_fcn.dot() pattern exactly.
        """
        result = 0.0
        for i in range(4):
            a = self._components[i]
            b = other._components[i]
            if a.squaredNorm() > 0 and b.squaredNorm() > 0:
                result += vp.dot(a, b)
        return result

    # =========================================================================
    # Save/Load operations
    # =========================================================================

    def save(self, name: str):
        """Save all components to disk."""
        names = ['scalar', 'i', 'j', 'k']
        for i, comp_name in enumerate(names):
            if self._complex_valued:
                self._components[i].save(f"{name}_quat_{comp_name}")
            else:
                self._components[i].saveTree(f"{name}_quat_{comp_name}")

    def load(self, name: str):
        """Load all components from disk."""
        names = ['scalar', 'i', 'j', 'k']
        for i, comp_name in enumerate(names):
            if self._complex_valued:
                self._components[i].read(f"{name}_quat_{comp_name}")
            else:
                self._components[i].loadTree(f"{name}_quat_{comp_name}")

    # =========================================================================
    # Pointwise evaluation
    # =========================================================================

    def __call__(self, position: np.ndarray) -> np.ndarray:
        """
        Evaluate the quaternionic function at a position.

        Returns: numpy array [q0, q1, q2, q3] (real or complex values)
        """
        if self._complex_valued:
            return np.array([comp(position) for comp in self._components])
        else:
            return np.array([comp(position) for comp in self._components])


# =============================================================================
# Utility functions
# =============================================================================

def quat_add(prec: float, output: QuatFunction, terms: List[Tuple[float, QuatFunction]]):
    """
    Add multiple scaled quaternionic functions: output = sum(coeff * quat)

    Args:
        prec: Precision for cropping
        output: Output quaternionic function
        terms: List of (coefficient, QuatFunction) tuples
    """
    for i in range(4):
        if output.is_complex:
            real_terms = []
            imag_terms = []
            for coeff, quat in terms:
                if quat.is_complex:
                    real_terms.append((coeff.real, quat._components[i].real))
                    real_terms.append((-coeff.imag, quat._components[i].imag))
                    imag_terms.append((coeff.real, quat._components[i].imag))
                    imag_terms.append((coeff.imag, quat._components[i].real))
                else:
                    real_terms.append((coeff, quat._components[i]))
                    imag_terms.append((0.0, quat._components[i]))

            vp.advanced.add(prec, output._components[i].real, real_terms)
            vp.advanced.add(prec, output._components[i].imag, imag_terms)
        else:
            add_terms = [(coeff, quat._components[i]) for coeff, quat in terms]
            vp.advanced.add(prec, output._components[i], add_terms)


def quat_multiply(prec: float, lhs: QuatFunction, rhs: QuatFunction) -> QuatFunction:
    """
    Multiply two quaternionic functions with explicit precision control.
    This is a convenience wrapper around lhs * rhs.
    """
    result = lhs * rhs
    result.crop(prec)
    return result


def quat_conjugate_multiply(prec: float, lhs: QuatFunction, rhs: QuatFunction) -> QuatFunction:
    """
    Compute lhs* * rhs (conjugate of lhs times rhs).
    Useful for overlap densities and similar operations.
    """
    lhs_conj = lhs.quaternion_conjugate()
    result = lhs_conj * rhs
    result.crop(prec)
    return result

