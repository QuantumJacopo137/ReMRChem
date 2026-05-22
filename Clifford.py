import re
from vampyr import vampyr3d as vp
import numpy as np
from typing import Optional, Union, List, Tuple




class ClifFunc:
    """
    A spinor operator is a element of the even grade subalgebra of Cl(1,3),
    which is scalar + bivector + pseudovector.
    g0^2 = 1, g1^2 = g2^2 = g3^2 = -1, and all basis elements anticommute with each other.
    {g_mu, g_nu} = 2 * eta_{mu,nu} where eta is the Minkowski metric diag(1,-1,-1,-1).

    A spinor function is thus a 8D object with component of indices:
    ['s', '01', '02', '03', '23', '31', '12', '0123'] 
    Which correspond to the scalar, bivector, and pseudoscalar parts of the spinor operator.
    The index label represent the basis element: 's' = scalar, '01' = g0*g1, '02' = g0*g2, '03' = g0*g3, '23' = g2*g3, '31' = g3*g1, '12' = g1*g2, '0123' = g0*g1*g2*g3.

    """

    """
    Given the 4-spor Phi = (phi_1, phi_2, phi_3, phi_4)^T, where each phi_i is a complex scalar function, 
    we can construct the corresponding spinor operator function PSI as:
        PSI = s + k1 * g01 + k2 * g02 + k3 * g03  + j1 * g23 + j2 * g31 + j3 * g12 + p * g0123
    In this case the action of the immaginary unit on 4-spinor is replaced in the even subalgebra by the right multiplication by g21, that is -g12 so that:
        i * Phi <-> PSI * g12
    Thus we can identify the components of PSI in terms of the components of Phi.
    The right multilpication by g21 will exchange the following components of PSI:
        -------------------
        | +1     -> -g12
        | +g12   -> +1
        -------------------
        | +g01   -> +g02
        | +g02   -> -g01
        -------------------
        | +g03   -> -g0123
        | +g0123 -> +g03
        -------------------
        | +g23   -> +g31
        | +g31   -> -g23
        -------------------
    The term which preserve the sgn after multiplication by i, that is Psi * g21 
    Are clearly the real components, while the terms which change the sgn are the immaginary components. Thus we can identify:
        s = Re(phi_1)
        k1 = Im(phi_4)
        k2 = Im(phi_3)
        k3 = Im(phi_2)
        j1 = Re(phi_4)
        j2 = Re(phi_3)
        j3 = Re(phi_2)
        p = Im(phi_1)

    """

    mra = None  # Shared MultiResolutionAnalysis object

    def __init__(self):
        """
        Initialize a spinor operator function.

        """

        self._components = [vp.FunctionTree(self.mra) for _ in range(8)]

        for comp in self._components:
            comp.setZero()



    @property
    def scalar(self):
        """Return the scalar (real) part q0."""
        return self._components[0]
    
    @property
    def pseudoscalar(self):
        """Return the pseudoscalar part q0123."""
        return self._components[7]

    @property
    def boost_rotors(self):
        """Return the boost rotor components [g01, g02, g03]."""
        return [self._components[1], self._components[2], self._components[3]]

    @property
    def rotation_rotors(self):
        """Return the rotation rotor components [g23, g31, g12]."""
        return [self._components[4], self._components[5], self._components[6]]


        return self._components[1:4]

    @property
    def components(self) -> List:
        """Return all components [q0, q1, q2, q3]."""
        return self._components

    def __getitem__(self, idx: Union[int, str]):
        """Access components by index (0-3) or name ('scalar', 'i', 'j', 'k')."""
        if isinstance(idx, str):
            mapping = {'scalar': 0, 's': 0, '': 0,
                        '01': 1, 'g01': 1,
                        '02': 2, 'g02': 2,
                        '03': 3, 'g03': 3,
                        '23': 4, 'g23': 4,
                        '31': 5, 'g31': 5,
                        '12': 6, 'g12': 6,
                        'pseudoscalar': 7,'p': 7, '0123': 7, 'g0123': 7}
            idx = mapping.get(idx.lower(), 0)
        return self._components[idx]

    def __setitem__(self, idx: Union[int, str], value):
        """Set a component by index or name."""
        if isinstance(idx, str):
            mapping = {'scalar': 0, 's': 0, '': 0,
                        '01': 1, 'g01': 1,
                        '02': 2, 'g02': 2,
                        '03': 3, 'g03': 3,
                        '23': 4, 'g23': 4,
                        '31': 5, 'g31': 5,
                        '12': 6, 'g12': 6,
                        'pseudoscalar': 7,'p': 7, '0123': 7, 'g0123': 7}
            idx = mapping.get(idx.lower(), 0)
        if hasattr(value, 'squaredNorm'):
            vp.advanced.copy_grid(self._components[idx], value)
            vp.advanced.copy_func(self._components[idx], value)
        else:
            self._components[idx] *= value

    def __len__(self):
        return 8

    # Lookup tables for basis multiplication: B_i * B_j = sign[i,j] * B[result[i,j]]
    # Basis: 0=s, 1=g01, 2=g02, 3=g03, 4=g23, 5=g31, 6=g12, 7=g0123
    #
    # Metric: g0^2 = +1, g1^2 = g2^2 = g3^2 = -1
    # Anticommutation: g_mu * g_nu = -g_nu * g_mu for mu != nu
    #
    # Derivation examples:
    #   g01 * g01 = g0*g1*g0*g1 = -g0*g0*g1*g1 = -(+1)*(-1) = +1 ✓
    #   g01 * g02 = g0*g1*g0*g2 = -g0*g0*g1*g2 = -(+1)*g12 = -g12 → idx=6, sign=-1
    #   g01 * g23 = g0*g1*g2*g3 = g0123 → idx=7, sign=+1
    #   g01 * g0123 = g0*g1*g0*g1*g2*g3 = -g0*g0*g1*g1*g2*g3 = -(+1)*(-1)*g23 = +g23 → idx=4, sign=+1
    #   g12 * g12 = g1*g2*g1*g2 = -g1*g1*g2*g2 = -(-1)*(-1) = -1 ✓
    #   g0123 * g0123 = g0*g1*g2*g3*g0*g1*g2*g3 = -g0*g0*g1*g2*g3*g1*g2*g3 = ... = -1 ✓

    _MULT_RESULT = [
        # j=0   1     2     3     4     5     6     7
        [0,    1,    2,    3,    4,    5,    6,    7   ],  # i=0 (s)
        [1,    0,    6,    5,    7,    3,    2,    4   ],  # i=1 (g01)
        [2,    6,    0,    4,    3,    7,    1,    5   ],  # i=2 (g02)
        [3,    5,    4,    0,    2,    1,    7,    6   ],  # i=3 (g03)
        [4,    7,    3,    2,    0,    6,    5,    1   ],  # i=4 (g23)
        [5,    3,    7,    1,    6,    0,    4,    2   ],  # i=5 (g31)
        [6,    2,    1,    7,    5,    4,    0,    3   ],  # i=6 (g12)
        [7,    4,    5,    6,    1,    2,    3,    0   ],  # i=7 (g0123)
    ]

    _MULT_SIGN = [
        # j=0   1     2     3     4     5     6     7
        [1,    1,    1,    1,    1,    1,    1,    1   ],  # i=0 (s) - identity
        [1,    1,   -1,   +1,    1,    1,   -1,    1   ],  # i=1 (g01)
        [1,   +1,    1,   -1,   -1,   +1,    1,    1   ],  # i=2 (g02)
        [1,   -1,    1,    1,   +1,   -1,    1,    1   ],  # i=3 (g03)
        [1,    1,   +1,   -1,   -1,   +1,   -1,   -1   ],  # i=4 (g23)
        [1,   -1,    1,    1,   -1,   -1,    1,   -1   ],  # i=5 (g31)
        [1,    1,   -1,    1,    1,   -1,   -1,   -1   ],  # i=6 (g12)
        [1,    1,    1,    1,   -1,   -1,   -1,   -1   ],  # i=7 (g0123)
    ]

    @staticmethod
    def basis_product(idx1: int, idx2: int) -> Tuple[int, int]:
        """
        Compute B_idx1 * B_idx2 = sign * B_result_idx

        Args:
            idx1: Index of left basis element (0-7)
            idx2: Index of right basis element (0-7)

        Returns:
            Tuple of (sign, result_idx) where sign is +1 or -1

        Basis mapping:
            0: 's'     (scalar)
            1: 'g01'   (bivector)
            2: 'g02'   (bivector)
            3: 'g03'   (bivector)
            4: 'g23'   (bivector)
            5: 'g31'   (bivector)
            6: 'g12'   (bivector)
            7: 'g0123' (pseudoscalar)
        """
        result_idx = ClifFunc._MULT_RESULT[idx1][idx2]
        sign = ClifFunc._MULT_SIGN[idx1][idx2]

        return (sign, result_idx)

    def __str__(self):
        return (f"\n-------------------------------------\n"
                f"Spinor operator orbital:\n"
                f"  scalar: {self._components[0]}\n"
                f"  g01:    {self._components[1]}\n"
                f"  g02:    {self._components[2]}\n"
                f"  g03:    {self._components[3]}\n"
                f"  g23:    {self._components[4]}\n"
                f"  g31:    {self._components[5]}\n"
                f"  g12:    {self._components[6]}\n"
                f"  g0123:  {self._components[7]}"
                f"\n-------------------------------------\n")   

    def setZero(self):
        """Set all components to zero."""
        for comp in self._components:
            comp.setZero()

    def copy(self, other: 'ClifFunc'):
        """Copy all components from another ClifFunc."""
        for i in range(8):
            vp.advanced.copy_grid(self._components[i], other._components[i])
            vp.advanced.copy_func(self._components[i], other._components[i])

    # =========================================================================
    # Norm operations
    # =========================================================================

    def squaredNorm(self) -> float:
        """
        Compute the squared L2 norm: ||PSI||^2 = ||s||^2 + ||g01||^2 + ||g02||^2 + ||g03||^2 + ||g23||^2 + ||g31||^2 + ||g12||^2 + ||g0123||^2
        """
        sn = 0.0
        for i in range(8):
            sn += self._components[i].squaredNorm()

        return sn

    def norm(self) -> float:
        """Compute the L2 norm: ||PSI|| = sqrt(||s||^2 + ... + ||g0123||^2)"""
        return np.sqrt(self.squaredNorm())

    def normalize(self):
        """Normalize the SOF to unit norm."""
        norm = self.norm()
        if norm > 1e-14:
            self.rescale(1.0 / norm)

    def rescale(self, factor: float):
        """Rescale all components by a real factor."""
        for i in range(8):
            self.components[i] *= factor
        return self
    
    def print_all_Sqnorms(self):
        """Print the squared norms of all components."""
        names = ['+1 ', 'g01', 'g02', 'g03', 'g23', 'g31', 'g12', 'g0123']
        for i in range(8):
            print(f"{names[i]} squared norm: {self._components[i].squaredNorm()}")
        print()
    # =========================================================================
    # Arithmetic operations
    # =========================================================================

    def __add__(self, other: 'ClifFunc') -> 'ClifFunc':
        """Add two SOF component-wise."""
        if not isinstance(other, ClifFunc):
            return NotImplemented

        output = ClifFunc()
        for i in range(8):
            #vp.advanced.add(-1.0, output._components[i], [(1.0, self._components[i]), (1.0, other._components[i])])
            output._components[i] = self._components[i] + other._components[i]

        return output

    def __sub__(self, other: 'ClifFunc') -> 'ClifFunc':
        """Subtract two SOFs component-wise."""
        if not isinstance(other, ClifFunc):
            return NotImplemented

        output = ClifFunc()
        for i in range(8):
            output._components[i] = self._components[i] - other._components[i]
        return output

    def __neg__(self) -> 'ClifFunc':
        """Negate all components."""
        output = ClifFunc()
        for i in range(8):
            output._components[i] = -self._components[i]
        return output

    def __rmul__(self, scalar: float):
        """Multiply by a scalar from the left."""
        return self.__mul__(scalar)

    def __mul__(self, other: Union['ClifFunc', float]):
        """
        Multiply by a scalar or another quaternionic function.

        For SOF multiplication PSI * A:
        (s + k1 * g01 + k2 * g02 + k3 * g03  + j1 * g23 + j2 * g31 + j3 * g12 + p * g0123) * A
        

        Using optimized formula that minimizes FunctionTree operations.
        """
        output = ClifFunc()

        if isinstance(other, (int, float)):
            # Scalar multiplication
            # Real scalar multiplication
            for i in range(8):
                output._components[i] = other * self._components[i]
            return output

        if isinstance(other, ClifFunc):
            # SOF multiplication
            
            for i in range(8):
                # Compute the product of self with the i-th basis element of other
                for j in range(8):
                    sign, idx = self.basis_product(i, j)
                    output._components[idx] += sign * self._components[i] * other._components[j]
                

            return output

        return NotImplemented
    
    

    
    
    def g01_PSI_g12(self) -> 'ClifFunc':
        """
        Multiply SOF by g01 on the left and g12 on the right: 
        g01 * PSI * g12 = - s * g02 + k1 * g12 + k2 + k3 * g23 - j1 * g03 + j2 g0123 - j3 * g01 - p * g31 
        So that 
        g01 * PSI * g12 = k3 - j3 * g01 - s * g02 - j1 * g03 + k3 * g23 - p * g31 + k1 * g12 + j2 * g0123
        
        inp|  out  | sign_out
        i=0:  2,        -1
        i=1:  6,        1
        i=2:  0,        1
        i=3:  4,        1
        i=4:  3,        -1
        i=5:  7,        1
        i=6:  1,        -1
        i=7:  5,        -1
        """
        q = self._components
        self._components = [-q[2], q[6], q[0], q[4], -q[3], q[7], -q[1], -q[5]]
         
        
    
    def g02_PSI_g12(self) -> 'ClifFunc':
        """
        Multiply SOF by g02 on the left and g12 on the right:
        g02 * PSI * g12 = s * g01 - k1 * 1 + k2 * g12 + k3 * g31 - j1 * g0123 - j2 * g03 - j3 * g02 + p * g23
        So that
        g02 * PSI * g12 = - k1 + s * g01 - j3 * g02 - j2 * g03 + k3 * g31 + p * g23 + k2 * g12 - j1 * g0123
        
        inp|  out  | sign_out
        i=0:  1,        1
        i=1:  0,        -1
        i=2:  6,        1
        i=3:  5,        1
        i=4:  7,        -1
        i=5:  3,        -1
        i=6:  2,        -1
        i=7:  4,        1
        """
        q = self._components
        self._components = [q[1], -q[0], q[6], q[5], -q[7], -q[3], -q[2], q[4]]
    
    def g03_PSI_g12(self) -> 'ClifFunc':
        """
        Multiply SOF by g03 on the left and g12 on the right:
        g03 * PSI * g12 = s * g0123 - k1 * g23 - k2 * g31 + k3 * g12 + j1 * g01 + j2 * g02 - j3 * g03 - p * 1
        So that
        g03 * PSI * g12 = - p * 1 + j1 * g01 + j2 * g02 - j3 * g03 - k1 * g23 - k2 * g31 + k3 * g12 + s * g0123
        
        inp|  out  | sign_out
        i=0:  7,        1
        i=1:  4,        -1
        i=2:  5,        -1
        i=3:  6,        1
        i=4:  1,        1
        i=5:  2,        1
        i=6:  3,        -1
        i=7:  0,        -1
        """

        q = self._components
        self._components = [q[7], -q[4], -q[5], q[6], q[1], q[2], -q[3], -q[0]]

    

    def PSI_g12(self) -> 'ClifFunc':
        """
        Multiply the SOF by g12 on the right: 
        PSI * g12 = - s * g12 - k1 * g02 + k2 * g01 + k3 * g0123 - j1 * g31 + j2 * g23 - j3 * 1 - p * g03
        So that
        PSI * g12 = - j3 * 1 + k2 * g01 - k1 * g02 - p * g03 + j2 * g23 - j1 * g31 - s * g12 + k3 * g0123
        """
        out = ClifFunc()
        q = self._components

        # Apply the transformation
        #                    's'    '01'   '02'  '03'  '23'   '31'    '12'    '0123'
        #out._components = [-q[6], -q[2], -q[1], -q[7], -q[5], -q[4], -q[0], +q[3]]
        #out._components = [-q[6], -q[2], +q[1], +q[7], -q[5], +q[4], +q[0], +q[3]]

        for i in range(8):
            sign, idx = self.basis_product(i, 6)  # B_i * g12
            out._components[idx] = sign * q[i]
            print(f"i={i}, out_idx={idx}, sign={sign}")
        return out
    
    def inline_PSI_g12(self):
        """
        In-place multiplication of the SOF by g12 on the right: 
        i=0, out_idx=6, sign=1
        i=1, out_idx=2, sign=-1
        i=2, out_idx=1, sign=1
        i=3, out_idx=7, sign=1
        i=4, out_idx=5, sign=-1
        i=5, out_idx=4, sign=1
        i=6, out_idx=0, sign=-1
        i=7, out_idx=3, sign=-1

        """
        q = self._components
        
        self._components = [q[6], -q[2], q[1], q[7], -q[5], q[4], -q[0], -q[3]]

    def PSI_times_basis(self, idx_A: int, from_the_left = False) -> 'ClifFunc':
        """
        Multiply the SOF from the right by a basis element: PSI * gA
        A = [1,7]
        """
        out = ClifFunc()
        q = self._components
        for i in range(8):
            if from_the_left:
                sign_PsiA, idx_PsiA = ClifFunc.basis_product(idx_A, i)  # gA * B_i
            else:
                sign_PsiA, idx_PsiA = ClifFunc.basis_product(i, idx_A)  # B_i * gA

            #out._components[i] = sign_PsiA * q[idx_PsiA]
            out._components[idx_PsiA] = sign_PsiA * q[i]


        return out
    

    def apply_potential(self: 'ClifFunc', factor: float, potential: vp.FunctionTree, func: 'ClifFunc', prec: float) -> 'ClifFunc':
            for i in range(8):
                if func._components[i].squaredNorm() > 0:
                    self._components[i] = factor * potential * func._components[i] 
            self.crop(prec)

    # =========================================================================
    # Involutions and operations
    # =========================================================================

    def reverse(self) -> 'ClifFunc':
        """
        Takes the reverse of the spinor operator function, which negates only the bivector components.
        """
        output = ClifFunc()
        output._components[0] = self._components[0]  # scalar part unchanged
        output._components[1] = -self._components[1]  
        output._components[2] = -self._components[2]
        output._components[3] = -self._components[3]
        output._components[4] = -self._components[4]   
        output._components[5] = -self._components[5]   
        output._components[6] = -self._components[6]   
        output._components[7] = self._components[7]   # pseudoscalar part unchanged
        return output
    
    def inline_reverse(self):
        """
        In-place reverse of the spinor operator function, which negates only the bivector components.
        """
        self._components[1] = -self._components[1]  
        self._components[2] = -self._components[2]
        self._components[3] = -self._components[3]
        self._components[4] = -self._components[4]   
        self._components[5] = -self._components[5]   
        self._components[6] = -self._components[6]
    

    def g0_PSI_g0(self) -> 'ClifFunc':
        """
        Multiply the SOF from both sides by g0: g0 * PSI * g0 = s - k1 * g01 - k2 * g02 - k3 * g03  + j1 * g23 + j2 * g31 + j3 * g12 - p * g0123
        """
        out = ClifFunc()
        q = self._components
        out._components = [q[0], -q[1], -q[2], -q[3], q[4], q[5], q[6], -q[7]]
        return out
    
    def inline_g0_PSI_g0(self):
        """
        In-place multiplication of the SOF from both sides by g0: g0 * PSI * g0 = s - k1 * g01 - k2 * g02 - k3 * g03  + j1 * g23 + j2 * g31 + j3 * g12 - p * g0123
        """
        q = self._components
        self._components = [q[0], -q[1], -q[2], -q[3], q[4], q[5], q[6], -q[7]]

    def GA_PSI_GB(self, idx_A, idx_B) -> 'ClifFunc':
        """
        It is going to apply one of the basis elements from each side
        A = [1,7]
        B = [0,7]
        """
        out = ClifFunc()
        q = self._components
        for i in range(8):
            sign_APsi, idx_APsi = ClifFunc.basis_product(idx_A, i)  # g02 * B_i
            sign_APsiB, idx_APsiB = ClifFunc.basis_product(idx_APsi, idx_B)  # (g02 * B_i) * g

            sign_tot = sign_APsi * sign_APsiB
            #out._components[i] = sign_tot * q[idx_APsiB]
            out._components[idx_APsiB] = sign_tot * q[i]
        
        return out
    

    # =========================================================================
    # Differential operators
    # =========================================================================

    def gradient(self, der: str = 'ABGV') -> List['ClifFunc']:
        """
        Compute the gradient of each component.

        Returns a list of 3 QuatFunctions representing [d/dx, d/dy, d/dz].
        """
    
        if der == 'ABGV':
            D = vp.ABGVDerivative(self.mra, 0.0, 0.0)
        elif der == 'PH':
            D = vp.PHDerivative(self.mra)
        elif der == 'BS':
            D = vp.BSDerivative(self.mra)
        else:
            raise ValueError(f"Unknown derivative type: {der}")

        result = []
        for d in range(3):
            q_grad = ClifFunc()
            for i in range(8):
                if self._components[i].squaredNorm() > 0:
                    q_grad._components[i] = D(self._components[i], d)
            result.append(q_grad)
        return result

    def derivative(self, direction: int = 0, der: str = 'ABGV') -> 'ClifFunc':
        """
        Compute the partial derivative with respect to a coordinate direction.

        Args:
            direction: 0=x, 1=y, 2=z
            der: Derivative type ('ABGV', 'PH', 'BS')
        """
        output = ClifFunc()

    
        if der == 'ABGV':
            D = vp.ABGVDerivative(self.mra, 0.0, 0.0)
        elif der == 'PH':
            D = vp.PHDerivative(self.mra)
        elif der == 'BS':
            D = vp.BSDerivative(self.mra)
        else:
            raise ValueError(f"Unknown derivative type: {der}")

        for i in range(8):
            if self._components[i].squaredNorm() > 0:
                output._components[i] = D(self._components[i], direction)

        return output

    
    
    def alpha_p(self, prec: float, der: str = 'ABGV') -> 'ClifFunc':
        """
        Optimized (alpha . p) PSI.
        Fuses all gradient, sandwich, and summation operations.
        """
        grad_x, grad_y, grad_z = self.gradient(der)
        
        # Use the fast, pre-computed sandwich product methods
        # This avoids creating intermediate ClifFunc objects entirely.
        grad_x.g01_PSI_g12()
        grad_y.g02_PSI_g12()
        grad_z.g03_PSI_g12()
        
        # Create output object and sum the terms
        output = ClifFunc()
        # Fuse the summation of all 3 terms into a single operation
        for i in range(8):
            # This vp.advanced.add call is efficient.
            vp.advanced.add(prec, output._components[i], [
                (1.0, grad_x._components[i]),
                (1.0, grad_y._components[i]),
                (1.0, grad_z._components[i])
            ])
            
        output.crop(prec)
        return output

    def classicT(self) -> float:
        """
        Compute the classical kinetic energy operator T = -1/2 * ∇^2 PSI
        """
        
        grad = self.gradient()
        result = 0.5 * (grad[0].squaredNorm() + grad[1].squaredNorm() + grad[2].squaredNorm())
            
        return result

    # ========================================================================
    # Integral operators (to be implemented as needed)
    # ========================================================================

    def apply_helmoltz(self: 'ClifFunc', mu: float, prec: float) -> 'ClifFunc':
        """
        Apply the Helmholtz operator (Laplacian + k^2) to a quaternionic function.

        H(q) = (∇^2 + k^2) q
        """
        mra = self.mra
        result = ClifFunc()
        H = vp.HelmholtzOperator(mra, mu, prec)

        for i in range(8):
            if self._components[i].squaredNorm() > 0:
                vp.advanced.apply(prec, result[i], H, self[i])
            #result[i] *= -1.0/(2*np.pi)

        return -1.0/(4*np.pi) * result


    # =========================================================================
    # Crop and precision operations
    # =========================================================================

    def crop(self, prec: float, abs: bool = False):
        """Crop all components to the given precision."""
        for comp in self._components: 
            comp.crop(prec, abs)


    # =========================================================================
    # Inner product
    # =========================================================================


    

    def dot(self, other: 'ClifFunc') -> complex:
        """
        Optimized inner product <self|other>.
        <Psi|Phi> = integral[ (g0*Tilde(Psi)*g0 * Phi)_s + i * (g0*Tilde(Psi)*g0 * Phi * g12)_s ]
        This is calculated by summing the component-wise dot products with the appropriate signs,
        avoiding the creation of intermediate ClifFunc objects.
        """
        # Real part: integral of scalar part of (psi_dagger * other)
        # (s*s' - j1*j1' - j2*j2' - j3*j3' + k1*k1' + k2*k2' + k3*k3' - p*p')
        

        #other_g21 = other.PSI_g12()
        

    

        result_real = 0.0
        result_imag = 0.0

        
        # no need for tilde as i am taking already the scalar part of the product, which is the same for Psi and Tilde(Psi)
        for i in range(8):
            RPC  = vp.dot(self._components[i], other._components[i])
            sign, idx = self.basis_product(i, 6)  # B_i * g12
            IPC  = sign * 1j * vp.dot(self._components[i], other._components[idx])
            #print (f"Component {i}: RPC = {RPC}, IPC = {IPC}")
            result_real += RPC
            result_imag += IPC
        
        return result_real + result_imag
        
    # =========================================================================
    # Save/Load operations
    # =========================================================================

    def save(self, name: str):
        """Save all components to disk."""
        names = ['scalar', '01', '02', '03', '23', '31', '12', '0123']
        for i, comp_name in enumerate(names):
            self._components[i].saveTree(f"{name}_quat_{comp_name}")

    def load(self, name: str):
        """Load all components from disk."""
        names = ['scalar', '01', '02', '03', '23', '31', '12', '0123']
        for i, comp_name in enumerate(names):
            self._components[i].loadTree(f"{name}_quat_{comp_name}")

    # =========================================================================
    # Pointwise evaluation
    # =========================================================================

    def __call__(self, position: np.ndarray) -> np.ndarray:
        """
        Evaluate the quaternionic function at a position.

        Returns: numpy array [scalar, g01, g02, g03, g23, g31, g12, g0123] evaluated at the given position.
        """
        return np.array([comp(position) for comp in self._components])


# =============================================================================
# Utility functions
# =============================================================================

def Clif_add(prec: float, output: ClifFunc, terms: List[Tuple[float, ClifFunc]]):
    """
    Add multiple scaled quaternionic functions: output = sum(coeff * quat)

    Args:
        prec: Precision for cropping
        output: Output quaternionic function
        terms: List of (coefficient, QuatFunction) tuples
    """
    for i in range(8):
        add_terms = [(coeff, quat._components[i]) for coeff, quat in terms]
        vp.advanced.add(prec, output._components[i], add_terms)


def Clif_multiply(prec: float, lhs: ClifFunc, rhs: ClifFunc) -> ClifFunc:
    """
    Multiply two quaternionic functions with explicit precision control.
    This is a convenience wrapper around lhs * rhs.
    """
    result = lhs * rhs
    result.crop(prec)
    return result


def Clif_reverse_multiply(prec: float, lhs: ClifFunc, rhs: ClifFunc) -> ClifFunc:
    """
    Compute Tilde(lhs) * rhs (conjugate of lhs times rhs).
    Useful for overlap densities and similar operations.
    """
    lhs_conj = lhs.reverse()
    result = lhs_conj * rhs
    result.crop(prec)
    return result



def apply_Dirac_Hestenes_hamiltonian(light_speed: float, prec: float, orbital: ClifFunc, shift = 0.0) -> ClifFunc:
    """
    Apply the Dirac Hamiltonian in Hestenes form to a spinor operator function.

    H(PSI) = c * (d/dx * g01 + d/dy * g02 + d/dz * g03) PSI * g12 + m * c^2 * g0 PSI g0 
    Args:
        prec: Precision for cropping intermediate results
        orbital: Input spinor operator function representing the wavefunction
    """
    

    
    # Compute the mass term: (beta * m) PSI
    output = light_speed * orbital.alpha_p(prec)
    if shift != 0.0:
        output += shift * orbital
    orbital.inline_g0_PSI_g0()
    output +=  light_speed**2 * orbital
    orbital.inline_g0_PSI_g0()

    output.crop(prec)
    
    return output

def Clif_starting_guess_NR(NR_sol: vp.FunctionTree, prec: float, light_speed: float) -> ClifFunc:
    """
    Create a starting guess for the Dirac equation from a non-relativistic solution.

    Args:
        NR_sol: Non-relativistic wavefunction as a FunctionTree
        prec: Precision for initializing the small components
    """
    orbital = ClifFunc()
    Ne_part = ClifFunc()
    orbital[0] = NR_sol
    Ne_part[0] = 1.0/(2*light_speed) * NR_sol 
   

    


    orbital = orbital +   Ne_part.alpha_p(prec) 

    orbital.normalize()
    orbital.crop(prec)

    return orbital


