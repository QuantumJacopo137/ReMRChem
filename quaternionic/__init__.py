"""
Quaternionic module for multiwavelet-based quaternionic function representations.

This module provides classes for representing and manipulating quaternionic
functions and orbitals using the vampyr multiwavelet library.

Classes:
    QuatFunction: Scalar quaternionic function on a multiwavelet basis
    CompQuatOrbital: 4-component spinor as complex quaternionic function
    QuatOrbital: 2-component spinor with real quaternionic function components
"""

from .quat_function import QuatFunction, quat_add, quat_multiply, quat_conjugate_multiply
from .quat_orbital import (
    QuatOrbital,
    ComponentView,
    apply_dirac_hamiltonian,
    apply_potential,
    apply_helmholtz,
    add_vector,
    apply_pauli_hamiltonian,
)

__all__ = [
    # Core function class
    'QuatFunction',
    'quat_add',
    'quat_multiply',
    'quat_conjugate_multiply',

    # Orbital classes
    'CompQuatOrbital',  # 4-component complex quaternionic spinor
    'QuatOrbital',      # 2-component real quaternionic spinor
    'ComponentView',    # View into quaternionic component as complex function

    # Utility functions
    'apply_dirac_hamiltonian',
    'apply_potential',
    'apply_helmholtz',
    'add_vector',
    'apply_pauli_hamiltonian',
]
