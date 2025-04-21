"""
Base classes for finite elements
"""

import numpy as np
from abc import ABC
from typing import List
from dataclasses import dataclass
from enum import Enum, auto
from .components.dof import DegreeOfFreedom
from .components.domain import Domain
from .components.function_space import FunctionSpace
from .components.basis import Basis


class ElementType(Enum):
    """Types of finite elements"""

    TRIANGLE = auto()
    QUADRILATERAL = auto()
    TETRAHEDRON = auto()
    HEXAHEDRON = auto()


@dataclass
class ElementDefinition:
    """Mathematical definition of a finite element according to Ciarlet

    A finite element is a triple (K, P, N) where:
    - K is the domain of the element (piecewise smooth boundary)
    - P is the space of shape functions
    - N is a base for the dual space P'
    """

    domain: Domain
    shape_function_space: FunctionSpace
    nodal_basis: List[DegreeOfFreedom]


class Element(ABC):
    """Base class for all finite elements"""

    def __init__(self, element_definition: ElementDefinition):
        """Initialize the element"""
        self.element_definition = element_definition
        self.shape_functions = self._define_shape_functions()

    @property
    def reference_domain(self) -> Domain:
        """Returns the reference element coordinates"""
        return self.element_definition.domain

    @property
    def shape_function_space(self) -> FunctionSpace:
        """Returns the shape function space"""
        return self.element_definition.shape_function_space

    @property
    def degrees_of_freedom(self) -> List[DegreeOfFreedom]:
        """Returns the degrees of freedom"""
        return self.element_definition.nodal_basis

    def _get_vandermonde_matrix(self) -> np.ndarray:
        """Returns the Vandermonde matrix"""
        canonical_basis = self.shape_function_space.basis
        dofs = self.degrees_of_freedom
        return np.array(
            [[dof(basis_func) for dof in dofs] for basis_func in canonical_basis]
        )

    def _define_shape_functions(self) -> Basis:
        """Defines the shape functions for the element"""
        vandermonde = self._get_vandermonde_matrix()
        inv_mat = np.linalg.inv(vandermonde)
        canonical_basis = self.shape_function_space.basis
        return inv_mat @ canonical_basis