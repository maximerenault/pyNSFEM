"""
Base classes for finite elements
"""

import numpy as np
from abc import ABC, abstractmethod
from typing import Tuple, List, Callable, Optional, Union
from dataclasses import dataclass
from enum import Enum, auto
from .components.dof import DegreeOfFreedom
from .components.domain import Domain
from .components.function_space import FunctionSpace
from .components.basis import BasisFunction
from ..core.mesh import Mesh
from ..utils.poly_basis import tri_poly_basis_auto


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

    def __init__(self):
        """Initialize the element"""
        self.element_definition = None
        self.degree = None  # Polynomial degree of shape functions
        self.num_dof = None  # Number of degrees of freedom per element
        self.dim = None  # Dimension of the element
        self.shape_functions = self._define_shape_functions()

    @property
    def reference_element(self) -> Domain:
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
        """Gets the columns of the Vandermonde matrix"""
        canonical_basis = self.element_definition.shape_function_space.basis
        dofs = self.element_definition.nodal_basis
        return np.array(
            [[dof(basis_func) for basis_func in canonical_basis] for dof in dofs]
        )

    def _define_shape_functions(self) -> List[BasisFunction]:
        """Defines the shape functions for the element"""
        vandermonde = self._get_vandermonde_matrix()
        inv_mat = np.linalg.inv(vandermonde)
        canonical_basis = self.element_definition.shape_function_space.basis
        return inv_mat @ canonical_basis


class TriElement(Element):
    """Base class for triangular elements"""

    def __init__(self):
        super().__init__()
        self.element_type = ElementType.TRIANGLE
        self.dim = 2
        # Reference triangle coordinates (unit triangle)
        self.element_definition = ElementDefinition(
            domain=np.array([[0, 0], [1, 0], [0, 1]]),
            shape_functions=[],
            degrees_of_freedom=[],
        )

    def _create_vandermonde_matrix(
        self,
    ) -> Tuple[Callable, Callable, Callable, np.ndarray]:
        """Creates the Vandermonde matrix for the element"""
        func, grad, hess = tri_poly_basis_auto(degree=self.degree)
        mat = np.array(self._get_vandermonde_columns(func, grad, hess)).T
        return func, grad, hess, mat

    def _get_vandermonde_columns(
        self, func: Callable, grad: Callable, hess: Callable
    ) -> List[np.ndarray]:
        """Gets the columns of the Vandermonde matrix"""
        pass

    def compute_jacobian(self, nodePts: np.ndarray) -> np.ndarray:
        """Computes the Jacobian matrix for triangular elements"""
        return nodePts[1:, :] - nodePts[0]

    def compute_area(self, nodePts: np.ndarray) -> float:
        """Computes the area of a triangular element"""
        J = self.compute_jacobian(nodePts)
        return 0.5 * np.linalg.det(J)

    def transform_to_physical(
        self, points: np.ndarray, nodePts: np.ndarray
    ) -> np.ndarray:
        """Transforms points from natural coordinates to physical coordinates"""
        return np.dot(points, nodePts)


class QuadElement(Element):
    """Base class for quadrilateral elements"""

    def __init__(self):
        super().__init__()
        self.element_type = ElementType.QUADRILATERAL
        self.dim = 2
        # Reference quadrilateral coordinates (unit square)
        self.element_definition = ElementDefinition(
            domain=np.array([[-1, -1], [1, -1], [1, 1], [-1, 1]]),
            shape_functions=[],
            degrees_of_freedom=[],
        )

    def compute_jacobian(self, nodePts: np.ndarray) -> np.ndarray:
        """Computes the Jacobian matrix for quadrilateral elements"""
        return np.array(
            [
                [nodePts[1, 0] - nodePts[0, 0], nodePts[2, 0] - nodePts[0, 0]],
                [nodePts[1, 1] - nodePts[0, 1], nodePts[2, 1] - nodePts[0, 1]],
            ]
        )

    def compute_area(self, nodePts: np.ndarray) -> float:
        """Computes the area of a quadrilateral element"""
        J = self.compute_jacobian(nodePts)
        return np.linalg.det(J)

    def transform_to_physical(
        self, points: np.ndarray, nodePts: np.ndarray
    ) -> np.ndarray:
        """Transforms points from natural coordinates to physical coordinates"""
        return np.dot(points, nodePts)
