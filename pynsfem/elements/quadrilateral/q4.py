"""
Q4 quadrangular element implementation
"""

import numpy as np
from typing import Tuple, List, Callable
from ...utils.quadrature import rect_gauss_points_weights
from ..base import Element


class ElementQuadQ4(Element):
    """Q4 quadrangular element with 4 DOF:
    - Point evaluation at each corner
    """

    def __init__(self):
        super().__init__()
        self.degree = 2
        self.num_dof = 4  # One DOF per vertex
        self.gauss_points, self.gauss_weights = rect_gauss_points_weights(order=self.degree)

        # Set up shape functions and their derivatives
        self._setup_shape_functions()

        # Update element definition with shape functions and DOF
        self.element_definition.shape_functions = self.shape_functions

    def _setup_shape_functions(self):
        """Sets up the shape functions and their derivatives"""
        # Shape functions in natural coordinates (xi, eta)
        self.shape_functions = [
            lambda xi, eta: 0.25 * (1 - xi) * (1 - eta),
            lambda xi, eta: 0.25 * (1 + xi) * (1 - eta),
            lambda xi, eta: 0.25 * (1 + xi) * (1 + eta),
            lambda xi, eta: 0.25 * (1 - xi) * (1 + eta),
        ]

        # Derivatives with respect to xi
        self.dN_dxi = [
            lambda xi, eta: -0.25 * (1 - eta),
            lambda xi, eta: 0.25 * (1 - eta),
            lambda xi, eta: 0.25 * (1 + eta),
            lambda xi, eta: -0.25 * (1 + eta),
        ]

        # Derivatives with respect to eta
        self.dN_deta = [
            lambda xi, eta: -0.25 * (1 - xi),
            lambda xi, eta: -0.25 * (1 + xi),
            lambda xi, eta: 0.25 * (1 + xi),
            lambda xi, eta: 0.25 * (1 - xi),
        ]

        # Set shape function derivatives for base class
        self.shape_function_derivatives = (self.dN_dxi, self.dN_deta)

        # Evaluate shape functions at Gauss points
        self.N_gp = np.array([[N(xi, eta) for xi, eta in self.gauss_points] for N in self.shape_functions])

        # Evaluate derivatives at Gauss points
        self.dN_dxi_gp = np.array([[dN(xi, eta) for xi, eta in self.gauss_points] for dN in self.dN_dxi])
        self.dN_deta_gp = np.array([[dN(xi, eta) for xi, eta in self.gauss_points] for dN in self.dN_deta])

    def define_elements(self, V: np.ndarray, F: np.ndarray) -> Tuple[np.ndarray, int]:
        """Defines element connectivity and number of unknowns"""
        unknowns = len(V)  # One DOF per vertex
        Elements = F  # Element connectivity is just the face indices
        return Elements, unknowns

    def project_function(self, V: np.ndarray, F: np.ndarray, Elements: np.ndarray, func: Callable) -> np.ndarray:
        """Projects a function onto the finite element space"""
        projection = np.zeros((np.max(Elements) + 1))

        # Project point values
        projection[: len(V)] = np.array(list(map(lambda p: func(p[0], p[1]), V)))

        # Assembly
        matA = np.zeros((len(projection), len(projection)))
        vecb = np.zeros((len(projection)))

        for i, el in enumerate(Elements):
            nodePts = V[el]
            detJ = self._compute_area(nodePts)

            # Element stiffness matrix and load vector
            el_A = self._compute_stiffness_matrix(nodePts)
            el_b = self._compute_load_vector(nodePts, func)

            matA[np.ix_(el, el)] += el_A
            vecb[el] += el_b

        return np.linalg.solve(matA, vecb)

    def _transform_to_physical(self, points: np.ndarray, nodePts: np.ndarray) -> np.ndarray:
        """Transforms points from natural coordinates to physical coordinates"""
        return [np.dot(self.N_gp[:, i], nodePts) for i in range(len(points))]
