"""
Any degree Lagrange triangular element implementation
"""

import numpy as np
from typing import Tuple, Callable, List
from ..base import TriElement
from ...utils.poly_basis import tri_poly_basis_auto
from ...utils.tri_subdiv import tri_subdiv

class ElementTriLagrange(TriElement):
    """Any degree Lagrange triangular element"""

    def __init__(self, degree: int):
        super().__init__()
        self.degree = degree
        self.num_dof = (degree + 1) * (degree + 2) // 2

        # Set up shape functions
        self.func, self.grad, self.hess, self.mat = self._create_vandermonde_matrix()
        self.inv_mat = np.linalg.inv(self.mat)

        # Define shape functions and their derivatives
        self.shape_functions = self._define_shape_functions()
        self.shape_function_derivatives = self._define_derivatives()
        self.shape_function_second_derivatives = self._define_second_derivatives()

        # Update element definition with shape functions and DOF
        self.element_definition.shape_functions = self.shape_functions

    def _get_vandermonde_columns(self, func: Callable, grad: Callable, hess: Callable) -> List[np.ndarray]:
        """Gets the columns of the Vandermonde matrix"""
        return (list(func(x, y)) for x, y in tri_subdiv(self.degree))

    def _define_shape_functions(self) -> List[Callable]:
        """Defines the shape functions for the element"""
        return [lambda x, y, i=i: self.inv_mat[i, :].dot(self.func(x, y)) for i in range(self.num_dof)]

    def _define_derivatives(self) -> Tuple[List[Callable], List[Callable]]: 
        """Defines the derivatives of the shape functions"""
        dN_dxi = [lambda x, y, i=i: self.inv_mat[i, :].dot(self.grad(x, y)[0]) for i in range(self.num_dof)]
        dN_deta = [lambda x, y, i=i: self.inv_mat[i, :].dot(self.grad(x, y)[1]) for i in range(self.num_dof)]
        return dN_dxi, dN_deta
    
    def _define_second_derivatives(self) -> Tuple[List[Callable], List[Callable], List[Callable]]:
        """Defines the second derivatives of the shape functions"""
        d2N_dxi2 = [lambda x, y, i=i: self.inv_mat[i, :].dot(self.hess(x, y)[0]) for i in range(self.num_dof)]
        d2N_dxieta = [lambda x, y, i=i: self.inv_mat[i, :].dot(self.hess(x, y)[1]) for i in range(self.num_dof)]
        d2N_deta2 = [lambda x, y, i=i: self.inv_mat[i, :].dot(self.hess(x, y)[2]) for i in range(self.num_dof)]
        return d2N_dxi2, d2N_dxieta, d2N_deta2

    def define_elements(self, V: np.ndarray, F: np.ndarray) -> Tuple[np.ndarray, int]:
        """Defines the elements for the element"""
        return np.array([np.arange(self.num_dof)]), self.num_dof

    def project_function(self, V: np.ndarray, F: np.ndarray, func: Callable) -> np.ndarray:
        """Projects a function onto the element"""
        return np.array([func(x, y) for x, y in self.Lin_transform.dot(V[F])])

