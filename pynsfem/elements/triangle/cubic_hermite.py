"""
Cubic Hermite triangular element implementation
"""

import numpy as np
from typing import Tuple, List, Callable
from ..base import Element


class ElementTriCubicHermite(Element):
    """Cubic Hermite triangular element with 10 DOF:
    - Point evaluation at each corner
    - Gradient evaluation at each corner
    - Point evaluation at the center
    """

    def __init__(self):
        super().__init__()
        self.degree = 3
        self.num_dof = 10  # 3 vertices × (1 value + 2 derivatives) + 1 center value

        # Set up shape functions and their derivatives
        self.func, self.grad, self.hess, self.mat = self._create_vandermonde_matrix()
        self.inv_mat = np.linalg.inv(self.mat)

        # Define shape functions and their derivatives
        self.shape_functions = self._define_shape_functions()
        self.shape_function_derivatives = self._define_derivatives()
        self.shape_function_second_derivatives = self._define_second_derivatives()

        # Update element definition with shape functions and DOF
        self.element_definition.shape_functions = self.shape_functions

    def _get_vandermonde_columns(self, func: Callable, grad: Callable) -> List[np.ndarray]:
        """Gets the columns of the Vandermonde matrix"""
        return (
            list(func(0, 0)),
            list(func(1, 0)),
            list(func(0, 1)),
            *np.array(list(grad(0, 0))).T.tolist(),
            *np.array(list(grad(1, 0))).T.tolist(),
            *np.array(list(grad(0, 1))).T.tolist(),
            list(func(1 / 3, 1 / 3)),
        )

    def _define_shape_functions(self) -> List[Callable]:
        """Defines the shape functions for the element"""
        return [lambda x, y, i=i: self.inv_mat[i, :].dot(self.func(x, y)) for i in range(self.num_dof)]

    def _define_derivatives(self) -> Tuple[List[Callable], List[Callable]]:
        """Defines the derivatives of shape functions"""
        dN_dxi = [lambda x, y, i=i: self.inv_mat[i, :].dot(self.grad(x, y)[0]) for i in range(self.num_dof)]
        dN_deta = [lambda x, y, i=i: self.inv_mat[i, :].dot(self.grad(x, y)[1]) for i in range(self.num_dof)]
        return dN_dxi, dN_deta
    
    def _define_second_derivatives(self) -> Tuple[List[Callable], List[Callable], List[Callable]]:
        """Defines the second derivatives of shape functions"""
        d2N_dxi2 = [lambda x, y, i=i: self.inv_mat[i, :].dot(self.hess(x, y)[0]) for i in range(self.num_dof)]
        d2N_dxieta = [lambda x, y, i=i: self.inv_mat[i, :].dot(self.hess(x, y)[1]) for i in range(self.num_dof)]
        d2N_deta2 = [lambda x, y, i=i: self.inv_mat[i, :].dot(self.hess(x, y)[2]) for i in range(self.num_dof)]
        return d2N_dxi2, d2N_dxieta, d2N_deta2

    def define_elements(self, V: np.ndarray, F: np.ndarray) -> Tuple[np.ndarray, int]:
        """Defines element connectivity and number of unknowns"""
        unknowns = len(V) * 3 + len(F)
        El1 = F
        El2 = F + len(V)  # gradx
        El3 = F + 2 * len(V)  # grady
        El2 = np.concatenate((El2[:, [0]], El3[:, [0]], El2[:, [1]], El3[:, [1]], El2[:, [2]], El3[:, [2]]), axis=1)
        El3 = np.arange(0, len(F), 1, dtype=int).reshape(-1, 1) + 3 * len(V)
        Elements = np.concatenate((El1, El2, El3), axis=1)
        return Elements, unknowns

    def project_function(self, V: np.ndarray, F: np.ndarray, Elements: np.ndarray, func: Callable) -> np.ndarray:
        """Projects a function onto the finite element space"""
        projection = np.zeros((np.max(Elements) + 1))
        projection[: len(V)] = np.array(list(map(lambda p: func(p[0], p[1]), V)))

        for el in Elements:
            nodePts = V[el[:3]]
            area = self._compute_area(nodePts)
            grads = el[3:-1]
            func_gp = list(map(lambda p: func(p[0], p[1]), self.Lin_transform.dot(nodePts)))
            projection[grads] += np.einsum("g, gj, g -> j", self.gauss_weights, self.Ns[:, 3:-1], func_gp) * area

        projection[-len(F) :] = np.array(list(map(lambda p: func(p[0], p[1]), np.sum(V[F], axis=1) / 3)))

        matA = np.zeros((len(projection), len(projection)))
        vecb = np.zeros((len(projection)))
        for i, el in enumerate(Elements):
            tri = F[i]
            nodePts = V[tri]
            area = self._compute_area(nodePts)
            el_A = self._compute_stiffness_matrix(nodePts)
            func_gp = list(map(lambda p: func(p[0], p[1]), self.Lin_transform.dot(nodePts)))
            el_b = self._compute_load_vector(nodePts, func)
            matA[np.ix_(el, el)] += el_A
            vecb[el] += el_b

        return np.linalg.solve(matA, vecb)
