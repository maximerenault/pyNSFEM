"""
Quintic Argyris triangular element implementation
"""

import numpy as np
from typing import Tuple, List, Callable
from ..element import Element


class ElementTriQuinticArgyris(Element):
    """Quintic Argyris triangular element with 21 DOF:
    - Point evaluation at each corner
    - First and second derivatives at each corner
    - Normal derivative at edge midpoints
    """

    def __init__(self):
        super().__init__()
        self.degree = 5
        self.num_dof = 21  # 3 vertices × (1 value + 2 first derivatives + 3 second derivatives) + 3 edge midpoints

        # Set up shape functions and their derivatives
        self.func, self.grad, self.hess, self.mat = self._create_vandermonde_matrix()
        self.inv_mat = np.linalg.inv(self.mat)

        # Define shape functions and their derivatives
        self.shape_functions = self._define_shape_functions()
        self.shape_function_derivatives = self._define_derivatives()
        self.shape_function_second_derivatives = self._define_second_derivatives()

        # Update element definition with shape functions and DOF
        self.element_definition.shape_functions = self.shape_functions
        self.element_definition.degrees_of_freedom = self._define_degrees_of_freedom()

    def _get_vandermonde_columns(self, func: Callable, grad: Callable, hess: Callable) -> List[np.ndarray]:
        """Gets the columns of the Vandermonde matrix"""
        return (
            list(func(0, 0)),  # Point value at (0,0)
            list(func(1, 0)),  # Point value at (1,0)
            list(func(0, 1)),  # Point value at (0,1)
            *np.array(list(grad(0, 0))).T.tolist(),  # First derivatives at (0,0)
            *np.array(list(grad(1, 0))).T.tolist(),  # First derivatives at (1,0)
            *np.array(list(grad(0, 1))).T.tolist(),  # First derivatives at (0,1)
            *np.array(list(hess(0, 0))).T.tolist(),  # Second derivatives at (0,0)
            *np.array(list(hess(1, 0))).T.tolist(),  # Second derivatives at (1,0)
            *np.array(list(hess(0, 1))).T.tolist(),  # Second derivatives at (0,1)
            np.array(list(grad(0.5, 0))).dot([0, -1]),  # Normal derivative at edge midpoint
            np.array(list(grad(0, 0.5))).dot([-1, 0]),  # Normal derivative at edge midpoint
            np.array(list(grad(0.5, 0.5))).dot([1, 1]) / np.sqrt(2),  # Normal derivative at edge midpoint
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
        """Defines global DOF indices and number of unknowns"""
        unknowns = len(V) * 6 + len(F) * 3  # 6 DOF per vertex + 3 DOF on edges
        El1 = F  # Point values
        El2 = F + len(V)  # First derivatives
        El3 = F + 2 * len(V)  # Second derivatives
        El4 = np.arange(0, len(F), 1, dtype=int).reshape(-1, 1) + 3 * len(V)  # Edge midpoints

        # Combine all DOF
        Elements = np.concatenate((El1, El2, El3, El4), axis=1)
        return Elements, unknowns

    def project_function(self, V: np.ndarray, F: np.ndarray, Elements: np.ndarray, func: Callable) -> np.ndarray:
        """Projects a function onto the finite element space"""
        projection = np.zeros((np.max(Elements) + 1))

        # Project point values
        projection[: len(V)] = np.array(list(map(lambda p: func(p[0], p[1]), V)))

        # Project derivatives
        for el in Elements:
            nodePts = V[el[:3]]
            area = self._compute_area(nodePts)

            # Project first and second derivatives
            grads = el[3:9]
            func_gp = list(map(lambda p: func(p[0], p[1]), self.Lin_transform.dot(nodePts)))
            projection[grads] += np.einsum("g, gj, g -> j", self.gauss_weights, self.Ns[:, 3:9], func_gp) * area

            # Project edge midpoints
            edge_mids = el[9:]
            projection[edge_mids] = np.array(list(map(lambda p: func(p[0], p[1]), np.sum(V[F], axis=1) / 3)))

        # Solve projection system
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
