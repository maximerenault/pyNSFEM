"""
Poisson equation solver using finite elements
"""

import numpy as np
from typing import Tuple, List, Dict, Any
from ..core import Mesh
from ..elements import ElementTriCubicHermite, ElementTriQuinticArgyris, ElementQuadQ4


class PoissonSolver:
    """
    Solver for the Poisson equation using finite elements.
    """
    
    def __init__(self, mesh: Mesh, element_type: str = 'tri_cubic_hermite'):
        """
        Initialize the Poisson solver.
        
        Args:
            mesh: Mesh object containing topology and geometry
            element_type: Type of element to use ('tri_cubic_hermite', 'tri_quintic_argyris', 'quad_q4', 'poly_basis')
        """
        self.mesh = mesh
        self.topology = mesh.topology
        self.geometry = mesh.geometry
        
        # Initialize element
        if element_type == 'tri_cubic_hermite':
            self.element = ElementTriCubicHermite()
        elif element_type == 'tri_quintic_argyris':
            self.element = ElementTriQuinticArgyris()
        elif element_type == 'quad_q4':
            self.element = ElementQuadQ4()
        else:
            raise ValueError(f"Unsupported element type: {element_type}")
        
        # Initialize system matrices
        self.K = None  # Stiffness matrix
        self.f = None  # Load vector
        self.u = None  # Solution vector
        
    def assemble_system(self, f: callable = None, g: callable = None) -> Tuple[np.ndarray, np.ndarray]:
        """
        Assemble the finite element system.
        
        Args:
            f: Source term function (optional)
            g: Dirichlet boundary condition function (optional)
            
        Returns:
            Tuple of (stiffness matrix, load vector)
        """
        n_dofs = self.mesh.n_nodes * self.element.n_dofs_per_node
        K = np.zeros((n_dofs, n_dofs))
        f_vec = np.zeros(n_dofs)
        
        # Get Gauss points and weights
        points, weights = self.element.get_gauss_points()
        
        # Loop over elements
        for e in range(self.mesh.n_elements):
            # Get element nodes and coordinates
            nodes = self.topology.get_element_nodes(e)
            coords = self.geometry.get_element_coordinates(e)
            
            # Get element stiffness matrix and load vector
            Ke, fe = self._assemble_element(nodes, coords, points, weights, f)
            
            # Assembly into global system
            for i in range(len(nodes)):
                for j in range(len(nodes)):
                    K[nodes[i], nodes[j]] += Ke[i, j]
                f_vec[nodes[i]] += fe[i]
        
        # Apply boundary conditions if provided
        if g is not None:
            K, f_vec = self._apply_boundary_conditions(K, f_vec, g)
        
        self.K = K
        self.f = f_vec
        return K, f_vec
    
    def _assemble_element(self, nodes: List[int], coords: np.ndarray,
                         points: np.ndarray, weights: np.ndarray,
                         f: callable = None) -> Tuple[np.ndarray, np.ndarray]:
        """
        Assemble element matrices.
        
        Args:
            nodes: List of element node indices
            coords: Element node coordinates
            points: Gauss quadrature points
            weights: Gauss quadrature weights
            f: Source term function (optional)
            
        Returns:
            Tuple of (element stiffness matrix, element load vector)
        """
        n_nodes = len(nodes)
        Ke = np.zeros((n_nodes, n_nodes))
        fe = np.zeros(n_nodes)
        
        # Loop over Gauss points
        for i, (xi, eta) in enumerate(points):
            # Get basis functions and derivatives
            N = self.element.get_basis_functions(xi, eta)
            dN_dxi, dN_deta = self.element.get_basis_derivatives(xi, eta)
            
            # Compute Jacobian
            J = np.zeros((2, 2))
            for j in range(n_nodes):
                J[0, 0] += dN_dxi[j] * coords[j, 0]
                J[0, 1] += dN_dxi[j] * coords[j, 1]
                J[1, 0] += dN_deta[j] * coords[j, 0]
                J[1, 1] += dN_deta[j] * coords[j, 1]
            
            # Compute determinant and inverse
            detJ = np.linalg.det(J)
            invJ = np.linalg.inv(J)
            
            # Compute derivatives with respect to physical coordinates
            dN_dx = np.zeros(n_nodes)
            dN_dy = np.zeros(n_nodes)
            for j in range(n_nodes):
                dN_dx[j] = dN_dxi[j] * invJ[0, 0] + dN_deta[j] * invJ[0, 1]
                dN_dy[j] = dN_dxi[j] * invJ[1, 0] + dN_deta[j] * invJ[1, 1]
            
            # Assemble element stiffness matrix
            for j in range(n_nodes):
                for k in range(n_nodes):
                    Ke[j, k] += (dN_dx[j] * dN_dx[k] + dN_dy[j] * dN_dy[k]) * detJ * weights[i]
            
            # Assemble element load vector if source term is provided
            if f is not None:
                # Compute physical coordinates at Gauss point
                x = np.sum(N * coords[:, 0])
                y = np.sum(N * coords[:, 1])
                f_val = f(x, y)
                
                for j in range(n_nodes):
                    fe[j] += N[j] * f_val * detJ * weights[i]
        
        return Ke, fe
    
    def _apply_boundary_conditions(self, K: np.ndarray, f: np.ndarray,
                                 g: callable) -> Tuple[np.ndarray, np.ndarray]:
        """
        Apply Dirichlet boundary conditions.
        
        Args:
            K: Stiffness matrix
            f: Load vector
            g: Dirichlet boundary condition function
            
        Returns:
            Tuple of (modified stiffness matrix, modified load vector)
        """
        # Get boundary nodes
        boundary_nodes = self.topology.get_boundary_nodes()
        
        # Apply boundary conditions
        for node in boundary_nodes:
            coords = self.geometry.get_node_coordinates(node)
            g_val = g(coords[0], coords[1])
            
            # Modify load vector
            f -= K[:, node] * g_val
            
            # Modify stiffness matrix
            K[node, :] = 0
            K[node, node] = 1
            f[node] = g_val
        
        return K, f
    
    def solve(self) -> np.ndarray:
        """
        Solve the Poisson equation.
        
        Returns:
            Solution vector
        """
        if self.K is None or self.f is None:
            raise ValueError("System must be assembled before solving")
        
        # Solve system
        self.u = np.linalg.solve(self.K, self.f)
        return self.u
    
    def get_solution_at_point(self, x: float, y: float) -> float:
        """
        Get solution value at a given point.
        
        Args:
            x: x-coordinate
            y: y-coordinate
            
        Returns:
            Solution value at the point
        """
        if self.u is None:
            raise ValueError("Solution must be computed before evaluating")
        
        # Find element containing the point
        element_idx = self.geometry.find_element_containing_point(x, y)
        if element_idx is None:
            raise ValueError("Point is not in the domain")
        
        # Get element nodes and coordinates
        nodes = self.topology.get_element_nodes(element_idx)
        coords = self.geometry.get_element_coordinates(element_idx)
        
        # Compute reference coordinates
        xi, eta = self.geometry.get_reference_coordinates(x, y, coords)
        
        # Get basis functions
        N = self.element.get_basis_functions(xi, eta)
        
        # Compute solution value
        u_val = 0
        for i, node in enumerate(nodes):
            u_val += N[i] * self.u[node]
        
        return u_val
