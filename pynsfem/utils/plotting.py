import numpy as np
import matplotlib.pyplot as plt
from typing import Optional, Tuple, List
from ..core import Mesh

def plot_mesh(mesh: Mesh, show_nodes: bool = True, show_elements: bool = True,
              show_boundary: bool = True, title: Optional[str] = None) -> None:
    """
    Plot the finite element mesh.
    
    Args:
        mesh: Mesh object to plot
        show_nodes: Whether to show nodes
        show_elements: Whether to show elements
        show_boundary: Whether to highlight boundary
        title: Optional title for the plot
    """
    plt.figure(figsize=(10, 10))
    
    # Plot elements
    if show_elements:
        for e in range(mesh.n_elements):
            nodes = mesh.topology.get_element_nodes(e)
            coords = mesh.geometry.get_element_coordinates(e)
            
            # Plot element edges
            for i in range(len(nodes)):
                j = (i + 1) % len(nodes)
                plt.plot([coords[i, 0], coords[j, 0]],
                        [coords[i, 1], coords[j, 1]],
                        'b-', alpha=0.5)
    
    # Plot nodes
    if show_nodes:
        coords = mesh.geometry.get_all_coordinates()
        plt.plot(coords[:, 0], coords[:, 1], 'ko', markersize=5)
    
    # Highlight boundary
    if show_boundary:
        boundary_nodes = mesh.topology.get_boundary_nodes()
        if len(boundary_nodes) > 0:
            boundary_coords = mesh.geometry.get_node_coordinates(boundary_nodes)
            plt.plot(boundary_coords[:, 0], boundary_coords[:, 1],
                    'r-', linewidth=2, label='Boundary')
    
    plt.grid(True)
    plt.axis('equal')
    if title:
        plt.title(title)
    plt.show()

def plot_solution(mesh: Mesh, u: np.ndarray, n_points: int = 100,
                 title: Optional[str] = None, cmap: str = 'viridis') -> None:
    """
    Plot the solution field on the mesh.
    
    Args:
        mesh: Mesh object
        u: Solution vector
        n_points: Number of points per element edge for interpolation
        title: Optional title for the plot
        cmap: Colormap to use
    """
    plt.figure(figsize=(12, 10))
    
    # Create triangulation for plotting
    coords = mesh.geometry.get_all_coordinates()
    elements = mesh.topology.get_all_elements()
    
    # Plot elements with interpolated solution
    for e in range(mesh.n_elements):
        nodes = mesh.topology.get_element_nodes(e)
        element_coords = mesh.geometry.get_element_coordinates(e)
        
        # Create interpolation points
        xi = np.linspace(0, 1, n_points)
        eta = np.linspace(0, 1, n_points)
        XI, ETA = np.meshgrid(xi, eta)
        
        # Interpolate solution
        u_interp = np.zeros((n_points, n_points))
        for i in range(n_points):
            for j in range(n_points):
                N = mesh.element.get_basis_functions(XI[i, j], ETA[i, j])
                u_interp[i, j] = np.sum(N * u[nodes])
        
        # Transform to physical coordinates
        x = np.zeros((n_points, n_points))
        y = np.zeros((n_points, n_points))
        for i in range(n_points):
            for j in range(n_points):
                N = mesh.element.get_basis_functions(XI[i, j], ETA[i, j])
                x[i, j] = np.sum(N * element_coords[:, 0])
                y[i, j] = np.sum(N * element_coords[:, 1])
        
        # Plot element
        plt.pcolormesh(x, y, u_interp, cmap=cmap, shading='auto')
    
    # Add colorbar
    plt.colorbar(label='Solution')
    
    # Plot mesh edges
    for e in range(mesh.n_elements):
        nodes = mesh.topology.get_element_nodes(e)
        coords = mesh.geometry.get_element_coordinates(e)
        
        for i in range(len(nodes)):
            j = (i + 1) % len(nodes)
            plt.plot([coords[i, 0], coords[j, 0]],
                    [coords[i, 1], coords[j, 1]],
                    'k-', alpha=0.3)
    
    plt.grid(True)
    plt.axis('equal')
    if title:
        plt.title(title)
    plt.show() 