import numpy as np
from typing import Tuple, List, Optional
from ..core import Mesh, Topology, Geometry

def create_rectangle_mesh(width: float, height: float, nx: int, ny: int,
                        element_type: str = 'tri') -> Mesh:
    """
    Create a rectangular mesh.
    
    Args:
        width: Width of the rectangle
        height: Height of the rectangle
        nx: Number of elements in x-direction
        ny: Number of elements in y-direction
        element_type: Type of element ('tri' or 'quad')
        
    Returns:
        Mesh object
    """
    # Create nodes
    x = np.linspace(0, width, nx + 1)
    y = np.linspace(0, height, ny + 1)
    X, Y = np.meshgrid(x, y)
    coords = np.column_stack((X.ravel(), Y.ravel()))
    
    # Create elements
    if element_type == 'tri':
        elements = []
        for j in range(ny):
            for i in range(nx):
                n1 = j * (nx + 1) + i
                n2 = n1 + 1
                n3 = (j + 1) * (nx + 1) + i
                n4 = n3 + 1
                
                # First triangle
                elements.append([n1, n2, n3])
                # Second triangle
                elements.append([n2, n4, n3])
    else:  # quad
        elements = []
        for j in range(ny):
            for i in range(nx):
                n1 = j * (nx + 1) + i
                n2 = n1 + 1
                n3 = (j + 1) * (nx + 1) + i
                n4 = n3 + 1
                elements.append([n1, n2, n4, n3])
    
    # Create mesh
    topology = Topology(elements)
    geometry = Geometry(coords)
    return Mesh(topology, geometry)

def create_circle_mesh(radius: float, n_elements: int,
                      element_type: str = 'tri') -> Mesh:
    """
    Create a circular mesh.
    
    Args:
        radius: Radius of the circle
        n_elements: Number of elements along the circumference
        element_type: Type of element ('tri' or 'quad')
        
    Returns:
        Mesh object
    """
    # Create nodes
    theta = np.linspace(0, 2*np.pi, n_elements + 1)
    x = radius * np.cos(theta)
    y = radius * np.sin(theta)
    coords = np.column_stack((x, y))
    
    # Create elements
    if element_type == 'tri':
        elements = []
        center = len(coords) - 1
        for i in range(n_elements):
            elements.append([i, (i + 1) % n_elements, center])
    else:  # quad
        elements = []
        for i in range(n_elements):
            elements.append([i, (i + 1) % n_elements, (i + 2) % n_elements, (i + 1) % n_elements])
    
    # Create mesh
    topology = Topology(elements)
    geometry = Geometry(coords)
    return Mesh(topology, geometry)

def create_annulus_mesh(inner_radius: float, outer_radius: float,
                       n_elements: int, element_type: str = 'tri') -> Mesh:
    """
    Create an annular mesh.
    
    Args:
        inner_radius: Inner radius of the annulus
        outer_radius: Outer radius of the annulus
        n_elements: Number of elements along each circumference
        element_type: Type of element ('tri' or 'quad')
        
    Returns:
        Mesh object
    """
    # Create nodes
    theta = np.linspace(0, 2*np.pi, n_elements + 1)
    x_inner = inner_radius * np.cos(theta)
    y_inner = inner_radius * np.sin(theta)
    x_outer = outer_radius * np.cos(theta)
    y_outer = outer_radius * np.sin(theta)
    
    coords = np.vstack((
        np.column_stack((x_inner, y_inner)),
        np.column_stack((x_outer, y_outer))
    ))
    
    # Create elements
    if element_type == 'tri':
        elements = []
        for i in range(n_elements):
            # Inner triangle
            elements.append([i, (i + 1) % n_elements, i + n_elements + 1])
            # Outer triangle
            elements.append([(i + 1) % n_elements, (i + 1) % n_elements + n_elements + 1, i + n_elements + 1])
    else:  # quad
        elements = []
        for i in range(n_elements):
            elements.append([
                i,
                (i + 1) % n_elements,
                (i + 1) % n_elements + n_elements + 1,
                i + n_elements + 1
            ])
    
    # Create mesh
    topology = Topology(elements)
    geometry = Geometry(coords)
    return Mesh(topology, geometry)

def create_triangle_mesh(base: float, height: float, n_elements: int) -> Mesh:
    """
    Create a triangular mesh.
    
    Args:
        base: Length of the base
        height: Height of the triangle
        n_elements: Number of elements along each edge
        
    Returns:
        Mesh object
    """
    # Create nodes
    x = np.linspace(0, base, n_elements + 1)
    y = np.linspace(0, height, n_elements + 1)
    X, Y = np.meshgrid(x, y)
    
    # Filter nodes to create triangle shape
    mask = Y <= height * (1 - X/base)
    coords = np.column_stack((X[mask], Y[mask]))
    
    # Create elements
    elements = []
    for j in range(n_elements):
        for i in range(n_elements - j):
            n1 = j * (n_elements + 1) + i
            n2 = n1 + 1
            n3 = (j + 1) * (n_elements + 1) + i
            n4 = n3 + 1
            
            # First triangle
            elements.append([n1, n2, n3])
            # Second triangle
            elements.append([n2, n4, n3])
    
    # Create mesh
    topology = Topology(elements)
    geometry = Geometry(coords)
    return Mesh(topology, geometry) 