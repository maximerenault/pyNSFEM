import numpy as np
from typing import Tuple, Set, List

from .topology import Topology
from .geometry import Geometry

class Mesh:
    """
    A class that combines topological and geometric information about a finite element mesh.
    """
    
    def __init__(self, elements: np.ndarray, nodes: np.ndarray):
        """
        Initialize the mesh with elements and nodes.
        
        Args:
            elements: Array of shape (n_elements, n_nodes_per_element) containing element connectivity
            nodes: Array of shape (n_nodes, n_dimensions) containing node coordinates
        """
        self.topology = Topology(elements)
        self.geometry = Geometry(nodes)
        self.n_elements = self.topology.n_elements
        self.n_nodes = self.geometry.n_nodes
        self.n_dimensions = self.geometry.n_dimensions
    
    # Topology methods
    def get_element_neighbors(self, element_id: int) -> Set[int]:
        """Get neighboring elements."""
        return self.topology.get_element_neighbors(element_id)
    
    def get_node_neighbors(self, node_id: int) -> Set[int]:
        """Get neighboring nodes."""
        return self.topology.get_node_neighbors(node_id)
    
    def get_edge_nodes(self, edge: Tuple[int, int]) -> Set[int]:
        """Get nodes connected to an edge."""
        return self.topology.get_edge_nodes(edge)
    
    def get_node_elements(self, node_id: int) -> Set[int]:
        """Get elements containing a node."""
        return self.topology.get_node_elements(node_id)
    
    def get_element_edges(self, element_id: int) -> List[Tuple[int, int]]:
        """Get edges of an element."""
        return self.topology.get_element_edges(element_id)
    
    def get_element_nodes(self, element_id: int) -> np.ndarray:
        """Get nodes of an element."""
        return self.topology.get_element_nodes(element_id)
    
    def get_unique_edges(self) -> Set[Tuple[int, int]]:
        """Get all unique edges in the mesh."""
        return self.topology.get_unique_edges()
    
    def get_edge_elements(self, edge: Tuple[int, int]) -> Set[int]:
        """Get elements containing an edge."""
        return self.topology.get_edge_elements(edge)
    
    # Geometry methods
    def get_node_coordinates(self, node_id: int) -> np.ndarray:
        """Get coordinates of a node."""
        return self.geometry.get_node_coordinates(node_id)
    
    def get_element_coordinates(self, element_id: int) -> np.ndarray:
        """Get coordinates of all nodes in an element."""
        element_nodes = self.get_element_nodes(element_id)
        return self.geometry.get_element_coordinates(element_nodes)
    
    def get_edge_coordinates(self, edge: Tuple[int, int]) -> np.ndarray:
        """Get coordinates of nodes defining an edge."""
        return self.geometry.get_edge_coordinates(edge)
    
    def get_edge_length(self, edge: Tuple[int, int]) -> float:
        """Get length of an edge."""
        return self.geometry.get_edge_length(edge)
    
    def get_element_center(self, element_id: int) -> np.ndarray:
        """Get center point of an element."""
        element_nodes = self.get_element_nodes(element_id)
        return self.geometry.get_element_center(element_nodes)
    
    def get_element_volume(self, element_id: int) -> float:
        """Get volume/area of an element."""
        element_nodes = self.get_element_nodes(element_id)
        return self.geometry.get_element_volume(element_nodes)
    
    # Combined methods
    def get_element_quality(self, element_id: int) -> float:
        """
        Calculate a quality measure for the element.
        For 2D triangles, this is the ratio of inscribed circle radius to circumscribed circle radius.
        For 3D tetrahedra, this is the ratio of inscribed sphere radius to circumscribed sphere radius.
        
        Args:
            element_id: ID of the element
            
        Returns:
            Quality measure between 0 and 1 (1 being perfect)
        """
        if self.n_dimensions == 2:
            # For triangles, use the ratio of areas
            coords = self.get_element_coordinates(element_id)
            # Calculate area using shoelace formula
            x = coords[:, 0]
            y = coords[:, 1]
            j = np.arange(3)
            j1 = (j + 1) % 3
            area = 0.5 * np.abs(np.sum(x[j] * y[j1] - x[j1] * y[j]))
            
            # Calculate edge lengths
            edges = self.get_element_edges(element_id)
            lengths = [self.get_edge_length(edge) for edge in edges]
            max_length = max(lengths)
            
            # Quality measure based on area and edge lengths
            return area / (max_length ** 2)
        elif self.n_dimensions == 3:
            # For tetrahedra, use the ratio of volumes
            coords = self.get_element_coordinates(element_id)
            # Calculate volume using determinant method
            v1 = coords[1] - coords[0]
            v2 = coords[2] - coords[0]
            v3 = coords[3] - coords[0]
            volume = np.abs(np.linalg.det(np.stack([v1, v2, v3])))
            
            # Calculate edge lengths
            edges = self.get_element_edges(element_id)
            lengths = [self.get_edge_length(edge) for edge in edges]
            max_length = max(lengths)
            
            # Quality measure based on volume and edge lengths
            return volume / (max_length ** 3)
        else:
            raise ValueError(f"Unsupported number of dimensions: {self.n_dimensions}") 