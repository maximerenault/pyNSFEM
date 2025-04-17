import numpy as np
from typing import Tuple

class Geometry:
    """
    A class to handle geometric operations and properties of a finite element mesh.
    """
    
    def __init__(self, nodes: np.ndarray):
        """
        Initialize the geometry with node coordinates.
        
        Args:
            nodes: Array of shape (n_nodes, n_dimensions) containing node coordinates
        """
        self.nodes = nodes
        self.n_nodes = len(nodes)
        self.n_dimensions = nodes.shape[1]
    
    def get_node_coordinates(self, node_id: int) -> np.ndarray:
        """
        Get coordinates of a given node.
        
        Args:
            node_id: ID of the node
            
        Returns:
            Array of coordinates
        """
        return self.nodes[node_id]
    
    def get_element_coordinates(self, element_nodes: np.ndarray) -> np.ndarray:
        """
        Get coordinates of all nodes in an element.
        
        Args:
            element_nodes: Array of node IDs for the element
            
        Returns:
            Array of shape (n_nodes_per_element, n_dimensions)
        """
        return self.nodes[element_nodes]
    
    def get_edge_coordinates(self, edge: Tuple[int, int]) -> np.ndarray:
        """
        Get coordinates of the nodes defining an edge.
        
        Args:
            edge: Tuple of two node IDs defining the edge
            
        Returns:
            Array of shape (2, n_dimensions) containing coordinates of both nodes
        """
        return self.nodes[list(edge)]
    
    def get_edge_length(self, edge: Tuple[int, int]) -> float:
        """
        Calculate the length of an edge.
        
        Args:
            edge: Tuple of two node IDs defining the edge
            
        Returns:
            Length of the edge
        """
        coords = self.get_edge_coordinates(edge)
        return np.linalg.norm(coords[1] - coords[0])
    
    def get_element_center(self, element_nodes: np.ndarray) -> np.ndarray:
        """
        Calculate the center point of an element.
        
        Args:
            element_nodes: Array of node IDs for the element
            
        Returns:
            Array of shape (n_dimensions,) containing the center coordinates
        """
        coords = self.get_element_coordinates(element_nodes)
        return np.mean(coords, axis=0)
    
    def get_element_volume(self, element_nodes: np.ndarray) -> float:
        """
        Calculate the volume/area of an element.
        For 1D elements, returns the length.
        For 2D elements, returns the area.
        For 3D elements, returns the volume.
        
        Args:
            element_nodes: Array of node IDs for the element
            
        Returns:
            Volume/area of the element
        """
        coords = self.get_element_coordinates(element_nodes)
        if self.n_dimensions == 1:
            return self.get_edge_length(element_nodes)
        elif self.n_dimensions == 2:
            # Calculate area using shoelace formula
            x = coords[:, 0]
            y = coords[:, 1]
            j = np.arange(len(coords))
            j1 = (j + 1) % len(coords)
            return 0.5 * np.abs(np.sum(x[j] * y[j1] - x[j1] * y[j]))
        elif self.n_dimensions == 3:
            # Calculate volume using determinant method
            # This is a simplified version - you might want to implement a more robust method
            # depending on your element types
            v1 = coords[1] - coords[0]
            v2 = coords[2] - coords[0]
            v3 = coords[3] - coords[0]
            return np.abs(np.linalg.det(np.stack([v1, v2, v3])))
        else:
            raise ValueError(f"Unsupported number of dimensions: {self.n_dimensions}") 