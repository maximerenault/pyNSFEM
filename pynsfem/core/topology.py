import numpy as np
from typing import Dict, List, Set, Tuple, Optional

class Topology:
    """
    A class to handle topological relationships in a finite element mesh.
    Provides methods to access element-to-element, node-to-node, and edge-to-node relationships.
    """
    
    def __init__(self, elements: np.ndarray):
        """
        Initialize the topology with elements.
        
        Args:
            elements: Array of shape (n_elements, n_nodes_per_element) containing element connectivity
        """
        self.elements = elements
        self.n_elements = len(elements)
        self.n_nodes_per_element = elements.shape[1]
        
        # Initialize dictionaries to store topological relationships
        self._element_to_elements: Dict[int, Set[int]] = {}
        self._node_to_nodes: Dict[int, Set[int]] = {}
        self._edge_to_nodes: Dict[Tuple[int, int], Set[int]] = {}
        self._node_to_elements: Dict[int, Set[int]] = {}
        self._edge_to_elements: Dict[Tuple[int, int], Set[int]] = {}
        self._unique_edges: Set[Tuple[int, int]] = set()
        
        # Build all topological relationships
        self._build_topology()
    
    def _build_topology(self) -> None:
        """Build all topological relationships."""
        self._build_element_to_elements()
        self._build_node_to_nodes()
        self._build_edge_to_nodes()
        self._build_node_to_elements()
        self._build_edge_to_elements()
    
    def _build_element_to_elements(self) -> None:
        """Build element-to-element relationships based on shared nodes."""
        for i, elem1 in enumerate(self.elements):
            self._element_to_elements[i] = set()
            for j, elem2 in enumerate(self.elements):
                if i != j:
                    # Check if elements share any nodes
                    if any(node in elem2 for node in elem1):
                        self._element_to_elements[i].add(j)
    
    def _build_node_to_nodes(self) -> None:
        """Build node-to-node relationships based on element connectivity."""
        for i in range(self.n_nodes):
            self._node_to_nodes[i] = set()
            # Find all elements containing this node
            for elem in self.elements:
                if i in elem:
                    # Add all other nodes in these elements as neighbors
                    for node in elem:
                        if node != i:
                            self._node_to_nodes[i].add(node)
    
    def _build_edge_to_nodes(self) -> None:
        """Build edge-to-node relationships based on element edges."""
        for elem in self.elements:
            n_nodes = len(elem)
            for i in range(n_nodes):
                j = (i + 1) % n_nodes
                edge = tuple(sorted([elem[i], elem[j]]))
                if edge not in self._edge_to_nodes:
                    self._edge_to_nodes[edge] = set()
                # Add all nodes in elements containing this edge
                for other_elem in self.elements:
                    if all(node in other_elem for node in edge):
                        self._edge_to_nodes[edge].update(other_elem)
    
    def _build_node_to_elements(self) -> None:
        """Build node-to-element relationships."""
        for i in range(self.n_nodes):
            self._node_to_elements[i] = set()
            for j, elem in enumerate(self.elements):
                if i in elem:
                    self._node_to_elements[i].add(j)
    
    def _build_edge_to_elements(self) -> None:
        """Build edge-to-element relationships and collect unique edges."""
        for elem_id, elem in enumerate(self.elements):
            n_nodes = len(elem)
            for i in range(n_nodes):
                j = (i + 1) % n_nodes
                edge = tuple(sorted([elem[i], elem[j]]))
                self._unique_edges.add(edge)
                if edge not in self._edge_to_elements:
                    self._edge_to_elements[edge] = set()
                self._edge_to_elements[edge].add(elem_id)
    
    def get_element_neighbors(self, element_id: int) -> Set[int]:
        """
        Get all elements that share at least one node with the given element.
        
        Args:
            element_id: ID of the element
            
        Returns:
            Set of neighboring element IDs
        """
        return self._element_to_elements[element_id]
    
    def get_node_neighbors(self, node_id: int) -> Set[int]:
        """
        Get all nodes that are connected to the given node through elements.
        
        Args:
            node_id: ID of the node
            
        Returns:
            Set of neighboring node IDs
        """
        return self._node_to_nodes[node_id]
    
    def get_edge_nodes(self, edge: Tuple[int, int]) -> Set[int]:
        """
        Get all nodes that belong to elements containing the given edge.
        
        Args:
            edge: Tuple of two node IDs defining the edge
            
        Returns:
            Set of node IDs
        """
        return self._edge_to_nodes.get(tuple(sorted(edge)), set())
    
    def get_node_elements(self, node_id: int) -> Set[int]:
        """
        Get all elements that contain the given node.
        
        Args:
            node_id: ID of the node
            
        Returns:
            Set of element IDs
        """
        return self._node_to_elements[node_id]
    
    def get_element_edges(self, element_id: int) -> List[Tuple[int, int]]:
        """
        Get all edges of a given element.
        
        Args:
            element_id: ID of the element
            
        Returns:
            List of edges as tuples of node IDs
        """
        elem = self.elements[element_id]
        n_nodes = len(elem)
        return [(elem[i], elem[(i + 1) % n_nodes]) for i in range(n_nodes)]
    
    def get_element_nodes(self, element_id: int) -> np.ndarray:
        """
        Get all nodes of a given element.
        
        Args:
            element_id: ID of the element
            
        Returns:
            Array of node IDs
        """
        return self.elements[element_id]
    
    def get_unique_edges(self) -> Set[Tuple[int, int]]:
        """
        Get all unique edges in the mesh.
        
        Returns:
            Set of edges as tuples of node IDs
        """
        return self._unique_edges
    
    def get_edge_elements(self, edge: Tuple[int, int]) -> Set[int]:
        """
        Get all elements that contain the given edge.
        
        Args:
            edge: Tuple of two node IDs defining the edge
            
        Returns:
            Set of element IDs
        """
        return self._edge_to_elements.get(tuple(sorted(edge)), set())
