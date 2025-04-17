"""
Element domain representation for finite elements
"""

from dataclasses import dataclass
from typing import List, Optional, Tuple, Callable
import numpy as np
from enum import Enum, auto

class DomainType(Enum):
    """Types of element domains"""
    LINE = auto()
    TRIANGLE = auto()
    QUADRILATERAL = auto()
    TETRAHEDRON = auto()
    HEXAHEDRON = auto()

@dataclass
class Domain:
    """Represents a geometric domain for finite elements
    
    The domain is characterized by:
    - Its type (triangle, quadrilateral, etc.)
    - Its reference coordinates
    - Its dimension
    - Coordinate mapping functions
    """
    
    domain_type: DomainType
    reference_coordinates: np.ndarray
    dimension: int
    
    def compute_jacobian(self, nodal_coordinates: np.ndarray) -> np.ndarray:
        """Computes the Jacobian matrix for coordinate transformation"""
        raise NotImplementedError("Must be implemented by subclasses")
    
    def compute_measure(self, nodal_coordinates: np.ndarray) -> float:
        """Computes the measure (length/area/volume) of the element"""
        raise NotImplementedError("Must be implemented by subclasses")
    
    def transform_to_physical(self, points: np.ndarray, nodal_coordinates: np.ndarray) -> np.ndarray:
        """Transforms points from reference coordinates to physical coordinates"""
        raise NotImplementedError("Must be implemented by subclasses")
    
    def transform_to_reference(self, points: np.ndarray, nodal_coordinates: np.ndarray) -> np.ndarray:
        """Transforms points from physical coordinates to reference coordinates"""
        raise NotImplementedError("Must be implemented by subclasses")


class TriangleDomain(Domain):
    """Triangle element domain"""
    
    def __init__(self):
        super().__init__(
            domain_type=DomainType.TRIANGLE,
            reference_coordinates=np.array([[0, 0], [1, 0], [0, 1]]),
            dimension=2
        )
    
    def compute_jacobian(self, nodal_coordinates: np.ndarray) -> np.ndarray:
        """Computes Jacobian for triangle element"""
        return nodal_coordinates[1:, :] - nodal_coordinates[0]
    
    def compute_measure(self, nodal_coordinates: np.ndarray) -> float:
        """Computes area of triangle element"""
        J = self.compute_jacobian(nodal_coordinates)
        return 0.5 * np.linalg.det(J)
    
    def transform_to_physical(self, points: np.ndarray, nodal_coordinates: np.ndarray) -> np.ndarray:
        """Transforms from reference to physical coordinates"""
        return np.dot(points, nodal_coordinates)
    
    def transform_to_reference(self, points: np.ndarray, nodal_coordinates: np.ndarray) -> np.ndarray:
        """Transforms from physical to reference coordinates"""
        # Implement inverse transformation
        J = self.compute_jacobian(nodal_coordinates)
        J_inv = np.linalg.inv(J)
        return np.dot(points - nodal_coordinates[0], J_inv)