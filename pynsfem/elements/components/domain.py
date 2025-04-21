"""
Element domain representation for finite elements
"""

from dataclasses import dataclass
import numpy as np
from math import ceil, sqrt
from enum import Enum, auto
from ...utils.power_tuples import power_tuples_sum


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

    def sample(self, num_points: int) -> np.ndarray:
        """Samples points in the domain"""
        raise NotImplementedError("Sampling not implemented for this domain type")

    def compute_jacobian(self, nodal_coordinates: np.ndarray) -> np.ndarray:
        """Computes the Jacobian matrix for coordinate transformation"""
        raise NotImplementedError("Must be implemented by subclasses")

    def compute_measure(self, nodal_coordinates: np.ndarray) -> float:
        """Computes the measure (length/area/volume) of the element"""
        raise NotImplementedError("Must be implemented by subclasses")

    def transform_to_physical(
        self, points: np.ndarray, nodal_coordinates: np.ndarray
    ) -> np.ndarray:
        """Transforms points from reference coordinates to physical coordinates"""
        raise NotImplementedError("Must be implemented by subclasses")

    def transform_to_reference(
        self, points: np.ndarray, nodal_coordinates: np.ndarray
    ) -> np.ndarray:
        """Transforms points from physical coordinates to reference coordinates"""
        raise NotImplementedError("Must be implemented by subclasses")


class TriangleDomain(Domain):
    """Triangle element domain"""

    def __init__(self, points: np.ndarray, dim: int):
        super().__init__(
            domain_type=DomainType.TRIANGLE, reference_coordinates=points, dimension=dim
        )

    def sample(self, num_points: int) -> np.ndarray:
        """Samples points following Lagrange's rule in the triangle"""
        vertices = np.array(self.reference_coordinates)
        side_n = (-3 + sqrt(1 + 8 * num_points)) / 2
        side_n = ceil(side_n)
        points = []
        vec_01 = vertices[1] - vertices[0]
        vec_02 = vertices[2] - vertices[0]
        for n in range(side_n + 1):
            for t in power_tuples_sum(2, n):
                points.append(vertices[0] + (t[0] * vec_01 + t[1] * vec_02) / side_n)
        return np.array(points)

    def compute_jacobian(self, nodal_coordinates: np.ndarray) -> np.ndarray:
        """Computes Jacobian for triangle element"""
        return nodal_coordinates[1:, :] - nodal_coordinates[0]

    def compute_measure(self, nodal_coordinates: np.ndarray) -> float:
        """Computes area of triangle element"""
        J = self.compute_jacobian(nodal_coordinates)
        return 0.5 * np.linalg.det(J)

    def transform_to_physical(
        self, points: np.ndarray, nodal_coordinates: np.ndarray
    ) -> np.ndarray:
        """Transforms from reference to physical coordinates"""
        return np.dot(points, nodal_coordinates)

    def transform_to_reference(
        self, points: np.ndarray, nodal_coordinates: np.ndarray
    ) -> np.ndarray:
        """Transforms from physical to reference coordinates"""
        # Implement inverse transformation
        J = self.compute_jacobian(nodal_coordinates)
        J_inv = np.linalg.inv(J)
        return np.dot(points - nodal_coordinates[0], J_inv)

class QuadrilateralDomain(Domain):
    """Quadrilateral element domain"""

    def __init__(self, points: np.ndarray, dim: int):
        super().__init__(
            domain_type=DomainType.QUADRILATERAL,
            reference_coordinates=points,
            dimension=dim,
        )

    def sample(self, num_points: int) -> np.ndarray:
        """Samples points in the quadrilateral domain"""
        side_n = ceil(sqrt(num_points))
        num_points = side_n * (side_n + 1) // 2
        tri1 = TriangleDomain(
            points=self.reference_coordinates[[1, 2, 0]], dim=self.dimension
        )
        points1 = tri1.sample(num_points)
        tri2 = TriangleDomain(
            points=self.reference_coordinates[[3, 0, 2]], dim=self.dimension
        )
        points2 = tri2.sample(num_points)
        points = np.concatenate((points1, points2[:-side_n]))
        return points