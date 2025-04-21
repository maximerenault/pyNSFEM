"""
Any degree Lagrange triangular element implementation
"""

import numpy as np
from typing import List
from ..base import Element, ElementDefinition
from ..components.domain import TriangleDomain, DomainType
from ..components.function_space import PolynomialSpace
from ..components.dof import PointEvaluation
from ...utils.power_tuples import power_tuples_sum


class ElementTriLagrange(Element):
    """Any degree Lagrange triangular element"""

    def __init__(self, degree: int, points: np.ndarray = None):
        self.dim = 2
        if points is None:
            points = np.zeros((3, self.dim), dtype=float)
            points[1][0] = 1.0
            points[2][1] = 1.0
        assert (
            len(points) == 3
        ), "Reference triangle should be defined with three points"
        assert len(points[0]) == self.dim, "Dimension incompatibility"
        self.degree = degree
        element_definition = ElementDefinition(
            domain=TriangleDomain(points, self.dim),
            shape_function_space=PolynomialSpace(degree, DomainType.TRIANGLE),
            nodal_basis=self._define_dofs(points),
        )

        super().__init__(element_definition)

    def _define_dofs(self, points: np.ndarray) -> List[PointEvaluation]:
        """Defines the degrees of freedom for the element"""
        dofs = []
        vec_01 = points[1] - points[0]
        vec_02 = points[2] - points[0]
        for d in range(self.degree + 1):
            for t in power_tuples_sum(2, d):
                dofs.append(PointEvaluation((t[0] * vec_01 + t[1] * vec_02)/self.degree))
        return dofs