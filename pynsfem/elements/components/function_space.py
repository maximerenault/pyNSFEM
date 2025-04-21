"""
Function space representation for finite elements
"""

from dataclasses import dataclass
from enum import Enum, auto
from .basis import Basis, PolynomialBasisFunction
from .domain import DomainType
from ...utils.power_tuples import power_tuples_max, power_tuples_sum


class FunctionSpaceType(Enum):
    """Types of function spaces"""

    POLYNOMIAL = auto()
    TRIGONOMETRIC = auto()
    NURBS = auto()
    EXPONENTIAL = auto()


@dataclass
class FunctionSpace:
    """Represents a function space for finite elements

    A function space is characterized by:
    - Its type (polynomial, trigonometric, etc.)
    - Its canonical basis
    - Its dimension (number of basis functions)
    """

    space_type: FunctionSpaceType
    basis: Basis
    dimension: int


class PolynomialSpace(FunctionSpace):
    """Polynomial function space of given degree"""

    def __init__(self, degree: int, domainType: DomainType):
        """Initialize polynomial space

        Args:
            degree: Polynomial degree
            domain: Domain of the element
        """
        self._degree = degree
        self._domainType = domainType
        basis = self.canonical_basis()
        super().__init__(
            space_type=FunctionSpaceType.POLYNOMIAL,
            basis=basis,
            dimension=len(basis),
        )

    def canonical_basis(self) -> Basis:
        """Computes the canonical basis for the polynomial space"""
        power_tuples = []
        if self._domainType == DomainType.TRIANGLE:
            """
            The degree of the basis functions is equal to the sum of the powers of the leading term
            example: for degree 2, the basis functions are:
            e_0 = 1, e_1 = x, e_2 = y, e_3 = x^2, e_4 = xy, e_5 = y^2
            """
            for power in range(self._degree + 1):
                power_tuples += list(power_tuples_sum(2, power))
        elif self._domainType == DomainType.TETRAHEDRON:
            """
            Same as for triangle, but in 3D
            """
            for power in range(self._degree + 1):
                power_tuples += list(power_tuples_sum(3, power))
        elif self._domainType == DomainType.QUADRILATERAL:
            """
            The degree of the basis functions is equal to the max of the powers of the leading term
            example: for degree 2, the basis functions are:
            e_0 = 1, e_1 = x, e_2 = y, e_3 = xy, e_4 = x^2,
            e_5 = y^2, e_6 = x^2y, e_7 = xy^2, e_8 = x^2y^2
            """
            for power in range(self._degree + 1):
                power_tuples += list(power_tuples_max(2, power))
        elif self._domainType == DomainType.HEXAHEDRON:
            """
            Same as for quadrilateral, but in 3D
            """
            for power in range(self._degree + 1):
                power_tuples += list(power_tuples_max(3, power))
        else:
            raise NotImplementedError(
                f"Canonical basis not implemented for domain type {self._domainType}"
            )
        cano_basis = Basis([PolynomialBasisFunction([1], [p]) for p in power_tuples])
        return cano_basis

    def __repr__(self) -> str:
        """String representation of the polynomial space"""
        return f"PolynomialSpace(degree={self._degree}, dimension={self.dimension})"
