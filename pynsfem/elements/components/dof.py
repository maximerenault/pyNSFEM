"""
Nodal basis (degrees of freedom) for finite elements
"""

from dataclasses import dataclass
from abc import ABC, abstractmethod
from typing import List, Tuple, Callable, Optional, Union
import numpy as np
from enum import Enum, auto
from .basis import BasisFunction
from .domain import Domain


class DoFType(Enum):
    """Types of degrees of freedom"""

    POINT_EVALUATION = auto()
    DERIVATIVE_EVALUATION = auto()
    INTEGRAL_EVALUATION = auto()


@dataclass
class DegreeOfFreedom(ABC):
    """Base class for degrees of freedom in finite elements

    We call here the degrees of freedom the nodal basis functions
    that will be used to define the finite element space.
    The degrees of freedom are defined in the reference element.
    """

    dof_type: DoFType

    @abstractmethod
    def __call__(self, basis_function: BasisFunction) -> float:
        """Evaluate the degree of freedom for a given basis function"""
        pass


class PointEvaluation(DegreeOfFreedom):
    """Point evaluation degree of freedom"""

    def __init__(self, point: np.ndarray):
        super().__init__(DoFType.POINT_EVALUATION)
        self.point = point

    def __call__(self, basis_function: BasisFunction) -> float:
        return basis_function(self.point)


class DerivativeEvaluation(DegreeOfFreedom):
    """Derivative evaluation degree of freedom"""

    def __init__(
        self,
        point: np.ndarray,
        direction: Union[np.ndarray, List[np.ndarray]],
        n_deriv: int = 1,
    ):
        super().__init__(DoFType.DERIVATIVE_EVALUATION)
        if n_deriv < 1:
            raise ValueError("Derivative order must be at least 1")
        if isinstance(direction[0], np.ndarray):
            assert len(direction) == n_deriv
        else:
            direction = [direction] * n_deriv
        self.point = point
        self.direction = direction
        self.n = n_deriv

    def __call__(self, basis_function: BasisFunction) -> float:
        bf = basis_function
        for i in range(self.n):
            bf = bf.deriv(self.direction[i])
        return bf(self.point)


class IntegralEvaluation(DegreeOfFreedom):
    """Integral evaluation degree of freedom"""

    def __init__(self, domain: Domain):
        super().__init__(DoFType.INTEGRAL_EVALUATION)
        self.domain = domain

    def __call__(self, basis_function: BasisFunction) -> float:
        return basis_function.integrate(self.domain)
