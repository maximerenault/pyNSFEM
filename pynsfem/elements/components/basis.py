"""
Basis function definitions for finite elements
"""

from typing import List, Tuple, Union
from abc import ABC, abstractmethod
from enum import Enum, auto
from ...utils.decorators import memoize
import numpy as np

Symbols = ["x", "y", "z"]

class BasisFunctionType(Enum):
    """Types of basis functions"""

    POLYNOMIAL = auto()
    TRIGONOMETRIC = auto()
    EXPONENTIAL = auto()
    NURBS = auto()


class Domain(ABC):
    """Base domain class for integration purposes"""

    def __init__(self):
        pass

    @abstractmethod
    def measure(self) -> float:
        """Return the measure of the domain"""
        pass


class BasisFunction(ABC):
    """Container for basis function"""

    def __init__(self):
        pass

    @abstractmethod
    def evaluate(self, point: np.ndarray) -> float:
        """Evaluate the basis function at a given point"""
        pass

    @abstractmethod
    def deriv(self, vector: np.ndarray) -> "BasisFunction":
        """Return derivative of the basis function"""
        pass

    @abstractmethod
    def integrate(self, domain: Domain) -> float:
        """Return the integral of the basis function over a given domain"""
        pass

    def __call__(self, point: np.ndarray) -> float:
        """Evaluate the basis function at a given point"""
        return self.evaluate(point)

    @abstractmethod
    def __add__(self, other: "BasisFunction") -> "BasisFunction":
        """Add two basis functions"""
        return NotImplemented

    @abstractmethod
    def __mul__(self, other: float) -> "BasisFunction":
        """Multiply a basis function by a scalar"""
        return NotImplemented

    @abstractmethod
    def __rmul__(self, other: float) -> "BasisFunction":
        """Multiply a basis function by a scalar"""
        return NotImplemented

    @abstractmethod
    def __sub__(self, other: "BasisFunction") -> "BasisFunction":
        """Subtract two basis functions"""
        return NotImplemented


class Basis(List[BasisFunction]):
    """Container for a list of basis functions"""

    def __init__(self, *args: List[BasisFunction]):
        """Initialize the basis with a list of basis functions"""
        super().__init__(*args)
        self.__array_priority__ = 100  # Higher priority for numpy operations

    def __matmul__(
        self, other: np.ndarray
    ) -> Union[BasisFunction, "Basis", List["Basis"]]:
        """Matrix multiplication operator (@)

        This allows for operations like matrix-vector multiplication with scalars.

        Args:
            other: a vector or matrix (numpy array) to multiply with

        Returns:
            If len(shape) == 1: a linear combination of basis functions (dot product)
            If len(shape) == 2: a new basis
            If len(shape) >= 3: a vector/matrix of basis
        """
        if isinstance(other, np.ndarray):
            if other.shape[-1] != len(self):
                raise ValueError(
                    f"Dimension mismatch: {len(self)} functions but {len(other)} coefficients"
                )
            if len(other.shape) == 1:
                result = None
                for i, func in enumerate(self):
                    term = func * other[i]
                    if result is None:
                        result = term
                    else:
                        result = result + term
                return result
            elif len(other.shape) == 2:
                return Basis([self @ vec for vec in other.T])
            else:
                return [self @ mat for mat in other]
        else:
            raise TypeError(f"Unsupported operand type for @: {type(other)}")

    def __rmatmul__(self, other: np.ndarray) -> BasisFunction:
        """Right matrix multiplication operator (@)
        See __matmul__.
        """
        if isinstance(other, np.ndarray):
            if len(other) != len(self):
                raise ValueError(
                    f"Dimension mismatch: {len(self)} functions but {len(other)} coefficients"
                )
            if len(other.shape) == 2:
                return Basis([self @ vec for vec in other])
            else:
                return self @ other
        else:
            raise TypeError(f"Unsupported operand type for @: {type(other)}")


class PolynomialBasisFunction(BasisFunction):

    def __init__(self, coefficients: List[float], powers: List[Tuple[int, ...]]):
        """Initialize polynomial basis function"""
        assert len(coefficients) == len(
            powers
        ), "Coefficients and powers must have the same length"
        assert all(isinstance(p, tuple) for p in powers), "Powers must be tuples"
        assert all(
            all(isinstance(i, int) for i in p) for p in powers
        ), "Powers must be integers"
        assert all(
            all(i >= 0 for i in p) for p in powers
        ), "Powers must be non-negative"
        assert all(
            len(p) == len(powers[0]) for p in powers
        ), "All powers must have the same length"

        if len(powers) == 0:
            self.degree = 0
            self.coefficients = []
            self.powers = []
        else:
            self.degree = max(sum(p) for p in powers)
            self.coefficients = coefficients
            self.powers = powers
            self.sort()
            self.add_neighbors()
            self.delete_zeros()
        self._type = BasisFunctionType.POLYNOMIAL

    def __add__(self, other: "PolynomialBasisFunction") -> "PolynomialBasisFunction":
        """Add two polynomial basis functions"""
        return PolynomialBasisFunction(
            self.coefficients + other.coefficients,
            self.powers + other.powers,
        )

    def __mul__(
        self, other: Union[float, int, "PolynomialBasisFunction"]
    ) -> "PolynomialBasisFunction":
        """Multiply two polynomial basis functions"""
        if isinstance(other, (float, int)):
            new_coefficients = [coeff * other for coeff in self.coefficients]
            return PolynomialBasisFunction(new_coefficients, self.powers)
        elif isinstance(other, PolynomialBasisFunction):
            new_coefficients = []
            new_powers = []
            for coeff1, power1 in self:
                for coeff2, power2 in other:
                    new_coefficients.append(coeff1 * coeff2)
                    new_powers.append(tuple(p1 + p2 for p1, p2 in zip(power1, power2)))
            return PolynomialBasisFunction(new_coefficients, new_powers)
        else:
            raise TypeError("Unsupported type for multiplication")

    def __rmul__(self, other: float) -> "PolynomialBasisFunction":
        """Right multiply a polynomial by a scalar"""
        return self.__mul__(other)

    def __sub__(self, other: "PolynomialBasisFunction") -> "PolynomialBasisFunction":
        """Subtract two polynomial basis functions"""
        return self + (other * (-1))

    def __getitem__(self, index: int) -> Tuple[float, Tuple[int, ...]]:
        """Get the coefficient and power at a given index"""
        return self.coefficients[index], self.powers[index]

    def __len__(self) -> int:
        """Get the number of terms in the polynomial"""
        return len(self.coefficients)

    def __iter__(self):
        """Iterate over the polynomial terms"""
        return zip(self.coefficients, self.powers)

    def __repr__(self) -> str:
        """String representation of the polynomial"""
        terms = []
        for coeff, power in self:
            term = f"{coeff} * "
            for i, p in enumerate(power):
                if p == 0:
                    continue
                repetitions = i//len(Symbols) + 1
                real_i = i%len(Symbols)
                if p == 1:
                    term += repetitions*Symbols[real_i] + " "
                elif p > 0:
                    term += repetitions*Symbols[real_i] + f"^{p} "
            terms.append(term)
        return " + ".join(terms)

    @property
    @memoize
    def gradient(self) -> List["PolynomialBasisFunction"]:
        """Return the gradient of the polynomial"""
        gradient = Basis()
        for i in range(len(self.powers[0])):
            new_coefficients = []
            new_powers = []
            for coeff, power in self:
                if power[i] > 0:
                    new_coefficients.append(coeff * power[i])
                    new_powers.append(
                        tuple(p - 1 if j == i else p for j, p in enumerate(power))
                    )
            gradient.append(PolynomialBasisFunction(new_coefficients, new_powers))
        return gradient

    def deriv(self, vector: np.ndarray) -> "PolynomialBasisFunction":
        """Return derivative of the polynomial along a given vector"""
        assert len(vector) == len(self.powers[0]), "Vector dimension mismatch"
        return self.gradient @ vector

    def evaluate(self, point: np.ndarray) -> float:
        """Evaluate the polynomial at a given point"""
        assert len(point) == len(self.powers[0]), "Point dimension mismatch"
        result = 0.0
        for coeff, power in self:
            result += coeff * np.prod([point[i] ** p for i, p in enumerate(power)])
        return result

    def integrate(self, domain: Domain) -> "PolynomialBasisFunction":
        """Return the integral of the polynomial over a given domain"""
        Warning("Integration not implemented")
        return 0

    def sort(self):
        """Sort the polynomial in lexicographical order"""
        self.powers, self.coefficients = zip(
            *sorted(zip(self.powers, self.coefficients))
        )
        self.powers = list(self.powers)
        self.coefficients = list(self.coefficients)

    def add_neighbors(self):
        n = len(self)
        for i in range(n - 1):
            term1 = self[i]
            term2 = self[i + 1]
            if term1[1] == term2[1]:
                self.coefficients[i + 1] += self.coefficients[i]
                self.coefficients[i] = 0

    def delete_zeros(self):
        to_del = []
        for i, coeff in enumerate(self.coefficients):
            if coeff == 0:
                to_del.append(i)
        for i in to_del[::-1]:
            self.coefficients.pop(i)
            self.powers.pop(i)
