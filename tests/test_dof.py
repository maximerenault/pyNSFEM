"""
Tests for the degrees of freedom (DoF) module
"""

import pytest
import numpy as np
from pynsfem.elements.components.dof import (
    DoFType,
    DegreeOfFreedom,
    PointEvaluation,
    DerivativeEvaluation,
    IntegralEvaluation,
)
from pynsfem.elements.components.basis import PolynomialBasisFunction
from pynsfem.elements.components.domain import Domain, TriangleDomain, DomainType


class MockBasisFunction:
    """Mock basis function for testing DoFs"""

    def __init__(self, eval_value=None, deriv_value=None, integrate_value=None):
        self.eval_value = eval_value or 1.0
        self.deriv_value = deriv_value or 1.0
        self.integrate_value = integrate_value or 1.0
        self.last_point = None
        self.last_direction = None
        self.deriv_called = False

    def __call__(self, point):
        self.last_point = point
        # If this is called after deriv(), return the derivative value
        if self.deriv_called:
            self.deriv_called = False
            return self.deriv_value
        return self.eval_value

    def deriv(self, direction):
        self.last_direction = direction
        self.deriv_called = True
        return self

    def integrate(self, domain):
        return self.integrate_value


class TestDoFTypes:
    """Test suite for DoF type enumeration"""

    def test_dof_types_exist(self):
        """Test that the DoF types are properly defined"""
        assert DoFType.POINT_EVALUATION is not None
        assert DoFType.DERIVATIVE_EVALUATION is not None
        assert DoFType.INTEGRAL_EVALUATION is not None


class TestPointEvaluation:
    """Test suite for the PointEvaluation DoF class"""

    def test_initialization(self):
        """Test initialization of PointEvaluation"""
        point = np.array([1.0, 2.0])
        dof = PointEvaluation(point)

        assert dof.dof_type == DoFType.POINT_EVALUATION
        assert np.array_equal(dof.point, point)

    def test_call_method(self):
        """Test the __call__ method of PointEvaluation"""
        point = np.array([1.0, 2.0])
        dof = PointEvaluation(point)

        # Test with mock basis function
        basis_func = MockBasisFunction(eval_value=3.5)
        result = dof(basis_func)

        assert result == 3.5
        assert np.array_equal(basis_func.last_point, point)

        # Test with a real polynomial basis function (constant function = 2)
        poly_func = PolynomialBasisFunction([2.0], [(0, 0)])
        result = dof(poly_func)

        assert result == 2.0


class TestDerivativeEvaluation:
    """Test suite for the DerivativeEvaluation DoF class"""

    def test_initialization(self):
        """Test initialization of DerivativeEvaluation"""
        point = np.array([1.0, 2.0])
        direction = np.array([0.5, 0.5])

        # Single direction
        dof1 = DerivativeEvaluation(point, direction)
        assert dof1.dof_type == DoFType.DERIVATIVE_EVALUATION
        assert np.array_equal(dof1.point, point)
        assert len(dof1.direction) == 1
        assert np.array_equal(dof1.direction[0], direction)
        assert dof1.n == 1

        # Multiple directions for higher derivatives
        directions = [np.array([1.0, 0.0]), np.array([0.0, 1.0])]
        dof2 = DerivativeEvaluation(point, directions, n_deriv=2)
        assert dof2.n == 2
        assert len(dof2.direction) == 2
        assert np.array_equal(dof2.direction[0], directions[0])
        assert np.array_equal(dof2.direction[1], directions[1])

    def test_invalid_n_deriv(self):
        """Test that initialization with n_deriv < 1 raises ValueError"""
        point = np.array([1.0, 2.0])
        direction = np.array([1.0, 0.0])

        with pytest.raises(ValueError, match="Derivative order must be at least 1"):
            DerivativeEvaluation(point, direction, n_deriv=0)

    def test_call_method(self):
        """Test the __call__ method of DerivativeEvaluation"""
        point = np.array([1.0, 2.0])
        direction = np.array([1.0, 0.0])  # x-direction

        dof = DerivativeEvaluation(point, direction)

        # Test with mock basis function
        basis_func = MockBasisFunction(deriv_value=2.5)
        result = dof(basis_func)

        assert result == 2.5
        assert np.array_equal(basis_func.last_direction, direction)
        assert np.array_equal(basis_func.last_point, point)

        # Test with a real polynomial basis function (f(x,y) = x^2 + y)
        poly_func = PolynomialBasisFunction([1.0, 1.0], [(2, 0), (0, 1)])
        # df/dx = 2x, so at (1,2) this should be 2
        dof_x = DerivativeEvaluation(point, np.array([1.0, 0.0]))
        result_x = dof_x(poly_func)
        assert pytest.approx(result_x) == 2.0

        # df/dy = 1, so at (1,2) this should be 1
        dof_y = DerivativeEvaluation(point, np.array([0.0, 1.0]))
        result_y = dof_y(poly_func)
        assert pytest.approx(result_y) == 1.0


class TestIntegralEvaluation:
    """Test suite for the IntegralEvaluation DoF class"""

    def test_initialization(self):
        """Test initialization of IntegralEvaluation"""
        domain = TriangleDomain(np.array([[0, 0], [1, 0], [0, 1]]), dim=2)
        dof = IntegralEvaluation(domain)

        assert dof.dof_type == DoFType.INTEGRAL_EVALUATION
        assert dof.domain == domain

    def test_call_method(self):
        """Test the __call__ method of IntegralEvaluation"""
        # Create a simple domain
        domain = TriangleDomain(np.array([[0, 0], [1, 0], [0, 1]]), dim=2)
        dof = IntegralEvaluation(domain)

        # Test with mock basis function
        basis_func = MockBasisFunction(integrate_value=0.5)
        result = dof(basis_func)

        assert result == 0.5


if __name__ == "__main__":
    pytest.main()
