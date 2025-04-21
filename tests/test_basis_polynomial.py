import numpy as np
import pytest
from pynsfem.elements.components.basis import (
    PolynomialBasisFunction as Polynomial,
    BasisFunctionType,
)


class TestPolynomial:
    """Test suite for the Polynomial basis function class"""

    def test_init(self):
        """Test initialization of polynomial basis functions"""
        # Test constant polynomial: 3
        p = Polynomial([3], [(0, 0)])
        assert p.degree == 0
        assert len(p.coefficients) == 1
        assert len(p.powers) == 1
        assert p._type == BasisFunctionType.POLYNOMIAL

        # Test linear polynomial: 2x + 3y
        p = Polynomial([2, 3], [(1, 0), (0, 1)])
        assert p.degree == 1
        # Coefficients and powers get sorted, so order might change
        assert sorted(p.coefficients) == [2, 3]
        assert (1, 0) in p.powers
        assert (0, 1) in p.powers

        # Test quadratic polynomial: 5x^2 + 2xy + 3y^2
        p = Polynomial([5, 2, 3], [(2, 0), (1, 1), (0, 2)])
        assert p.degree == 2
        assert sorted(p.coefficients) == [2, 3, 5]
        assert (2, 0) in p.powers
        assert (1, 1) in p.powers
        assert (0, 2) in p.powers

        # Test invalid inputs
        with pytest.raises(AssertionError):
            # Different lengths of coefficients and powers
            Polynomial([1, 2], [(0, 0)])

        with pytest.raises(AssertionError):
            # Powers not tuples
            Polynomial([1], [1])

        with pytest.raises(AssertionError):
            # Powers not integers
            Polynomial([1], [(1.5, 0)])

        with pytest.raises(AssertionError):
            # Negative powers
            Polynomial([1], [(-1, 0)])

    def test_evaluate(self):
        """Test evaluation of polynomial basis functions"""
        # Constant polynomial: 5
        p = Polynomial([5], [(0, 0)])
        assert p.evaluate(np.array([1.0, 2.0])) == 5
        assert p(np.array([1.0, 2.0])) == 5  # Test __call__ method

        # Linear polynomial: 2x + 3y
        p = Polynomial([2, 3], [(1, 0), (0, 1)])
        assert p.evaluate(np.array([1.0, 2.0])) == 2 * 1.0 + 3 * 2.0
        assert p(np.array([1.0, 2.0])) == 2 * 1.0 + 3 * 2.0

        # Quadratic polynomial: 5x^2 + 2xy + 3y^2
        p = Polynomial([5, 2, 3], [(2, 0), (1, 1), (0, 2)])
        expected = 5 * (1.0**2) + 2 * (1.0 * 2.0) + 3 * (2.0**2)
        assert p.evaluate(np.array([1.0, 2.0])) == expected
        assert p(np.array([1.0, 2.0])) == expected

    def test_add(self):
        """Test addition of polynomial basis functions"""
        # (5) + (2x + 3y) = 5 + 2x + 3y
        p1 = Polynomial([5], [(0, 0)])
        p2 = Polynomial([2, 3], [(1, 0), (0, 1)])
        p_sum = p1 + p2

        # Test the result at a point
        point = np.array([1.0, 2.0])
        expected = p1(point) + p2(point)
        assert p_sum(point) == expected

        # (2x + 3y) + (4x + 5y) = 6x + 8y
        p1 = Polynomial([2, 3], [(1, 0), (0, 1)])
        p2 = Polynomial([4, 5], [(1, 0), (0, 1)])
        p_sum = p1 + p2

        # Test the result at a point
        point = np.array([1.0, 2.0])
        expected = p1(point) + p2(point)
        assert p_sum(point) == expected

    def test_multiply_scalar(self):
        """Test multiplication by scalar"""
        # 3 * (2x + 3y) = 6x + 9y
        p = Polynomial([2, 3], [(1, 0), (0, 1)])
        p_mult = p * 3

        # Test the result at a point
        point = np.array([1.0, 2.0])
        expected = 3 * p(point)
        assert p_mult(point) == expected

        # Test right multiplication
        p_rmult = 3 * p
        assert p_rmult(point) == expected

    def test_multiply_polynomial(self):
        """Test multiplication of polynomials"""
        # (2x + 3y) * (4x + 5y) = 8x^2 + 10xy + 12xy + 15y^2 = 8x^2 + 22xy + 15y^2
        p1 = Polynomial([2, 3], [(1, 0), (0, 1)])
        p2 = Polynomial([4, 5], [(1, 0), (0, 1)])
        p_mult = p1 * p2

        # Test the result at a point
        point = np.array([1.0, 2.0])
        expected = p1(point) * p2(point)
        assert p_mult(point) == expected

    def test_subtract(self):
        """Test subtraction of polynomials"""
        # (5x^2 + 2xy + 3y^2) - (2x^2 + xy + y^2) = 3x^2 + xy + 2y^2
        p1 = Polynomial([5, 2, 3], [(2, 0), (1, 1), (0, 2)])
        p2 = Polynomial([2, 1, 1], [(2, 0), (1, 1), (0, 2)])
        p_diff = p1 - p2

        # Test the result at a point
        point = np.array([1.0, 2.0])
        expected = p1(point) - p2(point)
        assert p_diff(point) == expected

    def test_gradient(self):
        """Test gradient of polynomials"""
        # Gradient of constant: 5 -> [0, 0]
        p = Polynomial([5], [(0, 0)])
        grad = p.gradient
        assert len(grad) == 2  # 2D polynomial
        for g in grad:
            assert len(g) == 0  # Each component has no terms

        # Gradient of linear: 2x + 3y
        # ∇(2x + 3y) = [2, 3]
        p = Polynomial([2, 3], [(1, 0), (0, 1)])
        grad = p.gradient
        assert len(grad) == 2  # 2D gradient
        assert grad[0].evaluate(np.array([1.0, 2.0])) == 2  # ∂/∂x = 2
        assert grad[1].evaluate(np.array([1.0, 2.0])) == 3  # ∂/∂y = 3

        # Gradient of quadratic: 5x^2 + 2xy + 3y^2
        # ∇(5x^2 + 2xy + 3y^2) = [10x + 2y, 2x + 6y]
        p = Polynomial([5, 2, 3], [(2, 0), (1, 1), (0, 2)])
        grad = p.gradient

        # Verify ∂/∂x at (1,2)
        x_deriv = grad[0]
        expected_x_deriv = 10 * 1.0 + 2 * 2.0  # 10x + 2y = 10(1) + 2(2) = 14
        assert x_deriv.evaluate(np.array([1.0, 2.0])) == expected_x_deriv

        # Verify ∂/∂y at (1,2)
        y_deriv = grad[1]
        expected_y_deriv = 2 * 1.0 + 6 * 2.0  # 2x + 6y = 2(1) + 6(2) = 14
        assert y_deriv.evaluate(np.array([1.0, 2.0])) == expected_y_deriv

    def test_directional_derivative(self):
        """Test directional derivative of polynomials"""
        # Directional derivative of constant is zero
        p = Polynomial([5], [(0, 0)])
        direction = np.array([1.0, 0.0])  # x-direction
        p_deriv = p.deriv(direction)
        assert len(p_deriv) == 0  # No terms in derivative

        # Directional derivative of 2x + 3y in x-direction
        p = Polynomial([2, 3], [(1, 0), (0, 1)])

        # In x-direction [1,0]
        x_direction = np.array([1.0, 0.0])
        p_deriv_x = p.deriv(x_direction)
        assert p_deriv_x.evaluate(np.array([1.0, 2.0])) == 2  # d/dx(2x + 3y) = 2

        # In y-direction [0,1]
        y_direction = np.array([0.0, 1.0])
        p_deriv_y = p.deriv(y_direction)
        assert p_deriv_y.evaluate(np.array([1.0, 2.0])) == 3  # d/dy(2x + 3y) = 3

        # In [1,1] direction (normalized to unit vector would be [1/√2, 1/√2])
        diag_direction = np.array([1.0, 1.0])
        p_deriv_diag = p.deriv(diag_direction)
        # d/d[1,1](2x + 3y) = 2*1 + 3*1 = 5
        assert p_deriv_diag.evaluate(np.array([1.0, 2.0])) == 5

        # Directional derivative of 5x^2 + 2xy + 3y^2
        p = Polynomial([5, 2, 3], [(2, 0), (1, 1), (0, 2)])

        # In x-direction [1,0]
        x_direction = np.array([1.0, 0.0])
        p_deriv_x = p.deriv(x_direction)
        # d/dx(5x^2 + 2xy + 3y^2) = 10x + 2y = 10(1) + 2(2) = 14
        assert p_deriv_x.evaluate(np.array([1.0, 2.0])) == 14

        # In y-direction [0,1]
        y_direction = np.array([0.0, 1.0])
        p_deriv_y = p.deriv(y_direction)
        # d/dy(5x^2 + 2xy + 3y^2) = 2x + 6y = 2(1) + 6(2) = 14
        assert p_deriv_y.evaluate(np.array([1.0, 2.0])) == 14

        # In arbitrary direction [3,4] (not normalized)
        arbitrary_direction = np.array([3.0, 4.0])
        p_deriv_arb = p.deriv(arbitrary_direction)
        # d/d[3,4](5x^2 + 2xy + 3y^2) = 3*(10x + 2y) + 4*(2x + 6y)
        # = 30x + 6y + 8x + 24y = 38x + 30y
        # at (1,2): 38*1 + 30*2 = 38 + 60 = 98
        assert p_deriv_arb.evaluate(np.array([1.0, 2.0])) == 98

    def test_sort(self):
        """Test sorting of polynomial terms"""
        # Create polynomial with unsorted terms
        p = Polynomial([3, 2, 5], [(0, 1), (1, 0), (0, 0)])

        # After initialization, terms should be sorted (lexicographically)
        # This should be [(0, 0), (0, 1), (1, 0)]
        assert p.powers[0] == (0, 0)
        assert p.powers[1] == (0, 1)
        assert p.powers[2] == (1, 0)

    def test_add_neighbors(self):
        """Test adding like terms in a polynomial"""
        # Create polynomial with like terms
        p = Polynomial([2, 3], [(1, 0), (1, 0)])

        # After initialization, like terms should be combined
        assert len(p) == 1
        assert p.coefficients[0] == 5
        assert p.powers[0] == (1, 0)

        # Another test with mixed terms
        p = Polynomial([2, 3, 4], [(1, 0), (1, 0), (0, 1)])
        assert len(p) == 2
        # Find the 1,0 term and check it has coefficient 5
        for coeff, power in zip(p.coefficients, p.powers):
            if power == (1, 0):
                assert coeff == 5
            elif power == (0, 1):
                assert coeff == 4

    def test_delete_zeros(self):
        """Test deletion of zero terms"""
        # Create polynomial with a zero term
        p = Polynomial([0, 3], [(1, 0), (0, 1)])

        # After initialization, zero terms should be removed
        assert len(p) == 1
        assert p.coefficients[0] == 3
        assert p.powers[0] == (0, 1)

        # Test with all zero terms
        p = Polynomial([0, 0], [(1, 0), (0, 1)])
        assert len(p) == 0

    def test_getitem(self):
        """Test __getitem__ method"""
        p = Polynomial([2, 3], [(1, 0), (0, 1)])

        # Due to sorting, we need to figure out the actual ordering
        terms = [(c, pw) for c, pw in zip(p.coefficients, p.powers)]
        assert (2, (1, 0)) in terms
        assert (3, (0, 1)) in terms

    def test_len(self):
        """Test __len__ method"""
        p = Polynomial([2, 3, 4], [(1, 0), (0, 1), (2, 0)])
        assert len(p) == 3

        # After removing a term
        p = Polynomial([0, 3, 4], [(1, 0), (0, 1), (2, 0)])
        assert len(p) == 2

    def test_iter(self):
        """Test iteration over polynomial terms"""
        p = Polynomial([2, 3], [(1, 0), (0, 1)])
        terms = [(c, pw) for c, pw in p]
        assert (2, (1, 0)) in terms
        assert (3, (0, 1)) in terms
