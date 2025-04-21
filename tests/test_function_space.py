import pytest
import numpy as np
from pynsfem.elements.components.function_space import (
    FunctionSpaceType,
    PolynomialSpace,
)
from pynsfem.elements.components.domain import DomainType
from pynsfem.elements.components.basis import Basis
from pynsfem.utils.power_tuples import power_tuples_sum, power_tuples_max


class TestPolynomialSpace:
    """Test suite for the PolynomialSpace class"""

    def test_init(self):
        """Test initialization of polynomial spaces"""
        # Test degree 1 polynomial on triangle
        space = PolynomialSpace(degree=1, domainType=DomainType.TRIANGLE)
        assert space.space_type == FunctionSpaceType.POLYNOMIAL
        assert space.dimension == 3  # 1, x, y
        assert isinstance(space.basis, Basis)
        assert len(space.basis) == 3

        # Test degree 2 polynomial on triangle
        space = PolynomialSpace(degree=2, domainType=DomainType.TRIANGLE)
        assert space.space_type == FunctionSpaceType.POLYNOMIAL
        assert space.dimension == 6  # 1, x, y, x^2, xy, y^2
        assert len(space.basis) == 6

        # Test degree 1 polynomial on quadrilateral
        space = PolynomialSpace(degree=1, domainType=DomainType.QUADRILATERAL)
        assert space.space_type == FunctionSpaceType.POLYNOMIAL
        assert space.dimension == 4  # 1, x, y, xy
        assert len(space.basis) == 4

    def test_canonical_basis_triangle(self):
        """Test canonical basis generation for triangle domain"""
        # Test degree 1 polynomial on triangle
        space = PolynomialSpace(degree=1, domainType=DomainType.TRIANGLE)
        basis = space.canonical_basis()

        # Check the basis contains the correct functions
        assert len(basis) == 3

        # Create points for testing
        point1 = np.array([0.0, 0.0])
        point2 = np.array([1.0, 0.0])
        point3 = np.array([0.0, 1.0])

        # Find constant function (evaluates to 1 everywhere)
        constant_idx = -1
        for i, bf in enumerate(basis):
            if (
                abs(bf.evaluate(point1) - 1.0) < 1e-10
                and abs(bf.evaluate(point2) - 1.0) < 1e-10
            ):
                constant_idx = i
                break
        assert constant_idx != -1, "Constant basis function not found"

        # Find x function (evaluates to x coordinate)
        x_idx = -1
        for i, bf in enumerate(basis):
            if (
                abs(bf.evaluate(point2) - 1.0) < 1e-10
                and abs(bf.evaluate(point3)) < 1e-10
            ):
                x_idx = i
                break
        assert x_idx != -1, "x basis function not found"

        # Find y function (evaluates to y coordinate)
        y_idx = -1
        for i, bf in enumerate(basis):
            if (
                abs(bf.evaluate(point3) - 1.0) < 1e-10
                and abs(bf.evaluate(point2)) < 1e-10
            ):
                y_idx = i
                break
        assert y_idx != -1, "y basis function not found"

        # These should be different indices
        assert constant_idx != x_idx != y_idx

    def test_canonical_basis_quadrilateral(self):
        """Test canonical basis generation for quadrilateral domain"""
        # Test degree 1 polynomial on quadrilateral
        space = PolynomialSpace(degree=1, domainType=DomainType.QUADRILATERAL)
        basis = space.canonical_basis()

        # Check the basis contains the correct functions
        assert len(basis) == 4
        assert sum([sum(bf.coefficients) for bf in basis]) == len(basis)

        # Find the basis functions by examining their powers
        constant_found = False
        x_found = False
        y_found = False
        xy_found = False

        for bf in basis:
            power = bf.powers[0]  # Get the first (and only) power tuple
            if power == (0, 0):
                constant_found = True
            elif power == (1, 0):
                x_found = True
            elif power == (0, 1):
                y_found = True
            elif power == (1, 1):
                xy_found = True

        assert constant_found, "Constant basis function not found"
        assert x_found, "x basis function not found"
        assert y_found, "y basis function not found"
        assert xy_found, "xy basis function not found"

    def test_canonical_basis_tetrahedron(self):
        """Test canonical basis generation for tetrahedron domain"""
        # Test degree 1 polynomial on tetrahedron
        space = PolynomialSpace(degree=1, domainType=DomainType.TETRAHEDRON)
        basis = space.canonical_basis()

        # Check the basis contains the correct functions (1, x, y, z)
        assert len(basis) == 4
        assert sum([sum(bf.coefficients) for bf in basis]) == len(basis)

        # Find the basis functions by examining their powers
        constant_found = False
        x_found = False
        y_found = False
        z_found = False

        for bf in basis:
            power = bf.powers[0]  # Get the first (and only) power tuple
            if power == (0, 0, 0):
                constant_found = True
            elif power == (1, 0, 0):
                x_found = True
            elif power == (0, 1, 0):
                y_found = True
            elif power == (0, 0, 1):
                z_found = True

        assert constant_found, "Constant basis function not found"
        assert x_found, "x basis function not found"
        assert y_found, "y basis function not found"
        assert z_found, "z basis function not found"

    def test_canonical_basis_hexahedron(self):
        """Test canonical basis generation for hexahedron domain"""
        # Test degree 1 polynomial on hexahedron
        space = PolynomialSpace(degree=1, domainType=DomainType.HEXAHEDRON)
        basis = space.canonical_basis()

        # Check the basis contains the correct functions (1, x, y, z, xy, xz, yz, xyz)
        assert len(basis) == 8
        assert sum([sum(bf.coefficients) for bf in basis]) == len(basis)

        # Check that the expected powers are present
        expected_powers = [
            (0, 0, 0),  # 1
            (1, 0, 0),  # x
            (0, 1, 0),  # y
            (0, 0, 1),  # z
            (1, 1, 0),  # xy
            (1, 0, 1),  # xz
            (0, 1, 1),  # yz
            (1, 1, 1),  # xyz
        ]

        actual_powers = [bf.powers[0] for bf in basis]

        for power in expected_powers:
            assert power in actual_powers, f"Power {power} not found in basis"

    def test_unsupported_domain(self):
        """Test error handling for unsupported domain types"""
        with pytest.raises(NotImplementedError):
            PolynomialSpace(degree=1, domainType=None)

    def test_repr(self):
        """Test string representation of polynomial space"""
        space = PolynomialSpace(degree=2, domainType=DomainType.TRIANGLE)
        repr_str = repr(space)
        assert "PolynomialSpace" in repr_str
        assert "dimension=6" in repr_str

    def test_dimension_matches_power_tuples(self):
        """Test that dimension matches the expected number from power_tuples functions"""
        # Triangle - should match power_tuples_sum
        for degree in range(4):
            space = PolynomialSpace(degree=degree, domainType=DomainType.TRIANGLE)
            expected_count = sum(
                1 for p in range(degree + 1) for _ in power_tuples_sum(2, p)
            )
            assert space.dimension == expected_count

        # Quadrilateral - should match power_tuples_max
        for degree in range(4):
            space = PolynomialSpace(degree=degree, domainType=DomainType.QUADRILATERAL)
            expected_count = sum(
                1 for p in range(degree + 1) for _ in power_tuples_max(2, p)
            )
            assert space.dimension == expected_count

    def test_canonical_basis_powers(self):
        """Test that the canonical basis powers match the expected patterns"""
        # Triangle domain with degree 2
        space = PolynomialSpace(degree=2, domainType=DomainType.TRIANGLE)
        basis = space.canonical_basis()

        # Expected powers for triangle degree 2: (0,0), (1,0), (0,1), (2,0), (1,1), (0,2)
        expected_powers = [(0, 0), (1, 0), (0, 1), (2, 0), (1, 1), (0, 2)]
        actual_powers = [bf.powers[0] for bf in basis]

        # Check that all expected powers are present (regardless of order)
        for power in expected_powers:
            assert power in actual_powers

        # Quadrilateral domain with degree 1
        space = PolynomialSpace(degree=1, domainType=DomainType.QUADRILATERAL)
        basis = space.canonical_basis()

        # Expected powers for quadrilateral degree 1: (0,0), (1,0), (0,1), (1,1)
        expected_powers = [(0, 0), (1, 0), (0, 1), (1, 1)]
        actual_powers = [bf.powers[0] for bf in basis]

        # Check that all expected powers are present (regardless of order)
        for power in expected_powers:
            assert power in actual_powers
