import numpy as np
from numpy.testing import assert_allclose, assert_array_equal
from pynsfem.elements.triangle.lagrange import ElementTriLagrange
from pynsfem.utils.power_tuples import power_tuples_sum


class TestElementTriLagrange:
    """Test suite for triangular Lagrange elements"""

    def test_element_creation(self):
        """Test basic element creation and properties"""
        # Create the element with linear basis on the reference triangle
        vertices = np.array([[0, 0], [1, 0], [0, 1]])
        lag = ElementTriLagrange(degree=1, points=vertices)

        # Check we have the correct number of shape functions for linear elements
        assert len(lag.shape_functions) == 3

        # Check the dimension is correct
        assert lag.dim == 2

        # Check the reference coordinates are stored correctly
        assert_array_equal(
            lag.reference_domain.reference_coordinates, vertices
        )

    def test_shape_function_count(self):
        """Test that the correct number of shape functions are created for different degrees"""
        # For triangle elements, the number of shape functions should follow the pattern
        # n = (degree + 1) * (degree + 2) / 2
        vertices = np.array([[0, 0], [1, 0], [0, 1]])

        for degree in range(1, 4):
            lag = ElementTriLagrange(degree=degree, points=vertices)
            expected_count = (degree + 1) * (degree + 2) // 2
            assert (
                len(lag.shape_functions) == expected_count
            ), f"For degree {degree}, expected {expected_count} shape functions"

    def test_partition_of_unity(self):
        """Test that shape functions form a partition of unity"""
        # Create elements of different degrees
        vertices = np.array([[0, 0], [1, 0], [0, 1]])

        for degree in [1, 2, 3]:
            lag = ElementTriLagrange(degree=degree, points=vertices)

            # Sample points inside the triangle
            sample_points = [
                np.array([0.25, 0.25]),
                np.array([0.5, 0.25]),
                np.array([0.33, 0.33]),
            ]

            for point in sample_points:
                # Sum of all shape functions at a point should be 1
                total = sum(shape_fn(point) for shape_fn in lag.shape_functions)
                assert_allclose(
                    total,
                    1.0,
                    rtol=1e-10,
                    err_msg=f"Sum of shape functions at {point} should be 1",
                )

    def test_default_triangle_creation(self):
        """Test that element can be created with default triangle"""
        # Create element without specifying points
        lag = ElementTriLagrange(degree=1)

        # Verify default triangle setup
        expected_points = np.array([[0, 0], [1, 0], [0, 1]])
        assert lag.reference_domain.reference_coordinates.shape == (3, 2)
        assert_array_equal(
            lag.reference_domain.reference_coordinates, expected_points
        )

    def test_lagrangian_property(self):
        """Test that shape functions have the Lagrangian property at corresponding degrees of freedom"""
        # Create elements of degree 2
        vertices = np.array([[0, 0], [1, 0], [0, 1]])
        degree = 2
        lag = ElementTriLagrange(degree=degree, points=vertices)

        # Calculate expected node positions for the quadratic triangle
        nodes = []

        # Nodes are contained in the nodal_basis
        for dof in lag.degrees_of_freedom:
            nodes.append(dof.point)
        nodes = np.array(nodes)

        # At each node, exactly one shape function should be 1 and all others should be 0
        for i, node in enumerate(nodes):
            for j, shape_fn in enumerate(lag.shape_functions):
                value = shape_fn(node)
                if i == j:
                    assert_allclose(
                        value,
                        1.0,
                        rtol=1e-10,
                        err_msg=f"Shape function {j} should be 1 at its corresponding node {i}",
                    )
                else:
                    assert_allclose(
                        value,
                        0.0,
                        rtol=1e-10,
                        err_msg=f"Shape function {j} should be 0 at node {i}",
                    )

    def test_equilateral_triangle_centered_at_origin(self):
        """Test that Lagrange elements work correctly on an equilateral triangle centered at origin"""
        # Create an equilateral triangle centered at the origin
        # With vertices at:
        #   (0, 1)
        #   (-sqrt(3)/2, -1/2)
        #   (sqrt(3)/2, -1/2)
        vertices = np.array([[0, 1], [-np.sqrt(3) / 2, -0.5], [np.sqrt(3) / 2, -0.5]])

        # Test with linear elements (degree=1)
        degree = 1
        lag = ElementTriLagrange(degree=degree, points=vertices)

        # Check that we have the right number of shape functions
        assert len(lag.shape_functions) == 3

        # Check the reference coordinates are stored correctly
        assert_array_equal(
            lag.reference_domain.reference_coordinates, vertices
        )

        # Instead of testing directly at vertices, obtain the actual DOF points from the element
        nodes = []
        for dof in lag.degrees_of_freedom:
            nodes.append(dof.point)
        nodes = np.array(nodes)

        # Verify Lagrangian property at DOF points
        for i, node in enumerate(nodes):
            for j, shape_fn in enumerate(lag.shape_functions):
                value = shape_fn(node)
                if i == j:
                    assert_allclose(
                        value,
                        1.0,
                        rtol=1e-10,
                        err_msg=f"Shape function {j} should be 1 at its DOF point {i}",
                    )
                else:
                    assert_allclose(
                        value,
                        0.0,
                        rtol=1e-10,
                        err_msg=f"Shape function {j} should be 0 at DOF point {i} when i≠j",
                    )

        # Test partition of unity at the centroid
        centroid = np.array([0.0, 0.0])
        total = sum(shape_fn(centroid) for shape_fn in lag.shape_functions)
        assert_allclose(
            total,
            1.0,
            rtol=1e-10,
            err_msg="Sum of shape functions at centroid should be 1",
        )

        # Test for quadratic element (degree=2)
        degree = 2
        lag = ElementTriLagrange(degree=degree, points=vertices)

        # Check we have the correct number of shape functions for quadratic triangle
        assert len(lag.shape_functions) == 6

        # Get the DOF points from the element
        nodes = []
        for dof in lag.degrees_of_freedom:
            nodes.append(dof.point)
        nodes = np.array(nodes)

        # Verify Lagrangian property at DOF points
        for i, node in enumerate(nodes):
            for j, shape_fn in enumerate(lag.shape_functions):
                value = shape_fn(node)
                if i == j:
                    assert_allclose(
                        value,
                        1.0,
                        rtol=1e-10,
                        err_msg=f"Shape function {j} should be 1 at its DOF point {i}",
                    )
                else:
                    assert_allclose(
                        value,
                        0.0,
                        rtol=1e-10,
                        atol=1e-15,
                        err_msg=f"Shape function {j} should be 0 at DOF point {i} when i≠j",
                    )
