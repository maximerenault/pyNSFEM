import pytest
import numpy as np

@pytest.fixture
def random_points():
    """Generate random points for testing"""
    return np.random.rand(10, 2)

@pytest.fixture
def reference_triangle():
    """Reference triangle coordinates"""
    return np.array([[0.0, 0.0], [1.0, 0.0], [0.0, 1.0]])

@pytest.fixture
def reference_quad():
    """Reference quadrilateral coordinates"""
    return np.array([[-1.0, -1.0], [1.0, -1.0], [1.0, 1.0], [-1.0, 1.0]])

@pytest.fixture
def test_functions():
    """Test functions for projection"""
    return {
        'constant': lambda x, y: 1.0,
        'linear': lambda x, y: x + y,
        'quadratic': lambda x, y: x**2 + y**2,
        'cubic': lambda x, y: x**3 + y**3
    } 