import numpy as np
import pytest
from pynsfem.elements import ElementTriQuinticArgyris, ElementTriCubicHermite, ElementQuadQ4

# Test data
@pytest.fixture
def tri_vertices():
    return np.array([[0.0, 0.0], [1.0, 0.0], [0.0, 1.0]])

@pytest.fixture
def quad_vertices():
    return np.array([[-1.0, -1.0], [1.0, -1.0], [1.0, 1.0], [-1.0, 1.0]])

@pytest.fixture
def tri_faces():
    return np.array([[0, 1, 2]])

@pytest.fixture
def quad_faces():
    return np.array([[0, 1, 2, 3]])

def test_tri_quintic_argyris_shape_functions(tri_vertices, tri_faces):
    element = ElementTriQuinticArgyris()
    Elements, _ = element.define_elements(tri_vertices, tri_faces)
    
    # Test shape functions at each DOF (should be 1 at the DOF and 0 at the others)
    for i, shape_func in enumerate(element.shape_functions):
        for j, dof in enumerate(element.degrees_of_freedom):
            if j == i:
                assert np.isclose(shape_func(dof), 1.0)
            else:
                assert np.isclose(shape_func(dof), 0.0)

def test_tri_cubic_hermite_shape_functions(tri_vertices, tri_faces):
    element = ElementTriCubicHermite()
    Elements, _ = element.define_elements(tri_vertices, tri_faces)
    
    # Test shape functions at vertices
    for i in range(3):
        x, y = tri_vertices[i]
        for j in range(element.num_dof):
            if j == i:
                assert np.isclose(element.shape_functions[j](x, y), 1.0)
            else:
                assert np.isclose(element.shape_functions[j](x, y), 0.0)

def test_quad_q4_shape_functions(quad_vertices, quad_faces):
    element = ElementQuadQ4()
    Elements, _ = element.define_elements(quad_vertices, quad_faces)
    
    # Test shape functions at vertices
    for i in range(4):
        x, y = quad_vertices[i]
        for j in range(element.num_dof):
            if j == i:
                assert np.isclose(element.shape_functions[j](x, y), 1.0)
            else:
                assert np.isclose(element.shape_functions[j](x, y), 0.0)

def test_tri_quintic_argyris_derivatives(tri_vertices, tri_faces):
    element = ElementTriQuinticArgyris()
    Elements, _ = element.define_elements(tri_vertices, tri_faces)
    
    # Test derivatives at center point
    x, y = 1/3, 1/3
    for i in range(element.num_dof):
        # Sum of derivatives should be zero
        assert np.isclose(sum(dN(x, y) for dN in element.shape_function_derivatives[0]), 0.0)
        assert np.isclose(sum(dN(x, y) for dN in element.shape_function_derivatives[1]), 0.0)

def test_tri_cubic_hermite_derivatives(tri_vertices, tri_faces):
    element = ElementTriCubicHermite()
    Elements, _ = element.define_elements(tri_vertices, tri_faces)
    
    # Test derivatives at center point
    x, y = 1/3, 1/3
    for i in range(element.num_dof):
        # Sum of derivatives should be zero
        assert np.isclose(sum(dN(x, y) for dN in element.shape_function_derivatives[0]), 0.0)
        assert np.isclose(sum(dN(x, y) for dN in element.shape_function_derivatives[1]), 0.0)

def test_quad_q4_derivatives(quad_vertices, quad_faces):
    element = ElementQuadQ4()
    Elements, _ = element.define_elements(quad_vertices, quad_faces)
    
    # Test derivatives at center point
    x, y = 0.0, 0.0
    for i in range(element.num_dof):
        # Sum of derivatives should be zero
        assert np.isclose(sum(dN(x, y) for dN in element.shape_function_derivatives[0]), 0.0)
        assert np.isclose(sum(dN(x, y) for dN in element.shape_function_derivatives[1]), 0.0)

def test_tri_quintic_argyris_projection(tri_vertices, tri_faces):
    element = ElementTriQuinticArgyris()
    Elements, _ = element.define_elements(tri_vertices, tri_faces)
    
    # Test projection of constant function
    def constant_func(x, y):
        return 1.0
    
    projection = element.project_function(tri_vertices, tri_faces, Elements, constant_func)
    assert np.allclose(projection, 1.0)

def test_tri_cubic_hermite_projection(tri_vertices, tri_faces):
    element = ElementTriCubicHermite()
    Elements, _ = element.define_elements(tri_vertices, tri_faces)
    
    # Test projection of constant function
    def constant_func(x, y):
        return 1.0
    
    projection = element.project_function(tri_vertices, tri_faces, Elements, constant_func)
    assert np.allclose(projection, 1.0)

def test_quad_q4_projection(quad_vertices, quad_faces):
    element = ElementQuadQ4()
    Elements, _ = element.define_elements(quad_vertices, quad_faces)
    
    # Test projection of constant function
    def constant_func(x, y):
        return 1.0
    
    projection = element.project_function(quad_vertices, quad_faces, Elements, constant_func)
    assert np.allclose(projection, 1.0)

def test_tri_quintic_argyris_linear_projection(tri_vertices, tri_faces):
    element = ElementTriQuinticArgyris()
    Elements, _ = element.define_elements(tri_vertices, tri_faces)
    
    # Test projection of linear function
    def linear_func(x, y):
        return x + y
    
    projection = element.project_function(tri_vertices, tri_faces, Elements, linear_func)
    # Check values at vertices
    for i in range(3):
        x, y = tri_vertices[i]
        assert np.isclose(projection[i], linear_func(x, y))

def test_tri_cubic_hermite_linear_projection(tri_vertices, tri_faces):
    element = ElementTriCubicHermite()
    Elements, _ = element.define_elements(tri_vertices, tri_faces)
    
    # Test projection of linear function
    def linear_func(x, y):
        return x + y
    
    projection = element.project_function(tri_vertices, tri_faces, Elements, linear_func)
    # Check values at vertices
    for i in range(3):
        x, y = tri_vertices[i]
        assert np.isclose(projection[i], linear_func(x, y))

def test_quad_q4_linear_projection(quad_vertices, quad_faces):
    element = ElementQuadQ4()
    Elements, _ = element.define_elements(quad_vertices, quad_faces)
    
    # Test projection of linear function
    def linear_func(x, y):
        return x + y
    
    projection = element.project_function(quad_vertices, quad_faces, Elements, linear_func)
    # Check values at vertices
    for i in range(4):
        x, y = quad_vertices[i]
        assert np.isclose(projection[i], linear_func(x, y)) 