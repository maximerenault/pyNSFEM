import numpy as np
import pytest
from pynsfem.utils.poly_basis import tri_poly_basis, tri_poly_basis_auto, power_pair

def test_power_pair():
    """Test power pair generation for different degrees"""
    # Test degree 0
    assert power_pair(0) == [(0, 0)]
    
    # Test degree 1
    assert power_pair(1) == [(1, 0), (0, 1)]
    
    # Test degree 2
    assert power_pair(2) == [(1, 1), (2, 0), (0, 2)]
    
    # Test degree 3
    assert power_pair(3) == [(2, 1), (1, 2), (3, 0), (0, 3)]

def test_tri_poly_basis_degree_0():
    """Test polynomial basis functions of degree 0"""
    func, grad, hess = tri_poly_basis(0)
    
    # Test function value
    assert func(0.0, 0.0) == (1,)
    
    # Test gradient
    assert grad(0.0, 0.0) == ([0, 0],)
    
    # Test hessian
    assert hess(0.0, 0.0) == ([0, 0, 0],)

def test_tri_poly_basis_degree_1():
    """Test polynomial basis functions of degree 1"""
    func, grad, hess = tri_poly_basis(1)
    
    # Test function values at vertices
    assert func(0.0, 0.0) == (1, 0, 0)
    assert func(1.0, 0.0) == (1, 1, 0)
    assert func(0.0, 1.0) == (1, 0, 1)
    
    # Test gradients
    assert grad(0.0, 0.0) == ([0, 0], [1, 0], [0, 1])
    
    # Test hessians
    assert hess(0.0, 0.0) == ([0, 0, 0], [0, 0, 0], [0, 0, 0])

def test_tri_poly_basis_degree_2():
    """Test polynomial basis functions of degree 2"""
    func, grad, hess = tri_poly_basis(2)
    
    # Test function values at vertices
    assert func(0.0, 0.0) == (1, 0, 0, 0, 0, 0)
    assert func(1.0, 0.0) == (1, 1, 0, 1, 0, 0)
    assert func(0.0, 1.0) == (1, 0, 1, 0, 0, 1)
    
    # Test gradients at origin
    assert grad(0.0, 0.0) == ([0, 0], [1, 0], [0, 1], [0, 0], [0, 0], [0, 0])
    
    # Test hessians at origin
    assert hess(0.0, 0.0) == ([0, 0, 0], [0, 0, 0], [0, 0, 0], [2, 0, 0], [0, 1, 0], [0, 0, 2])

def test_tri_poly_basis_auto():
    """Test automatic polynomial basis generation"""
    func, grad, hess = tri_poly_basis_auto(degree=2)
    
    # Test function values at vertices
    values = list(func(0.0, 0.0))
    assert len(values) == 6  # Number of basis functions for degree 2
    assert values[0] == 1.0  # First basis function should be 1
    
    # Test gradients
    gradients = list(grad(0.0, 0.0))
    assert len(gradients) == 6  # Number of basis functions for degree 2
    assert gradients[0] == (0, 0)  # First basis function gradient should be zero
    
    # Test hessians
    hessians = list(hess(0.0, 0.0))
    assert len(hessians) == 6  # Number of basis functions for degree 2
    assert hessians[0] == (0, 0, 0)  # First basis function hessian should be zero

def test_tri_poly_basis_consistency():
    """Test consistency between manual and automatic basis generation"""
    for degree in range(4):
        func1, grad1, hess1 = tri_poly_basis(degree)
        func2, grad2, hess2 = tri_poly_basis_auto(degree)
        
        # Test at some random points
        points = [(0.0, 0.0), (1.0, 0.0), (0.0, 1.0), (0.5, 0.5)]
        for x, y in points:
            # Compare function values
            values1 = list(func1(x, y))
            values2 = list(func2(x, y))
            assert len(values1) == len(values2)
            for v1, v2 in zip(values1, values2):
                assert np.isclose(v1, v2)
            
            # Compare gradients
            grads1 = list(grad1(x, y))
            grads2 = list(grad2(x, y))
            assert len(grads1) == len(grads2)
            for g1, g2 in zip(grads1, grads2):
                assert np.allclose(g1, g2)
            
            # Compare hessians
            hess1_list = list(hess1(x, y))
            hess2_list = list(hess2(x, y))
            assert len(hess1_list) == len(hess2_list)
            for h1, h2 in zip(hess1_list, hess2_list):
                assert np.allclose(h1, h2) 