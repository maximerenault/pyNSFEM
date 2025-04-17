"""
Gauss quadrature rules for numerical integration
"""

import numpy as np
from typing import Tuple, List


def tri_gauss_points_weights(order: int = 4) -> Tuple[np.ndarray, np.ndarray]:
    """
    Returns Gauss points and weights for triangular elements.
    Quadrature rules from https://people.sc.fsu.edu/~jburkardt/datasets/quadrature_rules_tri/quadrature_rules_tri.html.

    Args:
        order: Order of quadrature (1-6)
        
    Returns:
        Tuple of (points, weights)
    """
    if order == 1:
        # 1st order (1 point)
        points = np.array([[1/3, 1/3]])
        weights = np.array([1.0])
    elif order == 2:
        # 2nd order (3 points)
        points = np.array([
            [1/6, 1/6],
            [2/3, 1/6],
            [1/6, 2/3]
        ])
        weights = np.array([1/3, 1/3, 1/3])
    elif order == 3:
        # 3rd order (4 points)
        points = np.array([
            [1/3, 1/3],
            [1/5, 1/5],
            [3/5, 1/5],
            [1/5, 3/5]
        ])
        weights = np.array([-27/48, 25/48, 25/48, 25/48])
    elif order == 4:
        points = np.array(
            [
                (0.816847572980459, 0.091576213509771),
                (0.091576213509771, 0.816847572980459),
                (0.091576213509771, 0.091576213509771),
                (0.108103018168070, 0.445948490915965),
                (0.445948490915965, 0.108103018168070),
                (0.445948490915965, 0.445948490915965),
            ]
        )
        weights = np.array(
            [
                0.109951743655322,
                0.109951743655322,
                0.109951743655322,
                0.223381589678011,
                0.223381589678011,
                0.223381589678011,
            ]
        )
    elif order == 5:
        points = np.array(
            [
                (0.33333333333333333, 0.33333333333333333),
                (0.79742698535308720, 0.10128650732345633),
                (0.10128650732345633, 0.79742698535308720),
                (0.10128650732345633, 0.10128650732345633),
                (0.05971587178976981, 0.47014206410511505),
                (0.47014206410511505, 0.05971587178976981),
                (0.47014206410511505, 0.47014206410511505),
            ]
        )
        weights = np.array(
            [
                0.22500000000000000,
                0.12593918054482717,
                0.12593918054482717,
                0.12593918054482717,
                0.13239415278850616,
                0.13239415278850616,
                0.13239415278850616,
            ]
        )
    else:
        raise ValueError(f"Unsupported quadrature order: {order}")
    
    return points, weights


def rect_gauss_points_weights(order=2):
    """
    Returns Gauss points and weights for rectangular elements.

    Args:
        order (int): Order of quadrature (default: 2)

    Returns:
        tuple: (points, weights) where points is an array of shape (n_points, 2)
               and weights is an array of shape (n_points,)
    """
    if order == 2:
        # 2x2 Gauss quadrature rule
        points = np.array(
            [
                (-1 / np.sqrt(3), -1 / np.sqrt(3)),
                (1 / np.sqrt(3), -1 / np.sqrt(3)),
                (1 / np.sqrt(3), 1 / np.sqrt(3)),
                (-1 / np.sqrt(3), 1 / np.sqrt(3)),
            ]
        )
        weights = np.array([1.0, 1.0, 1.0, 1.0])
    else:
        raise ValueError(f"Order {order} quadrature rule not implemented")

    return points, weights


def quad_gauss_points_weights(order: int = 2) -> Tuple[np.ndarray, np.ndarray]:
    """
    Get Gauss quadrature points and weights for quadrilateral elements.
    
    Args:
        order: Order of quadrature (1-6)
        
    Returns:
        Tuple of (points, weights)
    """
    # Get 1D Gauss points and weights
    if order == 1:
        xi = np.array([0.0])
        w = np.array([2.0])
    elif order == 2:
        xi = np.array([-0.577350269189626, 0.577350269189626])
        w = np.array([1.0, 1.0])
    elif order == 3:
        xi = np.array([-0.774596669241483, 0.0, 0.774596669241483])
        w = np.array([0.555555555555556, 0.888888888888889, 0.555555555555556])
    elif order == 4:
        xi = np.array([-0.861136311594053, -0.339981043584856,
                      0.339981043584856, 0.861136311594053])
        w = np.array([0.347854845137454, 0.652145154862546,
                     0.652145154862546, 0.347854845137454])
    elif order == 5:
        xi = np.array([-0.906179845938664, -0.538469310105683, 0.0,
                      0.538469310105683, 0.906179845938664])
        w = np.array([0.236926885056189, 0.478628670499366, 0.568888888888889,
                     0.478628670499366, 0.236926885056189])
    elif order == 6:
        xi = np.array([-0.932469514203152, -0.661209386466265, -0.238619186083197,
                      0.238619186083197, 0.661209386466265, 0.932469514203152])
        w = np.array([0.171324492379170, 0.360761573048139, 0.467913934572691,
                     0.467913934572691, 0.360761573048139, 0.171324492379170])
    else:
        raise ValueError(f"Unsupported quadrature order: {order}")
    
    # Create 2D points and weights
    n_points = order * order
    points = np.zeros((n_points, 2))