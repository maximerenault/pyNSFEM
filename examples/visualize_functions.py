"""
Example script for visualizing basis functions using BasisFunctionVisualizer

This example demonstrates how to use the BasisFunctionVisualizer class
to create 3D visualizations of polynomial basis functions.
"""

import numpy as np
from pynsfem.elements.components.basis import PolynomialBasisFunction
from pynsfem.elements.components.domain import TriangleDomain, QuadrilateralDomain
from pynsfem.visualization.basis_visualizer import BasisFunctionVisualizer


def main():
    """Main function to demonstrate basis visualization"""

    # Create a visualizer instance
    visualizer = BasisFunctionVisualizer(cmap="plasma")

    print("Creating example basis functions...")

    # Create some example polynomial basis functions

    # 1. Simple linear function: f(x,y) = x
    linear_x = PolynomialBasisFunction([1.0], [(1, 0)])

    # 2. Linear function: f(x,y) = y
    linear_y = PolynomialBasisFunction([1.0], [(0, 1)])

    # 3. Quadratic function: f(x,y) = x^2 + y^2
    quadratic = PolynomialBasisFunction([1.0, 1.0], [(2, 0), (0, 2)])

    # 4. Cubic bubble function: f(x,y) = x(1-x)y(1-y)
    # Expanding: xy - x^2y - xy^2 + x^2y^2
    bubble = PolynomialBasisFunction(
        [1.0, -1.0, -1.0, 1.0], [(1, 1), (2, 1), (1, 2), (2, 2)]
    )

    # Create domain objects for visualization
    # Standard triangle domain
    triangle_domain = TriangleDomain(points=np.array([[0, 0], [1, 0], [0, 1]]), dim=2)

    # Unit square domain
    square_domain = QuadrilateralDomain(
        points=np.array([[0, 0], [1, 0], [1, 1], [0, 1]]), dim=2
    )

    # Visualize individual basis functions
    print("\nVisualizing individual basis functions...")

    # Visualize linear function
    visualizer.plot_basis_function(
        linear_x,
        domain=triangle_domain,
        title="Linear Basis: f(x,y) = x",
        view_angles=(30, 45),
    )

    # Visualize quadratic function
    visualizer.plot_basis_function(
        quadratic,
        domain=triangle_domain,
        title="Quadratic Basis: f(x,y) = x² + y²",
        view_angles=(30, 45),
    )

    # Visualize cubic bubble function
    visualizer.plot_basis_function(
        bubble,
        domain=square_domain,
        title="Bubble Function: f(x,y) = x(1-x)y(1-y)",
        view_angles=(30, 45),
        num_points=100,
    )

    # Visualize a function and its derivatives
    print("\nVisualizing a function and its derivatives...")
    visualizer.plot_function_derivatives(bubble, domain=square_domain, num_points=100)

    # Visualize multiple basis functions in a grid
    print("\nVisualizing multiple basis functions in a grid...")
    basis_set = [linear_x, linear_y, quadratic, bubble]
    titles = [
        "f(x,y) = x",
        "f(x,y) = y",
        "f(x,y) = x² + y²",
        "f(x,y) = x(1-x)y(1-y)",
    ]

    visualizer.plot_basis_set(
        basis_set, domain=square_domain, titles=titles, num_points=100
    )

    # Show a wireframe plot for better visualization of shape
    print("\nWireframe visualization of the bubble function...")
    visualizer.plot_basis_function(
        bubble,
        domain=square_domain,
        wireframe=True,
        title="Bubble Function Wireframe",
        num_points=100,
    )

    print("Done!")


if __name__ == "__main__":
    main()
