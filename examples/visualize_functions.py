"""
Example script for visualizing basis functions using BasisFunctionVisualizer

This example demonstrates how to use the BasisFunctionVisualizer class
to create 3D visualizations of polynomial basis functions.
"""

from pynsfem.elements.components.basis import PolynomialBasisFunction
from pynsfem.elements.components.domain import TriangleDomain
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

    # Visualize individual basis functions
    print("\nVisualizing individual basis functions...")

    # Visualize linear function
    visualizer.plot_basis_function(
        linear_x, title="Linear Basis: f(x,y) = x", view_angles=(30, 45)
    )

    # Visualize quadratic function
    visualizer.plot_basis_function(
        quadratic, title="Quadratic Basis: f(x,y) = x² + y²", view_angles=(30, 45)
    )

    # Visualize cubic bubble function
    visualizer.plot_basis_function(
        bubble,
        title="Bubble Function: f(x,y) = x(1-x)y(1-y)",
        x_range=(0, 1),
        y_range=(0, 1),
        view_angles=(30, 45),
    )

    # Visualize a function and its derivatives
    print("\nVisualizing a function and its derivatives...")
    visualizer.plot_function_derivatives(
        bubble, x_range=(0, 1), y_range=(0, 1), num_points=40
    )

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
        basis_set, x_range=(0, 1), y_range=(0, 1), titles=titles, num_points=30
    )

    # Show a wireframe plot for better visualization of shape
    print("\nWireframe visualization of the bubble function...")
    visualizer.plot_basis_function(
        bubble,
        x_range=(0, 1),
        y_range=(0, 1),
        wireframe=True,
        title="Bubble Function Wireframe",
    )

    print("Done!")


if __name__ == "__main__":
    main()
