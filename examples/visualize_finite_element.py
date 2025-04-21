#!/usr/bin/env python3
"""
Example script demonstrating how to visualize finite element shape functions
for different element types (Lagrange and Argyris triangle elements).

This script shows:
1. How to create Lagrange triangle elements of different degrees
2. How to create an Argyris triangle element
3. How to visualize individual shape functions
4. How to visualize all shape functions in a grid layout
5. How to visualize derivatives of shape functions
"""

from pynsfem.elements.triangle.lagrange import ElementTriLagrange
from pynsfem.elements.triangle.argyris import ElementTriArgyris
from pynsfem.visualization.basis_visualizer import BasisFunctionVisualizer


def visualize_lagrange_elements():
    """Visualize Lagrange triangle element shape functions of different degrees"""
    print("Visualizing Lagrange triangle elements...")
    
    # Create a visualizer instance
    visualizer = BasisFunctionVisualizer(cmap="viridis")
    
    # Create elements of different degrees
    degrees = [1, 2, 3]
    
    for degree in degrees:
        # Create the element
        element = ElementTriLagrange(degree=degree)
        
        # Get the shape functions
        shape_functions = element.shape_functions
        
        print(f"  Lagrange degree {degree}: {len(shape_functions)} shape functions")
        
        # Plot all shape functions in a grid
        print(f"  Plotting all Lagrange degree {degree} shape functions...")
        titles = [f"N_{i}" for i in range(len(shape_functions))]
        visualizer.plot_basis_set(
            shape_functions,
            x_range=(-0.1, 1.1),
            y_range=(-0.1, 1.1),
            titles=titles,
            figsize=(12, 10)
        )
        
        # Plot first shape function and its derivatives
        print(f"  Plotting derivatives of first shape function for degree {degree}...")
        visualizer.plot_function_derivatives(
            shape_functions[0],
            x_range=(0, 1),
            y_range=(0, 1)
        )


def visualize_argyris_element():
    """Visualize Argyris triangle element shape functions"""
    print("Visualizing Argyris triangle element...")
    
    # Create a visualizer instance
    visualizer = BasisFunctionVisualizer(cmap="plasma")
    
    # Create the element
    element = ElementTriArgyris()
    
    # Get the shape functions
    shape_functions = element.shape_functions
    
    print(f"  Argyris element: {len(shape_functions)} shape functions")
    
    # Plot all shape functions in a grid
    # Argyris has 21 shape functions which makes for a large grid
    # Let's just plot a subset for demonstration
    print("  Plotting selected Argyris shape functions...")
    
    # Plot first 9 shape functions (first set of DOFs)
    subset = shape_functions[:9]
    titles = [f"N_{i}" for i in range(len(subset))]
    visualizer.plot_basis_set(
        subset,
        x_range=(-0.1, 1.1),
        y_range=(-0.1, 1.1),
        titles=titles,
        figsize=(15, 12),
        rows=3,
        cols=3
    )
    
    # Plot a single shape function in detail
    print("  Plotting single Argyris shape function in detail...")
    visualizer.plot_basis_function(
        shape_functions[0],
        x_range=(0, 1),
        y_range=(0, 1),
        title="Argyris Shape Function (First DOF)",
        view_angles=(30, 45),
        num_points=100
    )
    
    # Plot derivatives of one shape function
    print("  Plotting derivatives of first Argyris shape function...")
    visualizer.plot_function_derivatives(
        shape_functions[0],
        x_range=(0, 1),
        y_range=(0, 1)
    )


def main():
    """Main function to run all visualizations"""
    print("Finite Element Shape Function Visualization")
    print("===========================================")
    
    # Visualize Lagrange elements
    visualize_lagrange_elements()
    
    # Visualize Argyris element
    visualize_argyris_element()


if __name__ == "__main__":
    main()