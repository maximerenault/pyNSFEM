"""
Visualization tools for basis functions in 3D using matplotlib
"""

import numpy as np
import matplotlib.pyplot as plt


class BasisFunctionVisualizer:
    """
    Class for visualizing basis functions in 3D using matplotlib

    Provides tools to sample basis functions on their reference domains
    and create 3D visualizations of the resulting surfaces.
    """

    def __init__(self, fig_size=(10, 8), cmap="viridis"):
        """
        Initialize the visualizer

        Args:
            fig_size (tuple): Figure size
            cmap (str): Colormap to use for the plots
        """
        self.fig_size = fig_size
        self.cmap = cmap

    def sample_function(self, basis_function, x_range, y_range, num_points=50):
        """
        Sample a basis function on a rectangular grid

        Args:
            basis_function: The basis function to evaluate
            x_range (tuple): Range of x values (min, max)
            y_range (tuple): Range of y values (min, max)
            num_points (int): Number of sample points in each direction

        Returns:
            tuple: (X grid, Y grid, Z values)
        """
        x = np.linspace(x_range[0], x_range[1], num_points)
        y = np.linspace(y_range[0], y_range[1], num_points)
        X, Y = np.meshgrid(x, y)
        Z = np.zeros_like(X)

        # Evaluate the basis function at each point in the grid
        for i in range(X.shape[0]):
            for j in range(X.shape[1]):
                point = np.array([X[i, j], Y[i, j]])
                Z[i, j] = basis_function.evaluate(point)

        return X, Y, Z

    def plot_basis_function(
        self,
        basis_function,
        x_range=(0, 1),
        y_range=(0, 1),
        num_points=50,
        title=None,
        view_angles=None,
        wireframe=False,
        show=True,
    ):
        """
        Plot a basis function in 3D

        Args:
            basis_function: The basis function to plot
            x_range (tuple): Range of x values (min, max)
            y_range (tuple): Range of y values (min, max)
            num_points (int): Number of sample points in each direction
            title (str): Plot title, defaults to "Basis Function Visualization"
            view_angles (tuple): View angles (elevation, azimuth) in degrees
            wireframe (bool): Whether to plot as wireframe instead of surface
            show (bool): Whether to show the plot immediately

        Returns:
            tuple: (figure, axis)
        """
        X, Y, Z = self.sample_function(basis_function, x_range, y_range, num_points)

        # Create the figure and 3D axis
        fig = plt.figure(figsize=self.fig_size)
        ax = fig.add_subplot(111, projection="3d")

        # Plot the function
        if wireframe:
            surf = ax.plot_wireframe(X, Y, Z, cmap=self.cmap)
        else:
            surf = ax.plot_surface(
                X, Y, Z, cmap=self.cmap, linewidth=0.2, antialiased=True
            )
            fig.colorbar(surf, shrink=0.5, aspect=5)

        # Set labels and title
        ax.set_xlabel("x")
        ax.set_ylabel("y")
        ax.set_zlabel("u")

        if title is None:
            title = "Basis Function Visualization"
        ax.set_title(title)

        # Set view angle if specified
        if view_angles is not None:
            ax.view_init(elev=view_angles[0], azim=view_angles[1])

        if show:
            plt.show()

        return fig, ax

    def plot_basis_set(
        self,
        basis_set,
        x_range=(0, 1),
        y_range=(0, 1),
        num_points=30,
        rows=None,
        cols=None,
        figsize=None,
        titles=None,
    ):
        """
        Plot multiple basis functions in a grid

        Args:
            basis_set: List or collection of basis functions
            x_range (tuple): Range of x values (min, max)
            y_range (tuple): Range of y values (min, max)
            num_points (int): Number of sample points in each direction
            rows (int): Number of rows in the grid
            cols (int): Number of columns in the grid
            figsize (tuple): Figure size
            titles (list): Titles for each subplot

        Returns:
            tuple: (figure, axes)
        """
        n = len(basis_set)
        if n == 0:
            raise ValueError("Empty basis set")

        # Determine grid dimensions
        if rows is None and cols is None:
            # Make a square-ish grid
            cols = int(np.ceil(np.sqrt(n)))
            rows = int(np.ceil(n / cols))
        elif rows is None:
            rows = int(np.ceil(n / cols))
        elif cols is None:
            cols = int(np.ceil(n / rows))

        # Create figure
        if figsize is None:
            figsize = (5 * cols, 4 * rows)

        fig = plt.figure(figsize=figsize)

        # Create subplots
        axes = []
        for i, func in enumerate(basis_set):
            if i >= rows * cols:
                break

            # Create 3D subplot
            ax = fig.add_subplot(rows, cols, i + 1, projection="3d")
            axes.append(ax)

            # Sample and plot the function
            X, Y, Z = self.sample_function(func, x_range, y_range, num_points)
            surf = ax.plot_surface(
                X, Y, Z, cmap=self.cmap, linewidth=0.1, antialiased=True
            )

            # Set title if provided
            if titles is not None and i < len(titles):
                ax.set_title(titles[i])
            else:
                ax.set_title(f"Basis {i}")

            # Set labels
            ax.set_xlabel("x")
            ax.set_ylabel("y")

        plt.tight_layout()
        plt.show()

        return fig, axes

    def plot_function_derivatives(
        self,
        basis_function,
        x_range=(0, 1),
        y_range=(0, 1),
        num_points=30,
        include_function=True,
    ):
        """
        Plot a basis function and its derivatives

        Args:
            basis_function: The basis function to plot
            x_range (tuple): Range of x values (min, max)
            y_range (tuple): Range of y values (min, max)
            num_points (int): Number of sample points in each direction
            include_function (bool): Whether to include the original function

        Returns:
            tuple: (figure, axes)
        """
        gradient = basis_function.gradient
        n_plots = len(gradient) + (1 if include_function else 0)

        # Create figure
        rows = 1
        cols = n_plots
        if n_plots > 3:
            rows = 2
            cols = int(np.ceil(n_plots / 2))

        fig = plt.figure(figsize=(6 * cols, 5 * rows))

        plot_idx = 1

        # Plot the original function if requested
        if include_function:
            ax = fig.add_subplot(rows, cols, plot_idx, projection="3d")
            X, Y, Z = self.sample_function(basis_function, x_range, y_range, num_points)
            surf = ax.plot_surface(
                X, Y, Z, cmap=self.cmap, linewidth=0.1, antialiased=True
            )
            ax.set_title("Original Function")
            ax.set_xlabel("x")
            ax.set_ylabel("y")
            plot_idx += 1

        # Plot the derivatives
        for i, deriv in enumerate(gradient):
            ax = fig.add_subplot(rows, cols, plot_idx, projection="3d")
            X, Y, Z = self.sample_function(deriv, x_range, y_range, num_points)
            surf = ax.plot_surface(
                X, Y, Z, cmap=self.cmap, linewidth=0.1, antialiased=True
            )
            ax.set_title(f"∂/∂{['x', 'y', 'z'][i]}")
            ax.set_xlabel("x")
            ax.set_ylabel("y")
            plot_idx += 1

        plt.tight_layout()
        plt.show()

        return fig, fig.axes
