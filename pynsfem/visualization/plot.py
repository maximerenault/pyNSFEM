"""
Plotting functions for FEM results
"""

import matplotlib.pyplot as plt


def plot_solution(vertices, faces, solution, title="FEM Solution"):
    """
    Plots the FEM solution on a triangular mesh

    Args:
        vertices (np.ndarray): Array of vertex coordinates
        faces (np.ndarray): Array of face indices
        solution (np.ndarray): Solution vector
        title (str): Plot title
    """
    fig = plt.figure(figsize=(10, 8))
    ax = fig.add_subplot(111, projection="3d")

    # Plot the solution
    ax.plot_trisurf(vertices[:, 0], vertices[:, 1], solution, triangles=faces, cmap="viridis")

    ax.set_xlabel("x")
    ax.set_ylabel("y")
    ax.set_zlabel("u")
    ax.set_title(title)

    plt.show()


def plot_mesh(vertices, faces, title="FEM Mesh"):
    """
    Plots the triangular mesh

    Args:
        vertices (np.ndarray): Array of vertex coordinates
        faces (np.ndarray): Array of face indices
        title (str): Plot title
    """
    plt.figure(figsize=(8, 8))
    plt.triplot(vertices[:, 0], vertices[:, 1], faces, "b-", lw=0.5)
    plt.plot(vertices[:, 0], vertices[:, 1], "k.", markersize=5)
    plt.xlabel("x")
    plt.ylabel("y")
    plt.title(title)
    plt.axis("equal")
    plt.show()
