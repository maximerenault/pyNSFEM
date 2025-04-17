import numpy as np

def tri_subdiv(subdiv):
    """Generates a list of points corresponding to a subdivided base triangle ((0,0),(1,0),(0,1)).
    They match with Lagrange element points.

    Args:
        subdiv (int): subdivisions to create. 0 means only corner points. 2 means 2 points on edges and one at the center.

    Returns:
        arr((n,2)): list of points in base triangle
    """
    subdiv += 1
    V0 = np.array([0, 0])
    vecx = np.array([1, 0])
    vecy = np.array([0, 1])
    V = []
    multx, multy = np.meshgrid(range(subdiv + 1), range(subdiv + 1))
    mult = np.concatenate((multx.reshape(-1, 1), multy.reshape(-1, 1)), axis=1)
    for mx, my in mult:
        if mx + my > subdiv:
            continue
        V.append(V0 + mx / subdiv * vecx + my / subdiv * vecy)
    return np.array(V)