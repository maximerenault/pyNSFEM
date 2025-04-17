import numpy as np
from element_tri_cubic_hermite import cubic_hermite_define_elements
from pynsfem.utils.quadrature import tri_gauss_points_weights

arr = np.array


class FemException(Exception):
    pass


class BadShapeSpaceError(FemException):
    def __init__(self, space):
        super().__init__("unable to give shape function value for space {}".format(space))


class BadGradShapeSpaceError(FemException):
    def __init__(self, space):
        super().__init__("unable to give shape function gradient value for space {}".format(space))


class BadSizeSpaceError(FemException):
    def __init__(self, space):
        super().__init__("unable to give basis size for space {}".format(space))


class BadElemSpaceError(FemException):
    def __init__(self, space):
        super().__init__("unable to define elements for space {}".format(space))


class BadProjectSpaceError(FemException):
    def __init__(self, space):
        super().__init__("unable to project function on space {}".format(space))


def shape_tri(p=(1 / 3, 1 / 3), space="P1"):
    """
    Shape function values at Gauss point p.
    N.dot(fi) with fi the coefficients of f in the element's basis will give
    the value of f at Gauss point p in the triangle.
    """
    xi, eta = p
    if space == "P1":
        # Point evaluation : p1, p2, p3
        N = arr([1 - xi - eta, xi, eta], dtype=np.float64)
    elif space == "P2":
        # Point evaluation : p1, p2, p3, e1, e2, e3
        N = arr(
            [
                1 - 3 * xi - 3 * eta + 4 * xi * eta + 2 * xi**2 + 2 * eta**2,
                -xi + 2 * xi**2,
                -eta + 2 * eta**2,
                4 * xi - 4 * xi * eta - 4 * xi**2,
                4 * xi * eta,
                4 * eta - 4 * xi * eta - 4 * eta**2,
            ],
            dtype=np.float64,
        )
    else:
        raise BadShapeSpaceError(space)
    return N


def shape_tri_vec(p=arr([(1 / 3, 1 / 3)]), space="P1"):
    """
    Same as shape_tri but returns a vectorized set of values.
    """
    xi, eta = p[:, 0], p[:, 1]
    if space == "P1":
        # Point evaluation : p1, p2, p3
        N = arr([1 - xi - eta, xi, eta], dtype=np.float64).T
    elif space == "P2":
        # Point evaluation : p1, p2, p3, e1, e2, e3
        N = arr(
            [
                1 - 3 * xi - 3 * eta + 4 * xi * eta + 2 * xi**2 + 2 * eta**2,
                -xi + 2 * xi**2,
                -eta + 2 * eta**2,
                4 * xi - 4 * xi * eta - 4 * xi**2,
                4 * xi * eta,
                4 * eta - 4 * xi * eta - 4 * eta**2,
            ],
            dtype=np.float64,
        ).T
    else:
        raise BadShapeSpaceError(space)
    return N


def grad_shape_tri(p=(1 / 3, 1 / 3), space="P1"):
    """
    Shape function gradients at Gauss point p. In this case, for linear
    shape functions, the gradient is constant on the whole element, so p is not used.
    /!\ It is the gradient with respect to xi/eta. It has to be multiplied by the inverse of
    the Jacobian to get the gradient with respect to x/y.
    """
    xi, eta = p
    if space == "P1":
        gradN = arr([[-1, 1, 0], [-1, 0, 1]], dtype=np.float64)
    elif space == "P2":
        gradN = arr(
            [
                [-3 + 4 * eta + 4 * xi, -1 + 4 * xi, 0, 4 - 4 * eta - 8 * xi, 4 * eta, -4 * eta],
                [-3 + 4 * xi + 4 * eta, 0, -1 + 4 * eta, -4 * xi, 4 * xi, 4 - 4 * xi - 8 * eta],
            ],
            dtype=np.float64,
        )
    else:
        raise BadGradShapeSpaceError(space)
    return gradN


def grad_shape_tri_vec(p=arr([(1 / 3, 1 / 3)]), space="P1"):
    """
    Same as grad_shape_tri but returns a vectorized gradient for multiple values of p.
    """
    xi, eta = p[:, 0], p[:, 1]
    if space == "P1":
        gradN = arr([[[-1, 1, 0], [-1, 0, 1]]] * len(p), dtype=np.float64)
    elif space == "P2":
        gradN = arr(
            [
                [-3 + 4 * eta + 4 * xi, -1 + 4 * xi, 0, 4 - 4 * eta - 8 * xi, 4 * eta, -4 * eta],
                [-3 + 4 * xi + 4 * eta, 0, -1 + 4 * eta, -4 * xi, 4 * xi, 4 - 4 * xi - 8 * eta],
            ],
            dtype=np.float64,
        )
        gradN = np.einsum("jki->ijk", gradN)
    else:
        raise BadGradShapeSpaceError(space)
    return gradN


def size_base(space="P1"):
    if space == "P1":
        order = 1
    elif space == "P2":
        order = 2
    else:
        raise BadSizeSpaceError(space)
    return ((order + 1) * (order + 2)) // 2


def rectangle_mesh(v0, v1, n, m):
    """
    v0 is the lower left corner of the rectangle
    v1 is the top right corner of the rectangle
    n is the number of subdivisions along x-axis
    m is the number of subdivisions along y-axis
    """
    x0, y0 = v0
    x1, y1 = v1
    x = np.linspace(x0, x1, n + 1, endpoint=True)
    y = np.linspace(y0, y1, m + 1, endpoint=True)
    xv, yv = np.meshgrid(x, y)
    v = np.concatenate((xv.reshape(-1, 1), yv.reshape(-1, 1)), axis=1, dtype=np.float64)
    f = [
        [
            j * (n + 1) + i,
            j * (n + 1) + i + 1,
            (j + 1) * (n + 1) + i,
            (j + 1) * (n + 1) + i,
            j * (n + 1) + i + 1,
            (j + 1) * (n + 1) + i + 1,
        ]
        for j in range(m)
        for i in range(n)
    ]
    f = arr(f, dtype=int).reshape(-1, 3)
    e1 = [[i, i + 1] for i in range(n)]
    e2 = [[m * (n + 1) + i + 1, m * (n + 1) + i] for i in range(n)]
    e3 = [[(i + 1) * (n + 1), i * (n + 1)] for i in range(m)]
    e4 = [[(i + 1) * (n + 1) - 1, (i + 2) * (n + 1) - 1] for i in range(m)]
    e = np.concatenate((e1, e2, e3, e4), dtype=int)
    return v, f, e


def define_elements(V, F, space="P1"):
    """Takes a triangular mesh and a function space and returns the elements
    as well as the number of unknowns.

    Args:
        V (arr(p,2)): Vertices
        F (arr(t,3)): Triangles
        space (str, optional): Space in which to define elements. Defaults to "P1".
    """
    if space == "P1":
        Elements = F
        unknowns = len(V)
    elif space == "P2":
        Edges = np.concatenate(
            (F[:, (0, 1)], F[:, (1, 2)], F[:, (2, 0)])
        )  # we have first edge, second edge, third edge
        Edges.sort(axis=1)
        unq_Edges, unq_idx = np.unique(Edges, axis=0, return_inverse=True)
        unq_idx += len(V)  # we put the edges at the end of our unknowns vector
        El1 = F
        El2 = unq_idx.reshape(3, -1).T
        Elements = np.concatenate((El1, El2), axis=1)  # p1, p2, p3, e1, e2, e3
        unknowns = len(unq_Edges) + len(V)
    elif space == "Cubic_Hermite":
        Elements, unknowns = cubic_hermite_define_elements(V, F)
    else:
        raise BadElemSpaceError(space)
    return Elements, unknowns


def project_func_on_space(V, F, Elements, func, space="P1"):
    """Returns the vector of the global projection of func on the full
    finite element space.

    Args:
        V (arr(p,2)): Vertices
        F (arr(t,3)): Triangles
        Elements (arr(t,sizo)): Elements
        func (_type_): Function that takes x and y, and returns a float
        space (str, optional): Space of projection. Defaults to "P1".

    Raises:
        BadProjectSpaceError: Bad space given

    Returns:
        arr(): Projection in the finite element basis
    """
    projection = []
    if space == "P1":
        projection = arr(list(map(lambda p: func(p[0], p[1]), V)))
    elif space == "P2":
        proj1 = arr(list(map(lambda p: func(p[0], p[1]), V)))
        proj2 = np.zeros((np.max(Elements.flat) - len(V) + 1))
        for el in Elements:
            nodes = el[:3]
            edges = el[3:] - len(V)
            proj2[edges] = arr(list(map(lambda p: func(p[0], p[1]), (V[nodes] + V[np.roll(nodes, -1)]) / 2)))
        projection = np.concatenate((proj1, proj2))
    else:
        raise BadProjectSpaceError(space)
    return projection


def get_dirichlet(F, Elements, Nodes, space="P1", entity_type="edge"):
    """Take a set of Nodes and returns a set of Dirichlet nodes.

    Args:
        F (arr(t,3)): Triangles
        Elements (arr(t,sizo)): Elements
        Nodes (arr(dir)): Set of nodes which describe geometric entities on which to apply Dirichlet BC
        space (str, optional): Space of elements. Defaults to "P1".
        entity_type (str, optional): Type of entity on which to apply Dirichlet BC. Defaults to "edge".

    Returns:
        arr(): Unique DOFs which are concerned by Dirichlet BC
    """
    dirichlet = []
    if space == "P1":
        if entity_type == "node":
            dirichlet = np.unique(Nodes)
        elif entity_type == "edge":
            edges = np.concatenate((F[:, [0, 1]], F[:, [1, 2]], F[:, [2, 0]]))
            edges = np.unique(np.sort(edges, axis=1), axis=0)
            edge_dir = edges[(edges.flat == Nodes.reshape(-1, 1)).any(axis=0).reshape(-1, 2).all(axis=1)]
            dirichlet = np.unique(edge_dir)
        elif entity_type == "triangle":
            tri_dir = F[(F.flat == Nodes.reshape(-1, 1)).any(axis=0).reshape(-1, 3).all(axis=1)]
            dirichlet = np.unique(tri_dir)
    elif space == "P2":
        if entity_type == "node":
            dirichlet = np.unique(Nodes)
        elif entity_type == "edge":
            edges1 = F[:, [0, 1]]
            edges2 = F[:, [1, 2]]
            edges3 = F[:, [2, 0]]
            edge_dir1 = Elements[(edges1.flat == Nodes.reshape(-1, 1)).any(axis=0).reshape(-1, 2).all(axis=1)][
                :, [0, 1, 3]
            ]
            edge_dir2 = Elements[(edges2.flat == Nodes.reshape(-1, 1)).any(axis=0).reshape(-1, 2).all(axis=1)][
                :, [1, 2, 4]
            ]
            edge_dir3 = Elements[(edges3.flat == Nodes.reshape(-1, 1)).any(axis=0).reshape(-1, 2).all(axis=1)][
                :, [2, 0, 5]
            ]
            dirichlet = np.unique(np.concatenate((edge_dir1, edge_dir2, edge_dir3)))
        elif entity_type == "triangle":
            tri_dir = Elements[(F.flat == Nodes.reshape(-1, 1)).any(axis=0).reshape(-1, 3).all(axis=1)]
            dirichlet = np.unique(tri_dir)
    return dirichlet


if __name__ == "__main__":
    triangle = arr([[0, 0], [1, 0], [0.5, 0.5]], dtype=float)
    dN = grad_shape_tri()
    J = dN.dot(triangle)

    f = [4, 0, 0]
    # Gauss center point, order 1
    N1 = shape_tri((1 / 3, 1 / 3))
    val1 = N1.dot(f)

    # Gauss middle points, order 2
    N21 = shape_tri((0.5, 0))
    N22 = shape_tri((0.5, 0.5))
    N23 = shape_tri((0, 0.5))
    val2 = 1 / 3 * N21.dot(f) + 1 / 3 * N22.dot(f) + 1 / 3 * N23.dot(f)
    print(1 / 2 * np.linalg.det(J), val1, val2)

    V = arr([[0, 0], [1, 0], [0, 1], [-1, 0]])
    F = arr([[0, 1, 2], [0, 2, 3]])
    print(define_elements(V, F), define_elements(V, F, "P2"))

    # from plot import plot_tri_shape_funcs
    # plot_tri_shape_funcs(lambda x, y: shape_tri((x, y), "P2"), 20)

    get_dirichlet(F, F, arr([3, 1]), "P1", "node")
