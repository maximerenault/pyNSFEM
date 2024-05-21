import numpy as np
from element_poly_basis import tri_poly_basis_auto
from quadrature import tri_gauss_points_weights

arr = np.array


def cubic_hermite_row(func, grad):
    return func(0, 0), func(1, 0), func(0, 1), *grad(0, 0), *grad(1, 0), *grad(0, 1), func(1 / 3, 1 / 3)


def cubic_hermite_cols(func, grad):
    """Takes canonical basis functions and returns its vandermonde columns.
    Cubic Hermite elements have 10 DOF :
    point evaluation at each corner, point evaluation of the gradient at each corner,
    and point evaluation at the center of the element.

    Args:
        func (func): canonical basis functions
        grad (func): canonical basis function gradients

    Returns:
        tuple(list,): columns of Vandermonde matrix
    """
    return (
        list(func(0, 0)),
        list(func(1, 0)),
        list(func(0, 1)),
        *arr(list(grad(0, 0))).T.tolist(),
        *arr(list(grad(1, 0))).T.tolist(),
        *arr(list(grad(0, 1))).T.tolist(),
        list(func(1 / 3, 1 / 3)),
    )


def cubic_hermite_vandermonde_mat():
    """
    Returns Vandermonde matrix for func list of basis functions.
    To get shape function value at point (x,y) do solve(mat,func(x,y)).
    Or for multiple instances :
    lu, piv = lu_factor(mat)
    lu_solve((lu, piv), func(x,y))
    """
    func, grad, _ = tri_poly_basis_auto(degree=3)
    mat = arr(cubic_hermite_cols(func, grad)).T
    return func, mat


def cubic_hermite_define_elements(V, F):
    """
    From a mesh, returns the elements (list of indices for components of function in the basis)
    and the number of unknowns.
    """
    unknowns = len(V) * 3 + len(F)
    El1 = F
    El2 = F + len(V)  # gradx
    El3 = F + 2 * len(V)  # grady
    El2 = np.concatenate((El2[:, [0]], El3[:, [0]], El2[:, [1]], El3[:, [1]], El2[:, [2]], El3[:, [2]]), axis=1)
    El3 = np.arange(0, len(F), 1, dtype=int).reshape(-1, 1) + 3 * len(V)
    print(El3)
    # p1, p2, p3, grad1x, grad1y, grad2x, grad2y, grad3x, grad3y, centerpoint
    Elements = np.concatenate((El1, El2, El3), axis=1)
    return Elements, unknowns


def cubic_hermite_project_func(V, F, Elements, func):
    projection = np.zeros((np.max(Elements) + 1))
    projection[: len(V)] = arr(list(map(lambda p: func(p[0], p[1]), V)))

    bfunc, mat = cubic_hermite_vandermonde_mat()
    gauss_points, gauss_weights = tri_gauss_points_weights(order=5)
    bfunc_gp = arr(list(map(lambda p: list(bfunc(p[0], p[1])), gauss_points))).T
    Ns = np.linalg.solve(mat, bfunc_gp).T  # GAUSSxSIZO
    Lin_transform = arr(list(map(lambda p: [1 - p[0] - p[1], p[0], p[1]], gauss_points)))  # GAUSSx3
    for el in Elements:
        nodePts = V[el[:3]]
        J = nodePts[1:, :] - nodePts[0]
        area = 0.5 * np.linalg.det(J)
        grads = el[3:-1]
        func_gp = list(map(lambda p: func(p[0], p[1]), Lin_transform.dot(nodePts)))  # GAUSS
        projection[grads] += np.einsum("g, gj, g -> j", gauss_weights, Ns[:,3:-1], func_gp) * area

    projection[-len(F) :] = arr(list(map(lambda p: func(p[0], p[1]), np.sum(V[F], axis=1) / 3)))
    
    matA = np.zeros((len(projection), len(projection)))
    vecb = np.zeros((len(projection)))
    for i, el in enumerate(Elements):
        tri = F[i]
        nodePts = V[tri]
        J = nodePts[1:, :] - nodePts[0]  # Jacobian of the triangle : 2x2
        area = 0.5 * np.linalg.det(J)
        el_A = np.einsum("g, gi, gj -> ij", gauss_weights, Ns, Ns) * area
        func_gp = list(map(lambda p: func(p[0], p[1]), Lin_transform.dot(nodePts)))  # GAUSS
        el_b =  np.einsum("g, gj, g -> j", gauss_weights, Ns, func_gp) * area
        # print(el_A, el_b)
        # print(np.linalg.solve(el_A, el_b))
        matA[np.ix_(el,el)] += el_A
        vecb[el] += el_b
    projection = np.linalg.solve(matA, vecb)
    print(projection)
    return projection


if __name__ == "__main__":
    func = lambda x, y: x * y
    grad = lambda x, y: (y, x)
    print(cubic_hermite_row(func, grad))

    """Vandermonde matrix of cubic Hermite elements with canonical basis"""
    V = [
        [1, 1, 1, 0, 0, 0, 0, 0, 0, 1],
        [0, 1, 0, 1, 0, 1, 0, 1, 0, 1 / 3],
        [0, 0, 1, 0, 1, 0, 1, 0, 1, 1 / 3],
        [0, 0, 0, 0, 0, 0, 1, 1, 0, 1 / 9],
        [0, 1, 0, 0, 0, 2, 0, 0, 0, 1 / 9],
        [0, 0, 1, 0, 0, 0, 0, 0, 2, 1 / 9],
        [0, 0, 0, 0, 0, 0, 1, 0, 0, 1 / 27],
        [0, 0, 0, 0, 0, 0, 0, 1, 0, 1 / 27],
        [0, 1, 0, 0, 0, 3, 0, 0, 0, 1 / 27],
        [0, 0, 1, 0, 0, 0, 0, 0, 3, 1 / 27],
    ]

    func, V2 = cubic_hermite_vandermonde_mat()
    invV2 = np.linalg.inv(V2)
    diff = arr(V) - V2
    print(np.linalg.norm(diff))
    print(invV2)
    print(invV2.dot(list(func(0, 0))))

    # plot_tri_shape_funcs(lambda x, y: np.linalg.solve(V2, list(func(x, y))), subdiv=20)

    V = arr([[0, 0], [1, 0], [0, 1], [-1, 0]])
    F = arr([[0, 1, 2], [0, 2, 3]])
    print(cubic_hermite_define_elements(V, F))

    Els = arr([[0, 1, 2, 4, 5, 6, 7, 10, 11, 12], [0, 2, 3, 4, 5, 10, 11, 8, 9, 13]])
    func = lambda x, y: x**3 + y**2
    grad = lambda x, y: [3*x**2, 2*y]
    cubic_hermite_project_func(V, F, Els, func)
    print(list(map(lambda p: func(p[0], p[1]), V)))
    print(list(map(lambda p: grad(p[0], p[1]), V)))
    print(list(map(lambda p: func(p[0], p[1]), np.sum(V[F], axis=1) / 3)))