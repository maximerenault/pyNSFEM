import numpy as np
from element_poly_basis import tri_poly_basis_auto
from plot import plot_tri_shape_funcs

arr = np.array
sr = np.sqrt


def quintic_argyris_cols(func, grad, hess):
    return (
        list(func(0, 0)),
        list(func(1, 0)),
        list(func(0, 1)),
        *arr(list(grad(0, 0))).T.tolist(),
        *arr(list(grad(1, 0))).T.tolist(),
        *arr(list(grad(0, 1))).T.tolist(),
        *arr(list(hess(0, 0))).T.tolist(),
        *arr(list(hess(1, 0))).T.tolist(),
        *arr(list(hess(0, 1))).T.tolist(),
        arr(list(grad(0.5, 0))).dot([0,-1]),
        arr(list(grad(0, 0.5))).dot([-1,0]),
        arr(list(grad(0.5, 0.5))).dot([1/sr(2),1/sr(2)]),
    )


def quintic_argyris_vandermonde_mat():
    """
    Returns Vandermonde matrix for func list of basis functions.
    To get shape function value at point (x,y) do solve(mat,func(x,y)).
    Or for multiple instances :
    lu, piv = lu_factor(mat)
    lu_solve((lu, piv), func(x,y))
    """
    func, grad, hess = tri_poly_basis_auto(degree=5)
    mat = arr(quintic_argyris_cols(func, grad, hess)).T
    return func, mat


if __name__ == "__main__":
    func, V = quintic_argyris_vandermonde_mat()
    invV = np.linalg.inv(V)
    print(invV)
    print(invV.dot(list(func(0,0))))

    plot_tri_shape_funcs(lambda x, y: np.linalg.solve(V, list(func(x, y))), subdiv=20, raised=True)