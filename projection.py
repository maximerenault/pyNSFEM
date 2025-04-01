import numpy as np
from fem import (
    rectangle_mesh,
    shape_tri,
    grad_shape_tri,
    size_base,
    define_elements,
    project_func_on_space,
)
from quadrature import tri_gauss_points_weights
from scipy import sparse
from plot import plot_field
import time

t0 = time.time()

v0 = (0, 0)
v1 = (1, 1)
n = 5
m = 5
space = "P2"
sizo = size_base(space)
gauss_order = 4
dim = 2

Mesh = rectangle_mesh(v0, v1, n, m)
V, F, E = Mesh
Elems, vecsize = define_elements(V, F, space)

f = lambda x, y: (0.5 / (2.0 * np.pi) ** 2) * np.sin(2.0 * np.pi * x) * np.cos(2.0 * np.pi * y)
f = lambda x, y: x**2 + y**2
fh = project_func_on_space(V, F, Elems, f, space)

gauss_points, gauss_weights = tri_gauss_points_weights(gauss_order)

b = np.zeros(vecsize, dtype=np.float64)
data = np.zeros((len(Elems), sizo, sizo), dtype=np.float64)
rows = np.zeros((len(Elems), sizo, sizo), dtype=int)
cols = np.zeros((len(Elems), sizo, sizo), dtype=int)

for i, el in enumerate(Elems):
    tri = F[i]
    nodePts = V[tri]
    el_A = np.zeros((sizo, sizo))
    el_b = np.zeros(sizo)
    J = nodePts[1:, :] - nodePts[0]  # Jacobian of the triangle : 2x2
    area = 0.5 * np.linalg.det(J)
    for gp, gw in zip(gauss_points, gauss_weights):
        N = shape_tri(gp, space)  # value of shape functions at gp : SIZO
        Ntf = shape_tri(gp, "P1")  # linear transformation : 3

        el_A += gw * np.einsum("i, j -> ij", N, N) * area
        el_b += gw * N * f(*(Ntf.dot(nodePts))) * area

    data[i] = el_A
    rows[i] = np.concatenate([el.reshape(-1, 1) for _ in range(sizo)], axis=1)
    cols[i] = np.concatenate([el for _ in range(sizo)], axis=0).reshape((sizo, sizo))
    b[el] += el_b

A = sparse.csr_matrix((data.flat, (rows.flat, cols.flat)))
u = sparse.linalg.spsolve(A, b)

t1 = time.time()
print("Time : " + str(t1 - t0))

plot_field(V, F, Elems, u, field_type=space, raised=False, edges=True)
plot_field(V, F, Elems, fh, field_type=space, raised=False, edges=True)
plot_field(V, F, Elems, fh - u, field_type=space, raised=False, edges=True)
