import numpy as np
from fem import rectangle_mesh, shape_tri_vec, grad_shape_tri_vec
from pynsfem.utils.quadrature import tri_gauss_points_weights
from scipy import sparse
from plot import plot_field
import time

"""
Poisson solver
Solves -Lapl(u) = f
             u  = ud on Dirichlet boundary
"""

t0 = time.time()

v0 = (0, 0)
v1 = (1, 1)
n = 50
m = 50
order = 1
gauss_order = 2
dim = 2

Mesh = rectangle_mesh(v0, v1, n, m)
V, F, E = Mesh
f = lambda x, y: np.sin(2.0 * np.pi * x) * np.cos(2.0 * np.pi * y)
f = lambda x, y: 6 * x * y * (1 - y) - 2 * x**3
Source = f(V[:, 0], V[:, 1])  # source discretization increases the error

# Dirichlet : u = uD on Dirichlet border
dirichlet = np.unique(E.flat)  # activate Dirichlet on nodes of interest
uexact = lambda x, y: (0.5 / (2.0 * np.pi) ** 2) * np.sin(2.0 * np.pi * x) * np.cos(2.0 * np.pi * y)
uexact = lambda x, y: -y * (1 - y) * x**3
ud = uexact(V[:, 0], V[:, 1])

# Newmann : grad(u).n = gN on Newmann border

b = np.zeros(len(V), dtype=np.float64)
data = np.zeros((len(F), (3 * order), (3 * order)), dtype=np.float64)
rows = np.zeros(len(F) * (3 * order) ** 2, dtype=int)
cols = np.zeros(len(F) * (3 * order) ** 2, dtype=int)

gauss_points, gauss_weights = tri_gauss_points_weights(gauss_order)

for i, el in enumerate(F):
    nodePts = V[el]
    nodeSrc = Source[el]
    el_A = np.zeros((3 * order, 3 * order))
    el_b = np.zeros(3 * order)
    gradN = grad_shape_tri_vec(gauss_points)  # gradN with respect to (xi, eta) : GAUSSx2x3
    J = nodePts[1:, :] - nodePts[0]  # Jacobian of the triangle : 2x2
    gradN = np.linalg.solve(J, gradN)  # gradN with respect to (x, y) : GAUSSx2x3
    volume = 1 / 2 * np.linalg.det(J)  # volume
    el_A = np.einsum("g, gik, gil->kl", gauss_weights, gradN, gradN) * volume

    N = shape_tri_vec(gauss_points)  # shape function values : GAUSSx3
    el_b = np.einsum("g, gj, gk, k->j", gauss_weights, N, N, nodeSrc) * volume

    search = np.searchsorted(dirichlet, el)
    mask = np.where(el == dirichlet[search])
    el_A[mask] = 0
    el_A[mask, mask] = 1
    el_b[mask] = ud[el[mask]]

    data[i] = el_A
    b[el] += el_b

rows = np.concatenate([F.reshape(-1, 1) for _ in range(3 * order)], axis=-1).flat
cols = np.concatenate([F for _ in range(3 * order)], axis=-1).flat

A = sparse.csr_matrix((data.flat, (rows, cols)))
u = sparse.linalg.bicg(A, b)

t1 = time.time()
print("Time : " + str(t1 - t0))

plot_field(V, F, F, u[0] - ud)
