import numpy as np
from fem import rectangle_mesh, shape_tri_vec, grad_shape_tri_vec, gauss_points_weights
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


gauss_points, gauss_weights = gauss_points_weights(gauss_order)

# A matrix
nodePts = V[F]  # elements with explicit positions : TRIx3x2
nodeSrc = Source[F]  # values of source in elements : TRIx3
gradN = grad_shape_tri_vec(gauss_points)  # gradN with respect to (xi, eta) : GAUSSx2x3
J = np.einsum("gij, tjk->tgik", gradN, nodePts)  # Jacobian of the triangles : TRIxGAUSSx2x2
gradN = np.linalg.solve(J, gradN.reshape((1,) + gradN.shape))  # gradN with respect to (x, y) : TRIxGAUSSx2x3
volumes = 1 / 2 * np.linalg.det(J)  # triangle areas : TRIxGAUSS
data = np.einsum("g, tg, tgik, tgil->tkl", gauss_weights, volumes, gradN, gradN)  # local matrices : TRIx3x3

rows = np.concatenate([F.reshape(-1, 1) for _ in range(3 * order)], axis=-1).flat
cols = np.concatenate([F for _ in range(3 * order)], axis=-1).flat

# b vector
b = np.zeros(len(V), dtype=np.float64)
N = shape_tri_vec(gauss_points)  # shape function values : GAUSSx3
locb = np.einsum("g, tg, gi, gj, tj->ti", gauss_weights, volumes, N, N, nodeSrc)  # local b vectors : TRIx3
np.add.at(b, F, locb)

# Dirichlet BCs
data = data.reshape(-1, 3 * order)
search = np.searchsorted(dirichlet, F.flat)
Funique, idx = np.unique(F.flat, return_index=True)
searchun = search[idx]
data = np.where((F.flat == dirichlet[search]).reshape(-1, 1), 0.0, data)  # rows to 0
identity = np.array([np.identity(3 * order)] * len(F)).reshape(-1, 3 * order)
data[idx] = np.where(
    (Funique == dirichlet[searchun]).reshape(-1, 1), identity[idx], data[idx]
)  # first appearing rows to identity
b[dirichlet] = ud[dirichlet]

# Solve
A = sparse.csr_matrix((data.flat, (rows, cols)))
u = sparse.linalg.spsolve(A, b)

t1 = time.time()
print("Time : " + str(t1 - t0))

plot_field(V, F, u - ud)
plot_field(V, F, u)
