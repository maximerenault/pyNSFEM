import numpy as np
from fem import (
    rectangle_mesh,
    shape_tri,
    grad_shape_tri,
    size_base,
    define_elements,
    project_func_on_space,
    get_dirichlet,
)
from pynsfem.utils.quadrature import tri_gauss_points_weights
from scipy import sparse
from plot import plot_field
import time

"""
Poisson solver
Solves -Lapl(u) = f
             u  = uD on Dirichlet boundary
"""

t0 = time.time()

v0 = (0, 0)
v1 = (1, 1)
n = 5
m = 5
space = "P2"
sizo = size_base(space)
gauss_order = 2
dim = 2

Mesh = rectangle_mesh(v0, v1, n, m)
V, F, E = Mesh
Elems, vecsize = define_elements(V, F, space)

f = lambda x, y: np.sin(2.0 * np.pi * x) * np.cos(2.0 * np.pi * y)
# f = lambda x, y: 6 * x * y * (1 - y) - 2 * x**3

# Dirichlet : u = uD on Dirichlet border
dirichlet = get_dirichlet(F, Elems, E.flatten(), space, entity_type="edge")  # activate Dirichlet on edges of interest
uexact = lambda x, y: (0.5 / (2.0 * np.pi) ** 2) * np.sin(2.0 * np.pi * x) * np.cos(2.0 * np.pi * y)
# uexact = lambda x, y: y * (y - 1) * x**3
uD = project_func_on_space(V, F, Elems, uexact, space)

# Newmann : grad(u).n = gN on Newmann border
newmann = np.concatenate((np.argwhere(V[:, 1] == v0[1]), np.argwhere(V[:, 1] == v1[1]))).flatten()
gradu = lambda x, y: [-3 * y * (1 - y) * x**2, (2 * y - 1) * x**3]
gN = 0

b = np.zeros(vecsize, dtype=np.float64)
data = np.zeros((len(F), sizo, sizo), dtype=np.float64)
rows = np.zeros((len(F), sizo, sizo), dtype=int)
cols = np.zeros((len(F), sizo, sizo), dtype=int)

gauss_points, gauss_weights = tri_gauss_points_weights(gauss_order)

for i, el in enumerate(Elems):
    tri = F[i]
    nodePts = V[tri]
    el_A = np.zeros((sizo, sizo))
    el_b = np.zeros(sizo)
    for gp, gw in zip(gauss_points, gauss_weights):
        gradN = grad_shape_tri(gp, space)  # gradN with respect to (xi, eta) : 2xSIZO
        J = nodePts[1:, :] - nodePts[0]  # Jacobian of the triangle : 2x2
        gradN = np.linalg.solve(J, gradN)  # gradN with respect to (x, y) : 2xSIZO
        area = 0.5 * np.linalg.det(J)
        """
        phi_i = zeros[i]=1
        phi_j = zeros[j]=1
        gradphi_i = gradN.dot(phi_i)
        gradphi_j = gradN.dot(phi_j)
        aij = gradphi_i.dot(gradphi_j)
        """

        el_A += gw * gradN.T.dot(gradN) * area

        N = shape_tri(gp, space)  # value of shape functions at gp : 1xSIZO
        Ntf = shape_tri(gp, "P1")  # linear transformation : 1x3

        """
        val0 = sum(phi_0(gp) * src(gp) * area for gp in gauss_points)
        val1 = sum(phi_1(gp) * src(gp) * area for gp in gauss_points)
        val2 = sum(phi_2(gp) * src(gp) * area for gp in gauss_points)
        """

        el_b += gw * N * f(*(Ntf.dot(nodePts))) * area

    for j, row in enumerate(el):
        if row in dirichlet:
            el_A[j, :] = 0
            el_A[j, j] = 1
            el_b[j] = uD[row]

    data[i] = el_A
    rows[i] = np.concatenate([el.reshape(-1, 1) for _ in range(sizo)], axis=1)
    cols[i] = np.concatenate([el for _ in range(sizo)], axis=0).reshape((sizo, sizo))
    b[el] += el_b

A = sparse.csr_matrix((data.flat, (rows.flat, cols.flat)))
u = sparse.linalg.spsolve(A, b)

t1 = time.time()
print("Time : " + str(t1 - t0))

plot_field(V, F, Elems, u, field_type=space, raised=True, edges=True)
