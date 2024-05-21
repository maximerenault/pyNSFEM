"""
Solve
   -Laplace(u) = f  in  (0,1) x (0,1)
            u  = g  on  boundary
"""

import meshio
import numpy as np
import matplotlib.pyplot as plt
from scipy.sparse import lil_matrix, csc_matrix
from scipy.sparse.linalg import spsolve
import numpy.linalg as nla
from fem import rectangle_mesh


# Assemble matrix and rhs on one triangle
def assemble_cell(x, y, f):
    o = np.ones(3)
    H = np.array([o, x, y])
    area = 0.5 * nla.det(H)
    G = nla.solve(H, np.array([[0.0, 0.0], [1.0, 0.0], [0.0, 1.0]]))
    A = area * G @ G.T
    xc = np.sum(x) / 3.0
    yc = np.sum(y) / 3.0
    b = (1.0 / 3.0) * area * f(xc, yc)
    return A, b


v0 = (0, 0)
v1 = (1, 1)
n = 10
m = 10
Mesh = rectangle_mesh(v0, v1, n, m)
V, cells, faces = Mesh
x, y = V[:, 0], V[:, 1]

# Find unique boundary points
bpts = np.unique(faces)

npoints = len(x)
ncells = cells.shape[0]
nfaces = faces.shape[0]
nbpts = len(bpts)

print("points, cells, faces, bpts = ", npoints, ncells, nfaces, nbpts)

# RHS function
f = lambda x, y: np.sin(2.0 * np.pi * x) * np.cos(2.0 * np.pi * y)
f = lambda x, y: 6 * x * y * (1 - y) - 2 * x**3

# Exact solution
uexact = lambda x, y: (0.5 / (2.0 * np.pi) ** 2) * np.sin(2.0 * np.pi * x) * np.cos(2.0 * np.pi * y)
uexact = lambda x, y: -y * (1 - y) * x**3

b = np.zeros(npoints)  # rhs vector
A = lil_matrix((npoints, npoints))  # system matrix

# Loop over cells and assemble
for c in range(ncells):
    cell = cells[c, :]
    xv, yv = x[cell], y[cell]
    Aloc, bloc = assemble_cell(xv, yv, f)
    # Copy local to global
    b[cell] += bloc
    A[np.ix_(cell, cell)] += Aloc

# Solution array
u = np.zeros(npoints)

# Fill boundary values into solution array u
for p in bpts:
    u[p] = uexact(x[p], y[p])

# Modify matrix and rhs to apply dirichlet bc
b -= A @ u
for c in bpts:
    b[c] = u[c]
    A[c, :] = 0.0  # other entries in c'th row
    A[:, c] = 0.0  # other entries in c'th column
    A[c, c] = 1.0

A = csc_matrix(A)  # convert to csc since spsolve needs this

print("Solving Ax=b ...")
u = spsolve(A, b)

# Plot solution
plt.figure()
c = plt.tricontourf(x, y, cells, u, cmap="rainbow", levels=20)
plt.colorbar(c)
plt.tricontour(x, y, cells, u, colors="k", levels=20)
plt.xlabel("x")
plt.ylabel("y")
plt.title("Solution")
plt.axis("equal")
# plt.savefig('sol.svg')

plt.show()
