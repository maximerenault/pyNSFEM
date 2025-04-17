# pyNSFEM

A Python library for solving partial differential equations using the finite element method.

## Features

- Support for various finite elements:
  - Triangular cubic Hermite elements
  - Triangular quintic Argyris elements
  - Quadrilateral Q4 elements
  - Polynomial basis elements
- Mesh generation utilities for common geometries:
  - Rectangles
  - Circles
  - Annuli
  - Triangles
- Solvers for common PDEs:
  - Poisson equation
- Visualization tools for meshes and solutions

## Installation

```bash
pip install pynsfem
```

## Usage

### Creating a Mesh

```python
from pynsfem.utils.mesh_generation import create_rectangle_mesh
from pynsfem.utils.plotting import plot_mesh

# Create a rectangular mesh
mesh = create_rectangle_mesh(width=1.0, height=1.0, nx=10, ny=10)

# Plot the mesh
plot_mesh(mesh)
```

### Solving the Poisson Equation

```python
from pynsfem.solvers import PoissonSolver
from pynsfem.utils.plotting import plot_solution

# Create solver
solver = PoissonSolver(mesh, element_type='tri_cubic_hermite')

# Define source term and boundary conditions
def f(x, y):
    return 2 * np.pi**2 * np.sin(np.pi * x) * np.sin(np.pi * y)

def g(x, y):
    return 0.0

# Assemble and solve
solver.assemble_system(f=f, g=g)
u = solver.solve()

# Plot solution
plot_solution(mesh, u)
```

## Project Structure

```
pynsfem/
├── core/           # Core mesh and topology classes
├── elements/       # Finite element implementations
├── solvers/        # PDE solvers
└── utils/          # Utility functions
    ├── mesh_generation.py
    ├── plotting.py
    └── quadrature.py
```

## Dependencies

- NumPy
- Matplotlib

## License

MIT License 