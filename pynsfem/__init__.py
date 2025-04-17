"""
pyNSFEM - A Python library for Non-Standard Finite Element Methods
"""

from .core.mesh import Mesh
from .core.topology import Topology
from .core.geometry import Geometry
from .elements import (
    ElementTriCubicHermite,
    ElementTriQuinticArgyris,
    ElementQuadQ4
)
from .solvers.poisson import PoissonSolver
from .utils.plotting import plot_mesh, plot_solution

__version__ = "0.1.0"
__all__ = [
    'Mesh',
    'Topology',
    'Geometry',
    'ElementTriCubicHermite',
    'ElementTriQuinticArgyris',
    'ElementQuadQ4',
    'PoissonSolver',
    'plot_mesh',
    'plot_solution'
]
