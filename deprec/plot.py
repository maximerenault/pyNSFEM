import matplotlib.pyplot as plt
import matplotlib.tri as tri
import numpy as np
from fem import shape_tri_vec
from pynsfem.utils.tri_subdiv import tri_subdiv

arr = np.array
sr = np.sqrt


class PlotException(Exception):
    pass


class BadFieldError(PlotException):
    def __init__(self, V, E, f):
        super().__init__(
            "unable to show field, number of vertices {}, number of elements {}, number of DOFs {}, number of field values {}".format(
                len(V), len(E), len(E[0]), len(f)
            )
        )


class RaisedFieldError(PlotException):
    def __init__(self, field_type):
        super().__init__("unable to show field, 'raised' option doesn't work for {} fields".format(field_type))


def plot_field(V, F, E, f, field_type=None, raised=False, edges=False):
    """2D triangular mesh field plot function.

    Args:
        V (array): vertices
        F (array): triangles
        E (array): elements
        f (array): field to plot
        field_type (str, optional): "P0", "P1", "P2". Defaults to None.
        raised (bool, optional): enables raised representation. Defaults to False.

    Raises:
        BadFieldError: Wrong shape of field f
        RaisedFieldError: Wrong shape of field f for raised representation
    """
    cmap = plt.cm.Spectral

    if field_type == None:
        if len(f) == len(V):
            field_type = "P1"
        elif len(E[0]) == 6:
            field_type = "P2"
        elif len(f) == len(F):
            field_type = "P0"
        else:
            raise BadFieldError(V, E, f)

    if field_type == "P2":
        base_pts = tri_subdiv(subdiv=5)
        linear_transform = shape_tri_vec(base_pts, "P1")  # NB_PTSx3
        shape_funcs = shape_tri_vec(base_pts, "P2")  # NB_PTSx6
        newV = np.einsum("pi, fij -> fpj", linear_transform, V[F]).reshape(-1, 2)
        newf = np.einsum("pi, ei -> ep", shape_funcs, f[E]).flatten()
        newV, idx = np.unique(newV, axis=0, return_index=True)
        newf = newf[idx]

    if raised:
        if field_type == "P1":
            V = np.concatenate((V, f.reshape(-1, 1)), axis=1)
        elif field_type == "P2":
            V = np.concatenate((V, f.reshape(-1, 1)[:len(V)]), axis=1)
            newV = np.concatenate((newV, newf.reshape(-1, 1)), axis=1)
        else:
            raise RaisedFieldError(field_type)

    if V.shape[-1] == 3:
        fig = plt.figure(figsize=(8, 8))
        ax = plt.axes(projection="3d")
        if field_type == "P1":
            trisurf = ax.plot_trisurf(V[:, 0], V[:, 1], V[:, 2], triangles=F, cmap=cmap, edgecolor="grey")
        elif field_type == "P2":
            trisurf = ax.plot_trisurf(newV[:, 0], newV[:, 1], newV[:, 2], cmap=cmap, edgecolor="none")
            trisurfmesh = ax.plot_trisurf(V[:, 0], V[:, 1], V[:, 2], triangles=F, color="none", edgecolor="grey")
        fig.colorbar(trisurf, ax=ax)
        plt.show()

    elif V.shape[-1] == 2:
        fig, ax = plt.subplots()
        ax.set_aspect("equal")
        triang = tri.Triangulation(V[:, 0], V[:, 1], F)
        if field_type == "P1":
            tpc = ax.tripcolor(triang, f, cmap=cmap, shading="gouraud")
        elif field_type == "P0":
            tpc = ax.tripcolor(triang, facecolors=f, cmap=cmap)
        elif field_type == "P2":
            newtriang = tri.Triangulation(newV[:, 0], newV[:, 1])
            tpc = ax.tripcolor(newtriang, newf, cmap=cmap, shading="gouraud", edgecolor="none")
        fig.colorbar(tpc, ax=ax)
        if edges:
            ax.triplot(triang, c="grey")
        plt.show()


def plot_tri_shape_funcs(funcs, subdiv=0, raised=False):
    """This allows to visualize a set of functions defined on the base triangle ((0,0),(1,0),(0,1)).
    The variable "funcs" must take x (float) and y (float) separately as arguments and return an array of values.

    Args:
        funcs (func): function that returns an array of values when given a position : funcs(x,y)=arr([a,...])
        subdiv (int, optional): Number of subdivisions. Defaults to 0.
        raised (bool, optional): 3D representation or top view heatmap. Defaults to False.
    """
    cmap = plt.cm.Spectral
    proj = None
    if raised:
        proj = "3d"

    V = tri_subdiv(subdiv)
    triang = tri.Triangulation(V[:, 0], V[:, 1])
    values = arr(list(map(lambda p: funcs(p[0], p[1]), V))).T
    nbfuncs = len(values)
    nrows = int(sr(nbfuncs)) + 1
    fig = plt.figure()

    for i, val in enumerate(values):
        ax = fig.add_subplot(nrows, nrows, i + 1, projection=proj)
        ax.set_aspect("equal")
        if raised:
            tpc = ax.plot_trisurf(triang, val, cmap=cmap, linewidth=0, edgecolor="none", antialiased=False)
        else:
            ax.axis("off")
            tpc = ax.tripcolor(triang, val, shading="gouraud")
        fig.colorbar(tpc, ax=ax)
    fig.subplots_adjust(wspace=0.1, hspace=0.1)
    plt.show()


if __name__ == "__main__":
    funcs = lambda x, y: arr([1 - x - y, x, y])
    plot_tri_shape_funcs(funcs, 30, True)
