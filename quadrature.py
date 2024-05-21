import numpy as np

arr = np.array


class QuadratureException(Exception):
    pass


class BadGaussOrderError(QuadratureException):
    def __init__(self, order):
        super().__init__("unable to give gauss points for order {}".format(order))


def tri_gauss_points_weights(order=1):
    """
    Gives list of gauss points for triangles depending on order.
    Quadrature rules from https://people.sc.fsu.edu/~jburkardt/datasets/quadrature_rules_tri/quadrature_rules_tri.html.
    """
    if order == 1:
        return arr([(1 / 3, 1 / 3)]), arr([1])  # Midpoint rule
    elif order == 2:
        return arr([(0.5, 0), (0.5, 0.5), (0, 0.5)]), arr([1 / 3, 1 / 3, 1 / 3])
    elif order == 3:
        return arr([(1 / 3, 1 / 3), (0.6, 0.2), (0.2, 0.6), (0.2, 0.2)]), arr(
            [-0.5625, 1.5625 / 3, 1.5625 / 3, 1.5625 / 3]
        )
    elif order == 4:
        return arr(
            [
                (0.816847572980459, 0.091576213509771),
                (0.091576213509771, 0.816847572980459),
                (0.091576213509771, 0.091576213509771),
                (0.108103018168070, 0.445948490915965),
                (0.445948490915965, 0.108103018168070),
                (0.445948490915965, 0.445948490915965),
            ]
        ), arr(
            [
                0.109951743655322,
                0.109951743655322,
                0.109951743655322,
                0.223381589678011,
                0.223381589678011,
                0.223381589678011,
            ]
        )
    elif order == 5:
        return arr(
            [
                (0.33333333333333333, 0.33333333333333333),
                (0.79742698535308720, 0.10128650732345633),
                (0.10128650732345633, 0.79742698535308720),
                (0.10128650732345633, 0.10128650732345633),
                (0.05971587178976981, 0.47014206410511505),
                (0.47014206410511505, 0.05971587178976981),
                (0.47014206410511505, 0.47014206410511505),
            ]
        ), arr(
            [
                0.22500000000000000,
                0.12593918054482717,
                0.12593918054482717,
                0.12593918054482717,
                0.13239415278850616,
                0.13239415278850616,
                0.13239415278850616,
            ]
        )
    else:
        raise BadGaussOrderError(order)
