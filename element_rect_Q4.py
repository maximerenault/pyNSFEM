import numpy as np

arr = np.array
sr = np.sqrt


def q4_shape_func(p=(0, 0)):
    xi, eta = p
    N = arr(
        [
            1 / 4 * (1 - xi) * (1 - eta),
            1 / 4 * (1 + xi) * (1 - eta),
            1 / 4 * (1 + xi) * (1 + eta),
            1 / 4 * (1 - xi) * (1 + eta),
        ]
    )
    return N


def q4_gauss_points_weights():
    return arr(
        [(-sr(3) / 3, -sr(3) / 3), (sr(3) / 3, -sr(3) / 3), (sr(3) / 3, sr(3) / 3), (-sr(3) / 3, sr(3) / 3)]
    ), arr([1, 1, 1, 1])


def check_q4_vandermonde_determinant(p0, p1, p2, p3):
    x0, y0 = p0
    x1, y1 = p1
    x2, y2 = p2
    x3, y3 = p3
    s1 = (x1 - x0) * (x3 - x2) * (y2 * y3 + y0 * y1)
    s2 = (x0 - x2) * (x3 - x1) * (y1 * y3 + y0 * y2)
    s3 = (x3 - x0) * (x2 - x1) * (y1 * y2 + y0 * y3)
    return s1 + s2 + s3


if __name__ == "__main__":
    """With a rectangle at 45° the basis 1, x, y, xy cannot provide a finite element, detV = 0"""
    h = 0.5
    l = 0.2
    p0 = (0, 0)
    p1 = (h, h)
    p2 = (h - l, h + l)
    p3 = (-l, +l)
    print(check_q4_vandermonde_determinant(p0, p1, p2, p3))

    """With another angle, we can define a finite element, detV != 0"""
    h = 0.5
    l = 0.2
    p0 = (0, 0)
    p1 = (h, l)
    p2 = (h - l, l + l)
    p3 = (-l, +l)
    print(check_q4_vandermonde_determinant(p0, p1, p2, p3))
