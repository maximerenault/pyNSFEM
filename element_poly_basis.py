"""
Get canonical basis (e_i) for 2D polynomials of degree 1 and above
with gradients [de_i/dx, de_i/dy] and hessians [d2e_i/dx2, d2e_i/dxdy, d2e_i/dy2],
"""


def tri_poly_basis(degree=1):
    if degree == 0:
        func = lambda x, y: (1,)
        grad = lambda x, y: ([0, 0],)
        hess = lambda x, y: ([0, 0, 0],)
    elif degree == 1:
        func0, grad0, hess0 = tri_poly_basis(0)
        func = lambda x, y: func0(x, y) + (x, y)
        grad = lambda x, y: grad0(x, y) + ([1, 0], [0, 1])
        hess = lambda x, y: hess0(x, y) + ([0, 0, 0], [0, 0, 0])
    elif degree == 2:
        func1, grad1, hess1 = tri_poly_basis(1)
        func = lambda x, y: func1(x, y) + (x * y, x**2, y**2)
        grad = lambda x, y: grad1(x, y) + ([y, x], [2 * x, 0], [0, 2 * y])
        hess = lambda x, y: hess1(x, y) + ([0, 1, 0], [2, 0, 0], [0, 0, 2])
    elif degree == 3:
        func2, grad2, hess2 = tri_poly_basis(2)
        func = lambda x, y: func2(x, y) + (x**2 * y, x * y**2, x**3, y**3)
        grad = lambda x, y: grad2(x, y) + ([2 * x * y, x**2], [y**2, 2 * y * x], [3 * x**2, 0], [0, 3 * y**2])
        hess = lambda x, y: hess2(x, y) + ([2 * y, 2 * x, 0], [0, 2 * y, 2 * x], [6 * x, 0, 0], [0, 0, 6 * y])
    elif degree == 4:
        func3, grad3, hess3 = tri_poly_basis(3)
        func = lambda x, y: func3(x, y) + (x**2 * y**2, x**3 * y, x * y**3, x**4, y**4)
        grad = lambda x, y: grad3(x, y) + (
            [2 * x * y**2, 2 * x**2 * y],
            [3 * x**2 * y, x**3],
            [y**3, 3 * x * y**2],
            [4 * x**3, 0],
            [0, 4 * y**3],
        )
        hess = lambda x, y: hess3(x, y) + (
            [2 * y**2, 4 * x * y, 2 * x**2],
            [6 * x * y, 3 * x**2, 0],
            [0, 3 * y**2, 6 * x * y],
            [12 * x**2, 0, 0],
            [0, 0, 12 * y**2],
        )
    return func, grad, hess


def tri_poly_basis_auto(degree=1):
    """
    Returns basis functions generator for 2D polynomial of degree d.
    It also returns gradient and hessian generators of said functions.
    """
    powerpairs = []
    for power in range(degree + 1):
        powerpairs += power_pair(power)
    func = lambda x, y: (x**a * y**b for a, b in powerpairs)
    grad = lambda x, y: (
        (
            a * x ** (a - 1) * y**b if a != 0 else 0,
            x**a * b * y ** (b - 1) if b != 0 else 0,
        )
        for a, b in powerpairs
    )
    hess = lambda x, y: (
        (
            a * (a - 1) * x ** (a - 2) * y**b if a != 0 and a != 1 else 0,
            a * x ** (a - 1) * b * y ** (b - 1) if a != 0 and b != 0 else 0,
            x**a * b * (b - 1) * y ** (b - 2) if b != 0 and b != 1 else 0,
        )
        for a, b in powerpairs
    )
    return func, grad, hess


def power_pair(power=1):
    """
    Generates power pairs for degree=power 2D polynomials.
    The order prioritizes high powers for x.
    Example : power = 3
    out = [(2,1),(1,2),(3,0),(0,3)]
    """
    p1 = power // 2
    p2 = power - p1
    if power % 2 == 0:
        out = [(p1, p2)]
    else:
        p1, p2 = p2, p1
        out = [(p1, p2), (p2, p1)]
    while p1 != power:
        p1 += 1
        p2 -= 1
        out.append((p1, p2))
        out.append((p2, p1))
    return out


if __name__ == "__main__":
    func, grad, hess = tri_poly_basis(3)
    print(func(0, 1), grad(0, 1), hess(0, 1))

    print(power_pair(5))

    func, grad, hess = tri_poly_basis_auto(3)
    print(tuple(func(0, 1)), tuple(grad(0, 1)), tuple(hess(0, 1)))
