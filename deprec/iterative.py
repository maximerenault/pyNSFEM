import numpy as np


def newton(f, J, x0, epsilon, max_iter):
    """Approximate solution of f(x)=0 by Newton's method.

    Parameters
    ----------
    f : function
        Function for which we are searching for a solution f(x)=0.
    Df : function
        Derivative of f(x).
    x0 : number
        Initial guess for a solution f(x)=0.
    epsilon : number
        Stopping criteria is abs(f(x)) < epsilon.
    max_iter : integer
        Maximum number of iterations of Newton's method.

    Returns
    -------
    xn : number
        Implement Newton's method: compute the linear approximation
        of f(x) at xn and find x intercept by the formula
            x = xn - f(xn)/Df(xn)
        Continue until abs(f(xn)) < epsilon and return xn.
        If Df(xn) == 0, return None. If the number of iterations
        exceeds max_iter, then return None.

    Examples
    --------
    >>> f = lambda x: x**2 - x - 1
    >>> Df = lambda x: 2*x - 1
    >>> newton(f,Df,1,1e-8,10)
    Found solution after 5 iterations.
    1.618033988749989
    """
    xn = x0
    for n in range(0, max_iter):
        fxn = f(xn)
        if np.linalg.norm(fxn) < epsilon:
            print("Found solution after", n, "iterations.")
            return xn
        Jacxn = J(xn)
        if np.linalg.det(Jacxn) == 0:
            print("Zero derivative. No solution found.")
            return None
        xn = xn - np.linalg.solve(Jacxn, fxn)
    print("Exceeded maximum iterations. No solution found.")
    return None


def jacobi(A, b, x0, epsilon, max_iter):
    """Solve the system Ax = b using Jacobi iteration method.
    
    Parameters
    ----------
    A : ndarray
        Coefficient matrix
    b : ndarray
        Right-hand side vector
    x0 : ndarray
        Initial guess
    epsilon : float
        Convergence tolerance
    max_iter : int
        Maximum number of iterations
        
    Returns
    -------
    x : ndarray
        Solution vector
    """
    n = len(b)
    x = x0.copy()
    x_new = np.zeros_like(x)
    
    for n_iter in range(max_iter):
        for i in range(n):
            # Calculate sum of A[i,j]*x[j] for j != i
            sum_ax = A[i, :i].dot(x[:i]) + A[i, i+1:].dot(x[i+1:])
            x_new[i] = (b[i] - sum_ax) / A[i, i]
            
        # Check convergence
        if np.linalg.norm(x_new - x) < epsilon:
            print(f"Found solution after {n_iter + 1} iterations.")
            return x_new
            
        x = x_new.copy()
    
    print("Exceeded maximum iterations. No solution found.")
    return None


def gauss_seidel(A, b, x0, epsilon, max_iter):
    """Solve the system Ax = b using Gauss-Seidel iteration method.
    
    Parameters
    ----------
    A : ndarray
        Coefficient matrix
    b : ndarray
        Right-hand side vector
    x0 : ndarray
        Initial guess
    epsilon : float
        Convergence tolerance
    max_iter : int
        Maximum number of iterations
        
    Returns
    -------
    x : ndarray
        Solution vector
    """
    n = len(b)
    x = x0.copy()
    
    for n_iter in range(max_iter):
        x_old = x.copy()
        for i in range(n):
            # Calculate sum of A[i,j]*x[j] for j != i
            sum_ax = A[i, :i].dot(x[:i]) + A[i, i+1:].dot(x[i+1:])
            x[i] = (b[i] - sum_ax) / A[i, i]
            
        # Check convergence
        if np.linalg.norm(x - x_old) < epsilon:
            print(f"Found solution after {n_iter + 1} iterations.")
            return x
    
    print("Exceeded maximum iterations. No solution found.")
    return None


def sor(A, b, x0, epsilon, max_iter, omega):
    """Solve the system Ax = b using Successive Over-Relaxation (SOR) method.
    
    Parameters
    ----------
    A : ndarray
        Coefficient matrix
    b : ndarray
        Right-hand side vector
    x0 : ndarray
        Initial guess
    epsilon : float
        Convergence tolerance
    max_iter : int
        Maximum number of iterations
    omega : float
        Relaxation parameter (1 < omega < 2)
        
    Returns
    -------
    x : ndarray
        Solution vector
    """
    n = len(b)
    x = x0.copy()
    
    for n_iter in range(max_iter):
        x_old = x.copy()
        for i in range(n):
            # Calculate sum of A[i,j]*x[j] for j != i
            sum_ax = A[i, :i].dot(x[:i]) + A[i, i+1:].dot(x[i+1:])
            # SOR formula
            x[i] = (1 - omega) * x[i] + omega * (b[i] - sum_ax) / A[i, i]
            
        # Check convergence
        if np.linalg.norm(x - x_old) < epsilon:
            print(f"Found solution after {n_iter + 1} iterations.")
            return x
    
    print("Exceeded maximum iterations. No solution found.")
    return None
