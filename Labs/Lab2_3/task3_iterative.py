#!/usr/bin/env python3
import numpy as np
import time

def jacobi(A, b, x0=None, tol=1e-4, maxiter=10000):
    n = len(b)
    x = np.zeros(n) if x0 is None else x0.copy()
    D = np.diag(A)
    R = A - np.diagflat(D)
    for k in range(1, maxiter+1):
        x_new = (b - R.dot(x)) / D
        if np.linalg.norm(x_new - x, ord=np.inf) < tol:
            return x_new, k
        x = x_new
    return x, maxiter

def gauss_seidel(A, b, x0=None, tol=1e-4, maxiter=10000):
    n = len(b)
    x = np.zeros(n) if x0 is None else x0.copy()
    for k in range(1, maxiter+1):
        x_new = x.copy()
        for i in range(n):
            s1 = A[i, :i].dot(x_new[:i])
            s2 = A[i, i+1:].dot(x[i+1:])
            x_new[i] = (b[i] - s1 - s2) / A[i, i]
        if np.linalg.norm(x_new - x, ord=np.inf) < tol:
            return x_new, k
        x = x_new
    return x, maxiter

def main():
    A = np.array([[10,7,8,7],[7,5,6,5],[8,6,10,9],[7,5,9,10]], float)
    b = np.array([32,23,33,31], float)

    start = time.perf_counter()
    x_j, it_j = jacobi(A, b)
    t_j = time.perf_counter() - start

    start = time.perf_counter()
    x_gs, it_gs = gauss_seidel(A, b)
    t_gs = time.perf_counter() - start

    # Cholesky timing
    from numpy.linalg import cholesky, solve
    start = time.perf_counter()
    L = cholesky(A)
    y = solve(L, b)
    x_c = solve(L.T, y)
    t_c = time.perf_counter() - start

    print("Jacobi: iterations =", it_j, ", time =", t_j)
    print("Gauss-Seidel: iterations =", it_gs, ", time =", t_gs)
    print("Cholesky: time =", t_c)

if __name__ == '__main__':
    main()
