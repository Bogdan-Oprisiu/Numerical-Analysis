#!/usr/bin/env python3
import numpy as np

def is_symmetric(A, tol=1e-8):
    return np.allclose(A, A.T, atol=tol)

def is_positive_definite(A):
    try:
        np.linalg.cholesky(A)
        return True
    except np.linalg.LinAlgError:
        return False

def cholesky_decomposition(A):
    return np.linalg.cholesky(A)

def solve_with_cholesky(A, b):
    L = cholesky_decomposition(A)
    y = np.linalg.solve(L, b)
    x = np.linalg.solve(L.T, y)
    return x

def main():
    A = np.array([[10,7,8,7],[7,5,6,5],[8,6,10,9],[7,5,9,10]], float)
    b = np.array([32,23,33,31], float)

    print("Symmetric:", is_symmetric(A))
    print("Positive definite:", is_positive_definite(A))

    L = cholesky_decomposition(A)
    print("Cholesky factor L:\n", L)

    x = solve_with_cholesky(A, b)
    print("Solution x (via Cholesky):", x)

if __name__ == '__main__':
    main()
