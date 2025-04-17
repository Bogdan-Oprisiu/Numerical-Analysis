#!/usr/bin/env python3
import numpy as np
import scipy.linalg as la

def gaussian_elimination(A, b):
    A = A.astype(float).copy()
    b = b.astype(float).copy()
    n = len(b)
    for k in range(n-1):
        for i in range(k+1, n):
            if A[k, k] == 0:
                raise ZeroDivisionError("Zero pivot encountered")
            factor = A[i, k] / A[k, k]
            A[i, k:] -= factor * A[k, k:]
            b[i] -= factor * b[k]
    x = np.zeros(n)
    for i in range(n-1, -1, -1):
        if A[i, i] == 0:
            raise ZeroDivisionError("Zero pivot encountered")
        x[i] = (b[i] - np.dot(A[i, i+1:], x[i+1:])) / A[i, i]
    return x

def main():
    A = np.array([[10,7,8,7],[7,5,6,5],[8,6,10,9],[7,5,9,10]], float)
    A_tilde = np.array([[10,7,8.1,7.2],[7.08,5.04,6,5],[8,6,9.98,9],[6.99,4.99,9,9.98]], float)
    b = np.array([32,23,33,31], float)
    b_tilde = np.array([32.1,22.9,33.1,30.9], float)

    x = gaussian_elimination(A, b)
    x_tilde = gaussian_elimination(A_tilde, b_tilde)
    print("Solution x:", x)
    print("Solution x_tilde:", x_tilde)
    print("Difference x - x_tilde:", x - x_tilde)

    A_inv = la.inv(A)
    A_tilde_inv = la.inv(A_tilde)
    print("Eigenvalues of A:", la.eigvals(A))
    print("Eigenvalues of A_tilde:", la.eigvals(A_tilde))

    for M, name in [(A, 'A'), (A_tilde, 'A_tilde'), (A_inv, 'A_inv'), (A_tilde_inv, 'A_tilde_inv')]:
        print(f"Norms of {name}:")
        print("  1-norm:    ", la.norm(M, 1))
        print("  2-norm:    ", la.norm(M, 2))
        print("  inf-norm:  ", la.norm(M, np.inf))

if __name__ == '__main__':
    main()
