#!/usr/bin/env python3
"""
Exercise 2: Gradient Descent vs Newton on quadratic energy EA(x) = 1/2 x^T A x
- Compute gradient and Hessian
- Plot contour
- Experiment with different step sizes h for GD
- Compare GD trajectories with Newton
"""
import numpy as np
import matplotlib.pyplot as plt

# Define A and EA
A = np.array([[3,2],[2,6]], float)
def E_A(x):
    return 0.5 * x.dot(A).dot(x)

def grad_E_A(x):
    return A.dot(x)

def hess_E_A(x):
    return A

# Gradient Descent
def gradient_descent(grad, x0, h, tol=1e-4, maxiter=1000):
    x = np.array(x0, float)
    traj = [x.copy()]
    for i in range(maxiter):
        x_new = x - h*grad(x)
        traj.append(x_new.copy())
        if np.linalg.norm(x_new - x) < tol:
            break
        x = x_new
    return np.array(traj)

# Newton (exact for quadratic)
def newton(grad, hess, x0, tol=1e-4, maxiter=100):
    x = np.array(x0, float)
    traj = [x.copy()]
    for i in range(maxiter):
        dx = np.linalg.solve(hess(x), grad(x))
        x -= dx
        traj.append(x.copy())
        if np.linalg.norm(dx) < tol:
            break
    return np.array(traj)

# Plot contour and compare
def plot_compare(h):
    xs = np.linspace(-3, 3, 400)
    ys = np.linspace(-3, 3, 400)
    X, Y = np.meshgrid(xs, ys)
    Z = 0.5*(A[0,0]*X**2 + 2*A[0,1]*X*Y + A[1,1]*Y**2)
    plt.figure()
    plt.contour(X, Y, Z, levels=30)
    # start point
    x0 = np.array([2.5, 2.5])
    traj_gd = gradient_descent(grad_E_A, x0, h)
    traj_nt = newton(grad_E_A, hess_E_A, x0)
    plt.plot(traj_gd[:,0], traj_gd[:,1], '-o', label=f'GD h={h}')
    plt.plot(traj_nt[:,0], traj_nt[:,1], '-x', label='Newton')
    plt.legend()
    plt.title(f'GD vs Newton (h={h})')
    plt.xlabel('x1')
    plt.ylabel('x2')
    plt.show()

if __name__ == '__main__':
    for h in [0.05, 0.1, 0.2]:
        plot_compare(h)
