#!/usr/bin/env python3
"""
Exercise 1: Newton's method for optimization on E_L and E_R
- Compute gradient and Hessian for E_L and E_R
- Plot contour diagrams
- Implement generic Newton method
- Run from four random starts to approximate minima with tol=1e-4
- Plot trajectories on contours
"""
import numpy as np
import matplotlib.pyplot as plt

# Define E_L and E_R
def E_L(x):
    x1, x2 = x
    return x1**2/1.5 + x2**2/1.5 + 3*np.sin(x1 + x2/np.sqrt(2))**2

def grad_E_L(x):
    x1, x2 = x
    common = 6*np.sin(x1 + x2/np.sqrt(2))*np.cos(x1 + x2/np.sqrt(2))
    d1 = 2*x1/1.5 + common
    d2 = 2*x2/1.5 + common/np.sqrt(2)
    return np.array([d1, d2])

def hess_E_L(x):
    x1, x2 = x
    phi = x1 + x2/np.sqrt(2)
    sin2 = np.sin(phi)
    cos2 = np.cos(phi)
    # second derivative of 3 sin^2(phi) = 6 sin(phi)cos(phi)
    d11 = 2/1.5 + 6*(cos2**2 - sin2**2)
    d22 = 2/1.5 + 6*(cos2**2 - sin2**2)/2
    d12 = 6*(cos2**2 - sin2**2)/np.sqrt(2)
    return np.array([[d11, d12],[d12, d22]])

def E_R(x):
    x1, x2 = x
    return (1 - x1)**2 + 100*(x2 - x1**2)**2

def grad_E_R(x):
    x1, x2 = x
    d1 = -2*(1 - x1) - 400*x1*(x2 - x1**2)
    d2 = 200*(x2 - x1**2)
    return np.array([d1, d2])

def hess_E_R(x):
    x1, x2 = x
    h11 = 2 - 400*(x2 - x1**2) + 800*x1**2
    h22 = 200
    h12 = -400*x1
    return np.array([[h11, h12],[h12, h22]])

# Generic Newton method
def newton(f_grad, f_hess, x0, tol=1e-4, maxiter=100):
    x = np.array(x0, dtype=float)
    traj = [x.copy()]
    for i in range(maxiter):
        g = f_grad(x)
        H = f_hess(x)
        dx = np.linalg.solve(H, g)
        x -= dx
        traj.append(x.copy())
        if np.linalg.norm(dx, ord=2) < tol:
            break
    return np.array(traj)

# Plot contours and trajectories
def plot_newton(f, grad, hess, title, xmin, xmax, ymin, ymax):
    # contour
    xs = np.linspace(xmin, xmax, 400)
    ys = np.linspace(ymin, ymax, 400)
    X, Y = np.meshgrid(xs, ys)
    Z = np.array([f([xx,yy]) for xx,yy in zip(X.ravel(), Y.ravel())]).reshape(X.shape)
    plt.figure()
    plt.contour(X, Y, Z, levels=30)
    # random starts
    np.random.seed(0)
    for start in np.random.uniform([xmin, ymin], [xmax, ymax], (4,2)):
        traj = newton(grad, hess, start)
        plt.plot(traj[:,0], traj[:,1], '-o', markersize=3)
    plt.title(title)
    plt.xlabel('x1')
    plt.ylabel('x2')
    plt.show()

if __name__ == '__main__':
    plot_newton(E_L, grad_E_L, hess_E_L, 'Newton on E_L', -2, 2, -2, 2)
    plot_newton(E_R, grad_E_R, hess_E_R, 'Newton on E_R', -1, 2, -1, 3)
