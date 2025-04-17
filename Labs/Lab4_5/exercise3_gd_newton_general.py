#!/usr/bin/env python3
"""
Exercise 3: GD vs Newton on general energies E_L and E_R
- Experiment GD step sizes for E_L and E_R
- Compare Newton vs GD trajectories
"""
import numpy as np
import matplotlib.pyplot as plt
import warnings

# Suppress numpy overflow and invalid warnings
warnings.filterwarnings('ignore', category=RuntimeWarning)

# Definitions for E_L and E_R
def E_L(x):
    x1, x2 = x
    return x1**2/1.5 + x2**2/1.5 + 3*np.sin(x1 + x2/np.sqrt(2))**2

def grad_E_L(x):
    x1, x2 = x
    # gradient of x1^2/1.5 + x2^2/1.5 + 3 sin^2(phi)
    phi = x1 + x2/np.sqrt(2)
    dphi = 6*np.sin(phi)*np.cos(phi)
    return np.array([2*x1/1.5 + dphi,
                     2*x2/1.5 + dphi/np.sqrt(2)])

def hess_E_L(x):
    x1, x2 = x
    phi = x1 + x2/np.sqrt(2)
    sin2 = np.sin(phi); cos2 = np.cos(phi)
    d2 = 6*(cos2**2 - sin2**2)
    return np.array([[2/1.5 + d2, d2/np.sqrt(2)],
                     [d2/np.sqrt(2), 2/1.5 + d2/2]])


def E_R(x):
    x1, x2 = x
    return (1 - x1)**2 + 100*(x2 - x1**2)**2

def grad_E_R(x):
    x1, x2 = x
    return np.array([
        -2*(1 - x1) - 400*x1*(x2 - x1**2),
        200*(x2 - x1**2)
    ])

def hess_E_R(x):
    x1, x2 = x
    return np.array([
        [2 - 400*(x2 - x1**2) + 800*x1**2, -400*x1],
        [-400*x1, 200]
    ])

# Gradient Descent with overflow/NaN checks

def gradient_descent(grad, x0, h, tol=1e-4, maxiter=1000):
    x = np.array(x0, float)
    traj = [x.copy()]
    for i in range(maxiter):
        g = grad(x)
        x_new = x - h * g
        if not np.isfinite(x_new).all():
            warnings.warn(f"GD diverged at iteration {i}, stopping.")
            break
        traj.append(x_new.copy())
        if np.linalg.norm(x_new - x) < tol:
            break
        x = x_new
    return np.array(traj)

# Newton's method with checks

def newton(grad, hess, x0, tol=1e-4, maxiter=100):
    x = np.array(x0, float)
    traj = [x.copy()]
    for i in range(maxiter):
        H = hess(x)
        g = grad(x)
        try:
            dx = np.linalg.solve(H, g)
        except np.linalg.LinAlgError:
            warnings.warn(f"Hessian singular at iteration {i}, stopping Newton.")
            break
        x_new = x - dx
        if not np.isfinite(x_new).all():
            warnings.warn(f"Newton diverged at iteration {i}, stopping.")
            break
        traj.append(x_new.copy())
        if np.linalg.norm(dx) < tol:
            break
        x = x_new
    return np.array(traj)

# Plot helper comparing GD and Newton

def plot_compare(func_name, f, grad, hess, h_list, xmin, xmax, ymin, ymax):
    xs = np.linspace(xmin, xmax, 400)
    ys = np.linspace(ymin, ymax, 400)
    X, Y = np.meshgrid(xs, ys)
    Z = np.array([f([xx, yy]) for xx, yy in zip(X.ravel(), Y.ravel())]).reshape(X.shape)
    plt.figure()
    plt.contour(X, Y, Z, levels=30)

    # choose a start point near typical basin
    start = np.array([0.5, 0.5]) if func_name == 'E_L' else np.array([1.0, 1.0])

    # GD trajectories with different step sizes
    for h in h_list:
        traj = gradient_descent(grad, start, h)
        plt.plot(traj[:, 0], traj[:, 1], '-o', markevery=max(1, len(traj)//10), label=f'GD h={h}')

    # Newton trajectory
    traj_nt = newton(grad, hess, start)
    plt.plot(traj_nt[:, 0], traj_nt[:, 1], '-x', label='Newton')

    plt.title(f'{func_name}: GD vs Newton')
    plt.xlabel('x1')
    plt.ylabel('x2')
    plt.legend()
    plt.grid(True)
    plt.show()

if __name__ == '__main__':
    # Experiment with step sizes
    plot_compare('E_L', E_L, grad_E_L, hess_E_L, [0.01, 0.05, 0.1], -1, 1, -1, 1)
    plot_compare('E_R', E_R, grad_E_R, hess_E_R, [0.0005, 0.001, 0.002], -0.5, 2, -1, 3)
