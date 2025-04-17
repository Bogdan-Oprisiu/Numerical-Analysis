import numpy as np
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D
from scipy.optimize import minimize
import sympy as sp


# 1. Definition der Rosenbrock-Funktion
def rosenbrock(X):
    x, y = X
    return (1 - x) ** 2 + 100 * (y - x ** 2) ** 2


# 2. Gitter für 3D- und Konturplots
x_vals = np.linspace(-2, 2, 400)
y_vals = np.linspace(-1, 3, 400)
X, Y = np.meshgrid(x_vals, y_vals)
Z = (1 - X) ** 2 + 100 * (Y - X ** 2) ** 2

# 3. 3D-Oberflächenplot
fig = plt.figure(figsize=(8, 6))
ax = fig.add_subplot(111, projection='3d')
ax.plot_surface(X, Y, Z, rstride=5, cstride=5, alpha=0.7)
ax.set_xlabel('x')
ax.set_ylabel('y')
ax.set_zlabel('f(x,y)')
ax.set_title('Rosenbrock-Funktion: 3D-Oberfläche')
plt.show()

# 4. Konturplot
plt.figure(figsize=(6, 6))
levels = np.logspace(-0.5, 3.5, 20)
cs = plt.contour(X, Y, Z, levels=levels)
plt.clabel(cs, inline=1, fontsize=8)
plt.xlabel('x')
plt.ylabel('y')
plt.title('Rosenbrock-Funktion: Konturplot')
plt.grid(True)
plt.show()

# 5. Numerische Minimierung mit SciPy
res = minimize(rosenbrock, x0=[0, 0], method='BFGS')
print("Gefundenes Minimum bei x = {:.6f}, y = {:.6f}".format(*res.x))
print("Funktionswert am Minimum: f(x,y) = {:.6e}".format(res.fun))

# 6. Symbolische Untersuchung der Konvexität über die Hesse-Matrix
x, y = sp.symbols('x y')
f_sym = (1 - x) ** 2 + 100 * (y - x ** 2) ** 2
H = sp.hessian(f_sym, (x, y))
print("\nHesse-Matrix:")
sp.pprint(H)

# Eigenwerte an (1,1) und (0,0)
H_11 = H.subs({x: 1, y: 1})
eigs_11 = H_11.eigenvals()
print("\nEigenwerte der Hesse-Matrix bei (1,1):", eigs_11)

H_00 = H.subs({x: 0, y: 0})
eigs_00 = H_00.eigenvals()
print("Eigenwerte der Hesse-Matrix bei (0,0):", eigs_00)
