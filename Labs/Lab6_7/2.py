import numpy as np
import matplotlib.pyplot as plt
from scipy.optimize import least_squares

# Data for Weibull model
x_data = np.array([0.132, 0.511, 0.701, 0.891, 1.081, 1.27, 1.46, 1.65, 1.839, 2.029, 2.219])
y_data = np.array([0.1,   0.543, 0.506, 0.606, 0.622, 0.569, 0.453, 0.438, 0.316, 0.29,  0.195])

# Full Weibull model W(x; a, b)
def W(p, x):
    a, b = p
    return a * b * x**(b - 1) * np.exp(-a * x**b)

# Residual for full model
def resid_full(p):
    return W(p, x_data) - y_data

# Fit full model via Levenberg–Marquardt
p0 = [1.0, 1.0]
res_full = least_squares(resid_full, p0, method='lm')
a_fit, b_fit = res_full.x

print(f"Full Weibull fit: a = {a_fit:.6f}, b = {b_fit:.6f}")

# Reduced model Wr(x; a) = 2 a x e^{-a x^2}
def Wr(p, x):
    a, = p
    return 2 * a * x * np.exp(-a * x**2)

def resid_reduced(p):
    return Wr(p, x_data) - y_data

# Fit reduced model
p0_r = [1.0]
res_red = least_squares(resid_reduced, p0_r, method='lm')
a_red, = res_red.x

print(f"Reduced model fit: a = {a_red:.6f}")

# Plot data and both models
xs = np.linspace(x_data.min(), x_data.max(), 200)
plt.figure(figsize=(6, 4))
plt.scatter(x_data, y_data, label='Data', marker='o')

plt.plot(xs, W([a_fit, b_fit], xs), '-', label=f'Weibull fit (a={a_fit:.3f}, b={b_fit:.3f})')
plt.plot(xs, Wr([a_red], xs), '--', label=f'Reduced fit (a={a_red:.3f})')

plt.xlabel('x')
plt.ylabel('y')
plt.title('Weibull Model vs Reduced Model')
plt.legend()
plt.grid(True)
plt.show()
