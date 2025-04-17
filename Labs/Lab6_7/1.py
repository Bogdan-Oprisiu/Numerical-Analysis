import numpy as np
import matplotlib.pyplot as plt

# Exercise 1: Linear least squares for exponential model

# Original data (days 1 to 18)
x = np.arange(1, 19)
y = np.array([1, 1, 3, 3, 3, 3, 4, 6, 6, 9, 13, 15, 17, 29, 47, 59, 89, 123], dtype=float)

# Linearize by log-transform
y_tilde = np.log(y)

# Fit linear model: y_tilde = b * x + c
coeffs = np.polyfit(x, y_tilde, 1)
b, c = coeffs
a = np.exp(c)

print("Original fit parameters:")
print(f"  a = {a:.6f}, b = {b:.6f}")

# Extrapolate to day 30
x30 = 30
y30_pred = a * np.exp(b * x30)
print(f"Prediction for x=30: y = {y30_pred:.2f}")
print(f"Actual value: 1029. Prediction error: {y30_pred - 1029:.2f}")

# Missing-data subset
x_m = np.array([1, 4, 7, 10, 13, 16])
y_m = np.array([1, 3, 4, 9, 17, 59], dtype=float)
y_tilde_m = np.log(y_m)

# Fit on missing-data
coeffs_m = np.polyfit(x_m, y_tilde_m, 1)
b_m, c_m = coeffs_m
a_m = np.exp(c_m)

print("\nFit parameters with missing data:")
print(f"  a_m = {a_m:.6f}, b_m = {b_m:.6f}")

# Extrapolate missing-data model to day 30
y30_pred_m = a_m * np.exp(b_m * x30)
print(f"Missing-data prediction for x=30: y = {y30_pred_m:.2f}")
print(f"Error: {y30_pred_m - 1029:.2f}")

# Plot 1: log-data with regression lines
plt.figure(figsize=(8, 4))
plt.scatter(x, y_tilde, label='log(y) original data')
plt.plot(x, b * x + c, '-', label='Fit original')
plt.scatter(x_m, y_tilde_m, label='log(y) missing-data', marker='x')
plt.plot(x, b_m * x + c_m, '--', label='Fit missing-data')
plt.xlabel('x (days)')
plt.ylabel(r'$\ln(y)$')
plt.title('Linear Regression on Log-Transformed Data')
plt.legend()
plt.grid(True)

# Plot 2: exponential curves on original y-scale
plt.figure(figsize=(8, 4))
plt.scatter(x, y, label='y original data')
xs = np.linspace(1, 30, 200)
plt.plot(xs, a * np.exp(b * xs), '-', label='Exp fit original')
plt.plot(xs, a_m * np.exp(b_m * xs), '--', label='Exp fit missing-data')
plt.axvline(30, color='gray', linestyle=':')
plt.scatter([30], [1029], color='red', label='Actual at x=30')
plt.xlabel('x (days)')
plt.ylabel('y')
plt.title('Exponential Regression and Extrapolation to Day 30')
plt.legend()
plt.grid(True)

plt.show()
