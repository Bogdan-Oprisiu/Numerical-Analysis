import numpy as np
import matplotlib.pyplot as plt


# 1. Definition der Funktion und des Taylor-Polynoms
def f(x):
    return x ** 2 / (x ** 2 + 1)


def T2(x):
    return x ** 2  # Taylor-Polynom 2. Grades um x0 = 0


# 2. Wahl des x-Intervalls für die Darstellung
x_min, x_max = -5, 5
x = np.linspace(x_min, x_max, 1000)

# 3. Funktionswerte berechnen
y = f(x)
y_T2 = T2(x)

# 4. Plot erstellen
plt.figure(figsize=(8, 6))

# a) Graph von f(x)
plt.plot(x, y, label=r'$f(x) = \dfrac{x^2}{x^2 + 1}$', linewidth=2)

# b) Taylor-Approximation T2(x)
plt.plot(x, y_T2, label=r'$T_2(x) = x^2$', linestyle='--', linewidth=2)

# c) Horizontale Asymptote y = 1
plt.hlines(1, x_min, x_max, colors='gray', linestyles=':', label=r'Asymptote $y=1$')

# Achseneinstellungen
plt.xlim(x_min, x_max)
plt.ylim(-0.1, 1.1)
plt.xlabel('x')
plt.ylabel('y')
plt.title('Übung1: $f(x)$, $T_2(x)$ und Asymptote')
plt.legend()
plt.grid(True)

# Plot anzeigen
plt.show()
