import numpy as np
import matplotlib.pyplot as plt

# Given data
x = np.array([0, 8, 16, 24, 32, 40])
y = np.array([14.621, 11.843, 9.870, 8.418, 7.305, 6.413])


# 1. Newton's 5th order polynomial

n = len(x)

dd = np.zeros((n, n))
dd[:, 0] = y

for j in range(1, n):
    for i in range(n-j):
        dd[i, j] = (dd[i+1, j-1] - dd[i, j-1]) / (x[i+j] - x[i])

# Newton polynomial
def newton(T):
    result = dd[0, 0]
    product = 1

    for j in range(1, n):
        product *= (T - x[j-1])
        result += dd[0, j] * product

    return result


# 2. Quadratic spline coefficients

a = np.array([14.621, 11.843, 9.870, 8.418, 7.305])

b = np.array([
    -0.34725,
    -0.34725,
    -0.146,
    -0.217,
    -0.06125
])

c = np.array([
    0,
    0.012578125,
    -0.0044375,
    0.009734375,
    -0.00628125
])


# Quadratic spline
def spline(T):

    # Find which interval T belongs to
    if T <= 8:
        i = 0
    elif T <= 16:
        i = 1
    elif T <= 24:
        i = 2
    elif T <= 32:
        i = 3
    else:
        i = 4

    return a[i] + b[i] * (T - x[i]) + c[i] * (T - x[i])**2

# 3. Generate points for smooth curves

T = np.linspace(0, 40, 500)

newton_values = np.array([newton(t) for t in T])
spline_values = np.array([spline(t) for t in T])


# 4. Plot

plt.figure(figsize=(9, 6))

# Newton polynomial
plt.plot(T, newton_values, label="Newton 5th Order")

# Quadratic spline
plt.plot(T, spline_values, label="Quadratic Spline")

# Original data points
plt.scatter(x, y, color="black", label="Data Points", zorder=5)

plt.xlabel("Temperature T (°C)")
plt.ylabel("Dissolved Oxygen O (mg/L)")
plt.title("Newton Polynomial and Quadratic Spline")

plt.legend()
plt.grid(True)

plt.xlim(0, 40)

plt.show()