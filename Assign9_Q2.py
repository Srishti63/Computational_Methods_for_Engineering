import matplotlib
matplotlib.use("Agg")   # remove this line when running in Jupyter/Colab
import matplotlib.pyplot as plt
import numpy as np

# Settings
a = 10.0           # decay rate in dy/dt = -a*y
h = 0.1
t = np.arange(0.0, 1.0 + h / 2, h)
n = len(t)

# Solution arrays, y(0) = 1
ye = np.zeros(n)   # explicit Euler
yi = np.zeros(n)   # implicit Euler
ye[0] = yi[0] = 1.0

for j in range(n - 1):
    ye[j + 1] = (1 - a * h) * ye[j]    # y(j+1) = y(j) + h*(-a*y(j))
    yi[j + 1] = yi[j] / (1 + a * h)    # y(j+1) = y(j) / (1 + a*h)

# Analytical solution on a fine grid
tf = np.linspace(0.0, 1.0, 200)
ya = np.exp(-a * tf)

plt.figure(figsize=(9, 5))
plt.plot(tf, ya, "k-", lw=2, label="Analytical: $y(t) = e^{-10t}$")
plt.plot(t, ye, "ro--", ms=6, alpha=0.8, label="Explicit Euler ($h=0.1$)")
plt.plot(t, yi, "bs--", ms=6, alpha=0.8, label="Implicit Euler ($h=0.1$)")
plt.title("Comparison of Explicit & Implicit Euler Methods ($dy/dt = -10y$)")
plt.xlabel("Time t (s)")
plt.ylabel("y(t)")
plt.grid(True, ls="--", alpha=0.7)
plt.legend()
plt.tight_layout()
plt.savefig("q2_plot.png", dpi=150)   # in Jupyter/Colab use plt.show() instead

print("t     explicit   implicit   exact")
for j in range(n):
    print(f"{t[j]:.1f}  {ye[j]:9.5f}  {yi[j]:9.5f}  {np.exp(-a * t[j]):9.5f}")