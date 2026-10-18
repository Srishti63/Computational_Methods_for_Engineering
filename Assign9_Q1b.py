import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

m, k = 20.0, 20.0

def f(t, y, c):
    """y = [x, v];  dx/dt = v,  dv/dt = -(c*v + k*x)/m"""
    x, v = y
    return np.array([v, -(c * v + k * x) / m])

def rk4(c, t0=0.0, tf=15.0, h=1.0, y0=(1.0, 0.0)):
    n = int(round((tf - t0) / h))
    t = np.linspace(t0, tf, n + 1)
    y = np.zeros((n + 1, 2))
    y[0] = y0
    for i in range(n):
        k1 = f(t[i], y[i], c)
        k2 = f(t[i] + h/2, y[i] + h/2 * k1, c)
        k3 = f(t[i] + h/2, y[i] + h/2 * k2, c)
        k4 = f(t[i] + h,   y[i] + h * k3,   c)
        y[i + 1] = y[i] + h/6 * (k1 + 2*k2 + 2*k3 + k4)
    return t, y

plt.figure(figsize=(8, 5))
for c, mk in zip([5, 40, 60], ["o-", "s-", "^-"]):
    t, y = rk4(c)
    print(f"\nc = {c} Ns/m\n   t        x(t)       v(t)")
    for ti, (xi, vi) in zip(t, y):
        print(f"{ti:5.0f}  {xi:10.5f} {vi:10.5f}")
    plt.plot(t, y[:, 0], mk, label=f"c = {c} Ns/m")

plt.axhline(0, color="k", lw=0.5)
plt.xlabel("Time, t (s)")
plt.ylabel("Displacement, x (m)")
plt.title("Damped spring-mass system (RK4, h = 1 s)")
plt.grid(True, alpha=0.3)
plt.legend()
plt.tight_layout()
plt.savefig("/mnt/user-data/outputs/displacement_plot.png", dpi=150)