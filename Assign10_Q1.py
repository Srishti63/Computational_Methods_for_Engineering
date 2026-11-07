import numpy as np
import matplotlib.pyplot as plt

# ---------------- Gauss elimination (partial pivoting) ----------------
def gauss_elim(A, b):
    A = A.astype(float).copy()
    b = b.astype(float).copy()
    n = len(b)
    for k in range(n - 1):
        p = np.argmax(np.abs(A[k:, k])) + k
        if p != k:
            A[[k, p]] = A[[p, k]]
            b[[k, p]] = b[[p, k]]
        for i in range(k + 1, n):
            f = A[i, k] / A[k, k]
            A[i, k:] -= f * A[k, k:]
            b[i] -= f * b[k]
    x = np.zeros(n)
    for i in range(n - 1, -1, -1):
        x[i] = (b[i] - A[i, i + 1:] @ x[i + 1:]) / A[i, i]
    return x

# ---------------- Problem data ----------------
K = 0.49            # cal/(s.cm.C)
dx = dy = 10.0      # cm
N = 5               # nodes per side (i, j = 0..4)
nu = 4              # unknown nodes per side (i, j = 0..3)

def known(i, j):    # Dirichlet values: top T = 25i, right T = 25j
    return 25.0 * i if j == 4 else 25.0 * j

idx = lambda i, j: 4 * j + i

# ---------------- Assemble system ----------------
A = np.zeros((16, 16))
b = np.zeros(16)
for j in range(nu):
    for i in range(nu):
        p = idx(i, j)
        A[p, p] = -4.0
        for di, dj in [(1, 0), (-1, 0), (0, 1), (0, -1)]:
            ii, jj = i + di, j + dj
            if ii < 0: ii = 1          # insulated left: ghost = T(1,j)
            if jj < 0: jj = 1          # insulated bottom: ghost = T(i,1)
            if ii == 4 or jj == 4:     # Dirichlet neighbour -> RHS
                b[p] -= known(ii, jj)
            else:
                A[p, idx(ii, jj)] += 1.0

# ---------------- Solve ----------------
sol = gauss_elim(A, b)
print("Max |A@x - b| =", np.max(np.abs(A @ sol - b)))

T = np.zeros((N, N))               # T[j, i]
T[:4, :4] = sol.reshape(4, 4)
T[4, :] = 25.0 * np.arange(N)      # top
T[:, 4] = 25.0 * np.arange(N)      # right
np.set_printoptions(precision=4, suppress=True)
print("Temperature field (rows: y = 40 -> 0, cols: x = 0 -> 40):")
print(T[::-1])

# ---------------- Heat flux q = -K grad(T) ----------------
qx = np.zeros_like(T)
qy = np.zeros_like(T)
qx[:, 1:4] = -K * (T[:, 2:5] - T[:, 0:3]) / (2 * dx)
qx[:, 0] = 0.0                                                    # insulated
qx[:, 4] = -K * (3 * T[:, 4] - 4 * T[:, 3] + T[:, 2]) / (2 * dx)  # one-sided
qy[1:4, :] = -K * (T[2:5, :] - T[0:3, :]) / (2 * dy)
qy[0, :] = 0.0                                                    # insulated
qy[4, :] = -K * (3 * T[4, :] - 4 * T[3, :] + T[2, :]) / (2 * dy)  # one-sided
qmag = np.hypot(qx, qy)

# ---------------- Plots ----------------
x = np.arange(N) * dx
y = np.arange(N) * dy
X, Y = np.meshgrid(x, y)

fig, ax = plt.subplots(1, 2, figsize=(13, 5.5))

c1 = ax[0].contourf(X, Y, T, levels=20, cmap='inferno')
ax[0].contour(X, Y, T, levels=10, colors='k', linewidths=0.4)
ax[0].plot(X, Y, 'w.', ms=4)
fig.colorbar(c1, ax=ax[0], label='T (°C)')
ax[0].set_title('Temperature profile')
ax[0].set_xlabel('x (cm)'); ax[0].set_ylabel('y (cm)')
ax[0].set_aspect('equal')

c2 = ax[1].contourf(X, Y, qmag, levels=20, cmap='viridis')
ax[1].quiver(X, Y, qx, qy, color='w', pivot='mid')
fig.colorbar(c2, ax=ax[1], label='|q| (cal/s·cm²)')
ax[1].set_title('Heat flux  q = -K∇T')
ax[1].set_xlabel('x (cm)'); ax[1].set_ylabel('y (cm)')
ax[1].set_aspect('equal')

plt.tight_layout()
plt.show()