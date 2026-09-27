import numpy as np

# [b1, b2, b3, b4, b5, c2, c3, c4, c5]

A = np.array([
    [ 8,  0,  0,  0,  0,  0,  0,  0,  0],
    [ 0,  8,  0,  0,  0, 64,  0,  0,  0],
    [ 0,  0,  8,  0,  0,  0, 64,  0,  0],
    [ 0,  0,  0,  8,  0,  0,  0, 64,  0],
    [ 0,  0,  0,  0,  8,  0,  0,  0, 64],

    [-1,  1,  0,  0,  0,  0,   0,   0,  0],
    [ 0, -1,  1,  0,  0, -16,   0,   0,  0],
    [ 0,  0, -1,  1,  0,  0, -16,   0,  0],
    [ 0,  0,  0, -1,  1,  0,   0, -16,  0]
], dtype=float)

B = np.array([
    -2.778,
    -1.973,
    -1.452,
    -1.113,
    -0.892,
     0,
     0,
     0,
     0
], dtype=float)


# ---------------- GAUSS ELIMINATION ----------------

n = len(B)

# Forward elimination
for i in range(n):

    # Pivoting
    max_row = i
    for k in range(i + 1, n):
        if abs(A[k][i]) > abs(A[max_row][i]):
            max_row = k

    A[[i, max_row]] = A[[max_row, i]]
    B[[i, max_row]] = B[[max_row, i]]

    # Make elements below pivot zero
    for k in range(i + 1, n):

        factor = A[k][i] / A[i][i]

        for j in range(i, n):
            A[k][j] -= factor * A[i][j]

        B[k] -= factor * B[i]


# Back substitution
X = np.zeros(n)

for i in range(n - 1, -1, -1):

    sum_value = 0

    for j in range(i + 1, n):
        sum_value += A[i][j] * X[j]

    X[i] = (B[i] - sum_value) / A[i][i]


# Print answer
names = ["b1", "b2", "b3", "b4", "b5",
         "c2", "c3", "c4", "c5"]

for i in range(n):
    print(names[i], "=", X[i])