import numpy as np

mat = np.array([[7, 2, -3], [2, 5, -3], [1, -1, -6]], dtype=float)

vec = np.array([12, 18, -6], dtype=float)


def get_lu(arr):
    u = np.copy(arr)
    n = len(arr)
    m = len(arr[0])
    col = 0
    l = np.eye(n)
    for row1 in range(n - 1):
        for row2 in range(row1 + 1, m):
            l[row2][col] = u[row2][col] / u[row1][col]
            u[row2] = u[row2] - (u[row2][col] / u[row1][col]) * (u[row1])
        col = col + 1

    return l, u


l_mat, u_mat = get_lu(mat)
print("L Matrix")
print(l_mat)
print()
print("U Matrix")
print(u_mat)