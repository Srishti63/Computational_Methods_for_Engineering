import numpy as np

mat = np.array([[8, 20, 15], [20, 80, 50], [15, 50, 60]], dtype=float)


def cholesky_decomp(arr):

    e_vals = np.linalg.eigvalsh(arr)
    for val in e_vals:
        if val < 0:
            print("Not a Positive Semidefinite Matrix, cannot proceed")
            return False

    n = len(arr)
    l = np.zeros((n, n))

    for row in range(n):
        l[row][row] = np.sqrt(arr[row][row] - np.sum(l[row][:row] ** 2))
        for col in range(row + 1, n):
            l[col][row] = (
                arr[col][row] - np.sum(l[col][:row] * l[row][:row])
            ) / l[row][row]

    return l, np.transpose(l)


l_mat, l_trans = cholesky_decomp(mat)

if l_mat.any():
    print("L Matrix")
    print(l_mat)
    print()
    print("L Transpose Matrix")
    print(l_trans)
    print()
    print("L*L_T")
    print(np.matmul(l_mat, l_trans))