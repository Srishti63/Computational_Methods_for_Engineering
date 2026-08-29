"""
Numerical Methods: Linear Systems
Solving Ax = b using Gaussian Elimination with Partial Pivoting.
"""

import numpy as np


def solve_linear_system(A, b):
    """
    Solves a system of linear equations Ax = b via Gaussian Elimination.

    Parameters:
        A (array-like): Matrix of coefficients (n x n)
        b (array-like): Vector of constants (n)

    Returns:
        x (ndarray): Vector of solutions (n)
    """
    # Convert inputs to floating-point NumPy arrays
    A = np.array(A, dtype=float)
    b = np.array(b, dtype=float)
    n = len(b)

    # Form the augmented matrix [A | b]
    aug = np.column_stack((A, b))

    # --- Forward Elimination Stage ---
    for k in range(n - 1):
        # Locate maximum element in column k for partial pivoting
        pivot_idx = np.argmax(np.abs(aug[k:n, k])) + k

        # Check for singularity
        if np.isclose(aug[pivot_idx, k], 0.0):
            raise ValueError("System matrix is singular or ill-conditioned.")

        # Swap rows to place pivot element on diagonal
        if pivot_idx != k:
            aug[[k, pivot_idx]] = aug[[pivot_idx, k]]

        # Zero out sub-diagonal elements in column k
        for i in range(k + 1, n):
            multiplier = aug[i, k] / aug[k, k]
            aug[i, k:] -= multiplier * aug[k, k:]

    # --- Back Substitution Stage ---
    x = np.zeros(n)
    for i in range(n - 1, -1, -1):
        x[i] = (aug[i, -1] - np.dot(aug[i, i + 1:n], x[i + 1:n])) / aug[i, i]

    return x


if __name__ == "__main__":
    # Target system:
    #   2x1 - 6x2 -  x3 = -38
    #  -3x1 -  x2 + 7x3 = -34
    #  -8x1 +  x2 - 2x3 = -20

    coeff_matrix = [
        [2, -6, -1],
        [-3, -1, 7],
        [-8, 1, -2]
    ]
    const_vector = [-38, -34, -20]

    # Compute custom solution
    sol = solve_linear_system(coeff_matrix, const_vector)

    print("Computed Solutions:")
    for idx, val in enumerate(sol, start=1):
        print(f"x_{idx} = {val:8.4f}")

    # Cross-verify with NumPy standard solver
    sol_ref = np.linalg.solve(coeff_matrix, const_vector)
    print("\nReference Solutions (numpy.linalg.solve):")
    for idx, val in enumerate(sol_ref, start=1):
        print(f"x_{idx} = {val:8.4f}")