x = [0, 8, 16, 24, 32, 40]
y = [14.621, 11.843, 9.870, 8.418, 7.305, 6.413]

n = len(x)

# Divided difference table
dd = [[0.0 for j in range(n)] for i in range(n)]

# First column = y values
for i in range(n):
    dd[i][0] = y[i]

# Calculate divided differences
for j in range(1, n):
    for i in range(n - j):
        dd[i][j] = (dd[i+1][j-1] - dd[i][j-1]) / (x[i+j] - x[i])

# Print table
for i in range(n):
    print([round(dd[i][j], 10) for j in range(n-i)])

# Newton coefficients = first row
print("\nNewton coefficients:")
for j in range(n):
    print(f"a{j} = {dd[0][j]}")