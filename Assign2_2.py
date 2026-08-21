import math

# PART (a): GENERAL NEWTON-RAPHSON

def derivative(f, x, h=1e-6):
    return (f(x + h) - f(x - h)) / (2 * h)


def newton_raphson(f, x0, es=0.00001, max_iter=1000):

    x_old = x0

    for i in range(1, max_iter + 1):

        df = derivative(f, x_old)

        if df == 0:
            print("Derivative is zero. Method cannot continue.")
            return None

        x_new = x_old - f(x_old) / df

        if x_new != 0:
            ea = abs((x_new - x_old) / x_new) * 100
        else:
            ea = 100

        print(f"Iteration {i}")
        print(f"y = {x_new}")
        print(f"Error = {ea}%")
        print()

        if ea < es:
            return x_new

        x_old = x_new

    return x_new


# PART (b): CHANNEL FLOW PROBLEM

B = 20
S = 0.0002
n = 0.03
Q = 5


def f(y):

    discharge = (
        math.sqrt(S) * (B * y)**(5/3)
        / (n * (B + 2*y)**(2/3))
    )

    return discharge - Q

y0 = 1

root = newton_raphson(f, y0)


print("Final flow depth =", root, "m")