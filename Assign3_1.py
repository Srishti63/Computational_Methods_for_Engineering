import numpy as np
import matplotlib.pyplot as plt

# functions
def f(x):
    return np.exp((x - 1)**2) - 1

def df(x):
    return 2 * (x - 1) * np.exp((x - 1)**2)

def d2f(x):
    return 2 * np.exp((x - 1)**2) + 4 * (x - 1)**2 * np.exp((x - 1)**2)

def u(x):
    with np.errstate(divide='ignore', invalid='ignore'):
        res = f(x) / df(x)
        res = np.where(x == 1.0, 0.0, res) # Limit as x->1
    return res

def du(x):
    with np.errstate(divide='ignore', invalid='ignore'):
        res = 1 - (f(x) * d2f(x)) / (df(x)**2)
        # Limit of u'(x) as x->1 is 1/m = 1/2
        res = np.where(x == 1.0, 0.5, res)
    return res

# Plotting
x_vals = np.linspace(0, 2, 400)

plt.figure(figsize=(12, 5))

# Part (a) Plot
plt.subplot(1, 2, 1)
plt.plot(x_vals, f(x_vals), label="f(x)")
plt.plot(x_vals, df(x_vals), label="f'(x)")
plt.plot(x_vals, d2f(x_vals), label="f''(x)")
plt.axhline(0, color='black', linewidth=0.8, linestyle='--')
plt.title("Part (a): f(x) and its derivatives")
plt.legend()

# Part (b) Plot
plt.subplot(1, 2, 2)
plt.plot(x_vals, u(x_vals), label="u(x)")
plt.plot(x_vals, du(x_vals), label="u'(x)")
plt.axhline(0, color='black', linewidth=0.8, linestyle='--')
plt.title("Part (b): u(x) and u'(x)")
plt.legend()
plt.tight_layout()
plt.show()

# Part (c) Secant Methods
def secant_method(func, x_minus1, x_0, tol=1e-6, max_iter=50):
    x_prev = x_minus1
    x_curr = x_0
    
    for i in range(1, max_iter + 1):
        f_curr = func(x_curr)
        f_prev = func(x_prev)
        
        if abs(f_curr - f_prev) < 1e-14:
            break
            
        x_next = x_curr - f_curr * (x_prev - x_curr) / (f_prev - f_curr)
        
        if abs(x_next - x_curr) < tol:
            return x_next, i
            
        x_prev = x_curr
        x_curr = x_next
        
    return x_curr, max_iter

x_init_1 = -0.5
x_init_2 = 0.0
# Note on Modified Secant divergence: 
# The initial guesses of -0.5 and 0 create a negative secant slope on u(x). 
# This shoots the next guess far to the left (x ~ -8.85), where u(x) flattens towards an asymptote of 0. 
# The algorithm gets trapped taking massive steps toward negative infinity, eventually causing an overflow (NaN).

root_std, iter_std = secant_method(f, x_init_1, x_init_2)
root_mod, iter_mod = secant_method(u, x_init_1, x_init_2)

print(f"Standard Secant: Root = {root_std:.6f}, Iterations = {iter_std}")
print(f"Modified Secant: Root = {root_mod:.6f}, Iterations = {iter_mod}")