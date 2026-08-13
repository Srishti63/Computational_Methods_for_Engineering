import matplotlib.pyplot as plt
import numpy as np

T = 1.0 
t = np.linspace(0.001, T - 0.001, 1000) 

C_t = np.abs((2 * np.pi * t / T) * (1 / np.tan(2 * np.pi * t / T)))

plt.figure(figsize=(8, 4))
plt.plot(t, C_t, label="Condition Number C(t)")
plt.yscale("log")  
plt.xlabel("t")
plt.ylabel("Condition Number C(t)")
plt.title("Condition Number of y(t) = sin(2πt/T)")
plt.grid(True)
plt.show()