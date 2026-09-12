import numpy as np
import matplotlib.pyplot as plt
from scipy import stats

x = np.array([0.5, 0.8, 1.5, 2.1, 5.4, 3.2, 1.8, 0.9, 2.2, 0.7, 
              4.3, 3.8, 4.7, 6.4, 4.1, 3.1, 2.7, 1.6, 0.6, 1.1])

y = np.array([5.12, 7.06, 8.65, 11.78, 26.37, 19.91, 10.57, 7.39, 10.99, 9.36, 
              22.35, 22.30, 24.08, 29.91, 20.25, 18.53, 15.25, 11.04, 7.25, 10.08])

n = len(x)


# (a) Estimate parameters: slope, intercept, and variance of residuals
x_mean = np.mean(x)
y_mean = np.mean(y)

# Calculate Slope (beta_1) and Intercept (beta_0) using least squares
Sy = np.sum((x - x_mean) * (y - y_mean))
import numpy as np
import matplotlib.pyplot as plt
from scipy import stats


# Data Preparation
x = np.array([0.5, 0.8, 1.5, 2.1, 5.4, 3.2, 1.8, 0.9, 2.2, 0.7,
              4.3, 3.8, 4.7, 6.4, 4.1, 3.1, 2.7, 1.6, 0.6, 1.1])

y = np.array([5.12, 7.06, 8.65, 11.78, 26.37, 19.91, 10.57, 7.39, 10.99, 9.36,
              22.35, 22.30, 24.08, 29.91, 20.25, 18.53, 15.25, 11.04, 7.25, 10.08])

n = len(x)

# (a) Estimate parameters: slope, intercept, and variance of residuals
x_mean = np.mean(x)
y_mean = np.mean(y)

# Sxy = sum of cross-products of deviations (how x and y move together)
# Sxx = sum of squared deviations of x (spread of x)
Sxy = np.sum((x - x_mean) * (y - y_mean))
Sxx = np.sum((x - x_mean)**2)

slope = Sxy / Sxx
intercept = y_mean - slope * x_mean

# Predictions and residuals
y_pred = intercept + slope * x
residuals = y - y_pred

# Variance of residuals (MSE) - divided by n-2 for unbiased estimation
var_residuals = np.sum(residuals**2) / (n - 2)

print("--- Part (a) ---")
print(f"Slope: {slope:.4f}")
print(f"Intercept: {intercept:.4f}")
print(f"Variance of residuals: {var_residuals:.4f}\n")

# (b) & (c) Plots for Residual Analysis
plt.figure(figsize=(12, 5))

# (b) Residuals vs. x
plt.subplot(1, 2, 1)
plt.scatter(x, residuals, color='blue', alpha=0.7)
plt.axhline(0, color='red', linestyle='--')
plt.xlabel('x')
plt.ylabel('Residuals')
plt.title('(b) Residuals vs. x \n(Checks independence and homoscedasticity)')

# (c) Histogram of residuals
plt.subplot(1, 2, 2)
plt.hist(residuals, bins=6, color='green', edgecolor='black', alpha=0.7)
plt.xlabel('Residuals')
plt.ylabel('Frequency')
plt.title('(c) Histogram of Residuals \n(Checks normality distribution)')

plt.tight_layout()
plt.savefig('residuals_combined.png', dpi=150, bbox_inches='tight')
plt.show()

# (d) Coefficient of Determination (R-squared)
SST = np.sum((y - y_mean)**2)   # Total Sum of Squares
SSR = np.sum((y_pred - y_mean)**2)  # Regression Sum of Squares

R_squared = SSR / SST

print("--- Part (d) ---")
print(f"Coefficient of determination (R^2): {R_squared:.4f}\n")

# (e) Predict y for x = 7 with 90% Confidence Interval
x_new = 7
y_new_pred = intercept + slope * x_new

alpha = 0.10
t_crit = stats.t.ppf(1 - alpha/2, df=n-2)

# Standard error for a new individual prediction (prediction interval)
se_prediction = np.sqrt(var_residuals * (1 + 1/n + (x_new - x_mean)**2 / Sxx))
pi_lower = y_new_pred - t_crit * se_prediction
pi_upper = y_new_pred + t_crit * se_prediction

# Standard error for the mean response (confidence interval)
se_mean = np.sqrt(var_residuals * (1/n + (x_new - x_mean)**2 / Sxx))
ci_lower = y_new_pred - t_crit * se_mean
ci_upper = y_new_pred + t_crit * se_mean

print("--- Part (e) ---")
print(f"Predicted y for x = 7: {y_new_pred:.4f}")
print(f"90% Prediction Interval: ({pi_lower:.4f}, {pi_upper:.4f})")
print(f"(Note: If strictly looking for the CI of the *mean* response at x=7, it is ({ci_lower:.4f}, {ci_upper:.4f}))")