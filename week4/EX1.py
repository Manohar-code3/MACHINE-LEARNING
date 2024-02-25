# Simple Linear Regression

# 1. Implement Linear Regression and calculate sum of residual error on the following
# Datasets.
# x = [0, 1, 2, 3, 4, 5, 6, 7, 8, 9]
# y = [1, 3, 2, 5, 7, 8, 8, 9, 10, 12]
#  Compute the regression coefficients using analytic formulation and calculate Sum
# Squared Error (SSE) and R2 value.
#  Implement gradient descent (both Full-batch and Stochastic with stopping
# criteria) on Least Mean Square loss formulation to compute the coefficients of
# regression matrix and compare the results using performance measures such as R2
# SSE etc.

import numpy as np

# Define the data
x = np.array([0, 1, 2, 3, 4, 5, 6, 7, 8, 9])
y = np.array([1, 3, 2, 5, 7, 8, 8, 9, 10, 12])

# Analytical solution
def linear_regression_analytical(x, y):
    n = len(x)
    x_mean = np.mean(x)
    y_mean = np.mean(y)
    xy_mean = np.mean(x * y)
    x_squared_mean = np.mean(x ** 2)
    
    # Calculate the coefficients
    b1 = (xy_mean - x_mean * y_mean) / (x_squared_mean - x_mean ** 2)
    b0 = y_mean - b1 * x_mean
    
    # Calculate the predicted y values
    y_pred = b0 + b1 * x
    
    # Calculate the sum of squared errors (SSE)
    sse = np.sum((y - y_pred) ** 2)
    
    # Calculate the R-squared value
    ssr = np.sum((y_pred - y_mean) ** 2)
    sst = np.sum((y - y_mean) ** 2)
    r_squared = ssr / sst
    
    return b0, b1, sse, r_squared

# Full-batch gradient descent
def linear_regression_full_batch(x, y, learning_rate=0.01, epochs=1000, tolerance=1e-6):
    n = len(x)
    b0 = 0
    b1 = 0
    prev_loss = float('inf')
    
    for epoch in range(epochs):
        y_pred = b0 + b1 * x
        loss = np.sum((y_pred - y) ** 2) / (2 * n)
        
        if abs(prev_loss - loss) < tolerance:
            break
        
        prev_loss = loss
        
        # Update the coefficients using gradient descent
        b0 -= learning_rate * np.sum(y_pred - y) / n
        b1 -= learning_rate * np.sum((y_pred - y) * x) / n
    
    # Calculate the R-squared value
    y_mean = np.mean(y)
    ssr = np.sum((y_pred - y_mean) ** 2)
    sst = np.sum((y - y_mean) ** 2)
    r_squared = ssr / sst
    
    return b0, b1, loss, r_squared

# Stochastic gradient descent
def linear_regression_stochastic(x, y, learning_rate=0.01, epochs=1000, tolerance=1e-6):
    n = len(x)
    b0 = 0
    b1 = 0
    prev_loss = float('inf')
    
    for epoch in range(epochs):
        for i in range(n):
            y_pred = b0 + b1 * x[i]
            loss = (y_pred - y[i]) ** 2 / 2
            
            if abs(prev_loss - loss) < tolerance:
                break
            
            prev_loss = loss
            
            # Update the coefficients using stochastic gradient descent
            b0 -= learning_rate * (y_pred - y[i])
            b1 -= learning_rate * (y_pred - y[i]) * x[i]
    
    # Calculate the R-squared value
    y_mean = np.mean(y)
    ssr = np.sum((y_pred - y_mean) ** 2)
    sst = np.sum((y - y_mean) ** 2)
    r_squared = ssr / sst
    
    return b0, b1, loss, r_squared

# Analytical solution
b0_analytical, b1_analytical, sse_analytical, r_squared_analytical = linear_regression_analytical(x, y)
print("Analytical solution:")
print("b0:", b0_analytical)
print("b1:", b1_analytical)
print("SSE:", sse_analytical)
print("R-squared:", r_squared_analytical)

# Full-batch gradient descent
b0_full_batch, b1_full_batch, sse_full_batch, r_squared_full_batch = linear_regression_full_batch(x, y)
print("\nFull-batch gradient descent:")
print("b0:", b0_full_batch)
print("b1:", b1_full_batch)
print("SSE:", sse_full_batch)
print("R-squared:", r_squared_full_batch)

# Stochastic gradient descent
b0_stochastic, b1_stochastic, sse_stochastic, r_squared_stochastic = linear_regression_stochastic(x, y)
print("\nStochastic gradient descent:")
print("b0:", b0_stochastic)
print("b1:", b1_stochastic)
print("SSE:", sse_stochastic)
print("R-squared:", r_squared_stochastic)
