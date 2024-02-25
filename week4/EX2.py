# Import necessary libraries
import numpy as np
import pandas as pd
from sklearn import datasets, linear_model
from sklearn.model_selection import train_test_split

# Load the Boston Housing dataset
boston_data = datasets.load_boston()
df = pd.DataFrame(data=boston_data['data'])

# Split the data into training and testing sets
X = boston_data.data
Y = boston_data.target
X_train, X_test, Y_train, Y_test = train_test_split(X, Y, test_size=0.4, random_state=1)

# Analytical solution
def linear_regression_analytical(X, Y):
    n = len(X)
    X_mean = np.mean(X)
    Y_mean = np.mean(Y)
    XY_mean = np.mean(X * Y)
    X_squared_mean = np.mean(X ** 2)
    
    # Calculate the coefficients
    b1 = (XY_mean - X_mean * Y_mean) / (X_squared_mean - X_mean ** 2)
    b0 = Y_mean - b1 * X_mean
    
    # Calculate the predicted Y values
    Y_pred = b0 + b1 * X
    
    # Calculate the sum of squared errors (SSE)
    SSE = np.sum((Y - Y_pred) ** 2)
    
    # Calculate the R-squared value
    SSR = np.sum((Y_pred - Y_mean) ** 2)
    SST = np.sum((Y - Y_mean) ** 2)
    R_squared = SSR / SST
    
    return b0, b1, SSE, R_squared

# Full-batch gradient descent
def linear_regression_full_batch(X, Y, learning_rate=0.01, epochs=1000, tolerance=1e-6):
    n = len(X)
    b0 = 0
    b1 = 0
    prev_loss = float('inf')
    
    for epoch in range(epochs):
        Y_pred = b0 + b1 * X
        loss = np.sum((Y_pred - Y) ** 2) / (2 * n)
        
        if abs(prev_loss - loss) < tolerance:
            break
        
        prev_loss = loss
        
        # Update the coefficients using gradient descent
        b0 -= learning_rate * np.sum(Y_pred - Y) / n
        b1 -= learning_rate * np.sum((Y_pred - Y) * X) / n
    
    # Calculate the R-squared value
    Y_mean = np.mean(Y)
    SSR = np.sum((Y_pred - Y_mean) ** 2)
    SST = np.sum((Y - Y_mean) ** 2)
    R_squared = SSR / SST
    
    return b0, b1, loss, R_squared

# Stochastic gradient descent
def linear_regression_stochastic(X, Y, learning_rate=0.01, epochs=1000, tolerance=1e-6):
    n = len(X)
    b0 = 0
    b1 = 0
    prev_loss = float('inf')
    
    for epoch in range(epochs):
        for i in range(n):
            Y_pred = b0 + b1 * X[i]
            loss = (Y_pred - Y[i]) ** 2 / 2
            
            if abs(prev_loss - loss) < tolerance:
                break
            
            prev_loss = loss
            
            # Update the coefficients using stochastic gradient descent
            b0 -= learning_rate * (Y_pred - Y[i])
            b1 -= learning_rate * (Y_pred - Y[i]) * X[i]
    
    # Calculate the R-squared value
    Y_mean = np.mean(Y)
    SSR = np.sum((Y_pred - Y_mean) ** 2)
    SST = np.sum((Y - Y_mean) ** 2)
    R_squared = SSR / SST
    
    return b0, b1, loss, R_squared

# Analytical solution
b0_analytical, b1_analytical, SSE_analytical, R_squared_analytical = linear_regression_analytical(X_train[:, 5], Y_train)
print("Analytical solution:")
print("b0:", b0_analytical)
print("b1:", b1_analytical)
print("SSE:", SSE_analytical)
print("R-squared:", R_squared_analytical)

# Full-batch gradient descent
b0_full_batch, b1_full_batch, SSE_full_batch, R_squared_full_batch = linear_regression_full_batch(X_train[:, 5], Y_train)
print("\nFull-batch gradient descent:")
print("b0:", b0_full_batch)
print("b1:", b1_full_batch)
print("SSE:", SSE_full_batch)
print("R-squared:", R_squared_full_batch)

# Stochastic gradient descent
b0_stochastic, b1_stochastic, SSE_stochastic, R_squared_stochastic = linear_regression_stochastic(X_train[:, 5], Y_train)
print("\nStochastic gradient descent:")
print("b0:", b0_stochastic)
print("b1:", b1_stochastic)
print("SSE:", SSE_stochastic)
print("R-squared:", R_squared_stochastic)
