import numpy as np
import matplotlib.pyplot as plt
import pandas as pd

# Load the Iris dataset
file_path = "E:\MACHINE LEARNING\LAB ML\week8\iris.csv"
iris_df = pd.read_csv(file_path)

# Extract features and labels
X = iris_df.iloc[:, :-1].values  # Features
y = iris_df.iloc[:, -1].values   # Labels

# Standardize the features
X_mean = np.mean(X, axis=0)
X_std = np.std(X, axis=0)
X = (X - X_mean) / X_std

# Compute the covariance matrix
cov_matrix = np.cov(X.T)

# Compute eigenvalues and eigenvectors
eigenvalues, eigenvectors = np.linalg.eig(cov_matrix)

# Sort eigenvalues and eigenvectors in descending order
idx = eigenvalues.argsort()[::-1]
eigenvalues = eigenvalues[idx]
eigenvectors = eigenvectors[:, idx]

# Choose the top k eigenvectors corresponding to the largest eigenvalues
k = 2
top_eigenvectors = eigenvectors[:, :k]

# Project the data onto the new subspace
transformed_data = np.dot(X, top_eigenvectors)

# Plot the scatter plot for samples in the transformed domain
plt.figure(figsize=(8, 6))
for label in np.unique(y):
    plt.scatter(transformed_data[y == label, 0], transformed_data[y == label, 1], label=label)
plt.xlabel('Principal Component 1')
plt.ylabel('Principal Component 2')
plt.title('PCA: Iris Dataset')
plt.legend()
plt.show()
