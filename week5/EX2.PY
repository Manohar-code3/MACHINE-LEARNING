from sklearn.datasets import load_iris
import numpy as np

# Step 1: Import the Iris dataset
iris = load_iris()
data = iris.data

# Step 2: Extract the features (attributes)
features = data[:, :]

# Step 3: Function to calculate Euclidean distance between two data points
def euclidean_distance(point1, point2):
    """
    Calculate the Euclidean distance between two data points.

    Parameters:
    point1 (numpy.ndarray): 1D array representing the features of the first data point.
    point2 (numpy.ndarray): 1D array representing the features of the second data point.

    Returns:
    float: The Euclidean distance between the two data points.
    """
    return np.sqrt(np.sum((point1 - point2) ** 2))

# Step 4: Calculate the Euclidean Distance Matrix
num_samples = len(features)
distance_matrix = np.zeros((num_samples, num_samples))

for i in range(num_samples):
    for j in range(num_samples):
        if i == j:
            distance_matrix[i][j] = 0.0
        else:
            distance_matrix[i][j] = euclidean_distance(features[i], features[j])

# Print the Euclidean Distance Matrix
print("Euclidean Distance Matrix:")
print(distance_matrix)
