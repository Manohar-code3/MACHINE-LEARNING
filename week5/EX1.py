import numpy as np

def euclidean_distance(x, y):
    """
    Calculate the Euclidean distance between two points in n-dimensional space.

    Parameters:
    x (numpy.ndarray): 1D array representing the coordinates of the first point.
    y (numpy.ndarray): 1D array representing the coordinates of the second point.

    Returns:
    float: The Euclidean distance between the two points.
    """
    return np.sqrt(np.sum((x - y) ** 2))

def manhattan_distance(x, y):
    """
    Calculate the Manhattan distance between two points in n-dimensional space.

    Parameters:
    x (numpy.ndarray): 1D array representing the coordinates of the first point.
    y (numpy.ndarray): 1D array representing the coordinates of the second point.

    Returns:
    float: The Manhattan distance between the two points.
    """
    return np.sum(np.abs(x - y))

def minkowski_distance(x, y, p):
    """
    Calculate the Minkowski distance between two points in n-dimensional space.

    Parameters:
    x (numpy.ndarray): 1D array representing the coordinates of the first point.
    y (numpy.ndarray): 1D array representing the coordinates of the second point.
    p (int): The parameter defining the norm. When p=1, Manhattan distance is calculated; when p=2, Euclidean distance is calculated.

    Returns:
    float: The Minkowski distance between the two points.
    """
    return np.sum(np.abs(x - y) ** p) ** (1/p)

# Take input from the user for the coordinates of the first point
x = np.array([float(input("Enter x coordinate of first point: ")),
              float(input("Enter y coordinate of first point: ")),
              float(input("Enter z coordinate of first point: "))])

# Take input from the user for the coordinates of the second point
y = np.array([float(input("Enter x coordinate of second point: ")),
              float(input("Enter y coordinate of second point: ")),
              float(input("Enter z coordinate of second point: "))])

# Calculate distances using different metrics
euclidean_dist = euclidean_distance(x, y)
manhattan_dist = manhattan_distance(x, y)
minkowski_dist = minkowski_distance(x, y, p=3)

# Print the results
print("Euclidean distance:", euclidean_dist)
print("Manhattan distance:", manhattan_dist)
print("Minkowski distance (p=3):", minkowski_dist)
