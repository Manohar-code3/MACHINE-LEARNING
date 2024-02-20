import numpy as np

# Function to compute covariance matrix and correlation matrix
def compute_matrices(feature_matrix):
    # Compute the covariance matrix
    covariance_matrix = np.cov(feature_matrix)
    
    # Compute the correlation matrix
    correlation_matrix = np.corrcoef(feature_matrix)
    
    return covariance_matrix, correlation_matrix

# Function to print the matrices and insights
def print_matrices_and_insights(covariance_matrix, correlation_matrix):
    print("Covariance Matrix:")
    print(covariance_matrix)
    print("\nCorrelation Matrix:")
    print(correlation_matrix)
    
    # Insights based on covariance matrix
    print("\nInsights based on Covariance Matrix:")
    print("1. The diagonal elements of the covariance matrix represent the variance of each feature.")
    print("2. The off-diagonal elements represent the covariance between pairs of features.")
    print("3. A positive covariance indicates that the features tend to increase or decrease together.")
    print("4. A negative covariance indicates that one feature increases while the other decreases.")
    print("5. A covariance of zero indicates no linear relationship between the features.")
    
    # Insights based on correlation matrix
    print("\nInsights based on Correlation Matrix:")
    print("1. The diagonal elements of the correlation matrix represent the correlation coefficient of each feature with itself, which is always 1.")
    print("2. The off-diagonal elements represent the correlation coefficient between pairs of features.")
    print("3. The correlation coefficient ranges from -1 to 1.")
    print("4. A correlation coefficient of 1 indicates a perfect positive linear relationship.")
    print("5. A correlation coefficient of -1 indicates a perfect negative linear relationship.")
    print("6. A correlation coefficient of 0 indicates no linear relationship.")
    print("7. The correlation matrix is a normalized version of the covariance matrix, where each element is divided by the product of the standard deviations of the corresponding features.")
    
# Example usage
# Create a matrix of MxN dimensions representing the M-dimensional feature vector for N number of samples
feature_matrix = np.array([[1, 2, 3], [4, 5, 6], [7, 8, 9]])

# Compute the covariance matrix and correlation matrix
covariance_matrix, correlation_matrix = compute_matrices(feature_matrix)

# Print the matrices and insights
print_matrices_and_insights(covariance_matrix, correlation_matrix)
