import numpy as np
import matplotlib.pyplot as plt
from sklearn.decomposition import PCA

# Define the data matrix
data = np.array([(2, 1), (3, 4), (5, 0), (7, 6), (9, 2)])

# Plot the original data points
plt.figure(figsize=(8, 4))
plt.subplot(1, 2, 1)
plt.scatter(data[:, 0], data[:, 1], c='b', label='Original Data')
plt.xlabel('Feature 1')
plt.ylabel('Feature 2')
plt.title('Original Data')

# Perform PCA
pca = PCA(n_components=2)
transformed_data = pca.fit_transform(data)

# Plot the transformed data points
plt.subplot(1, 2, 2)
plt.scatter(transformed_data[:, 0], transformed_data[:, 1], c='r', label='Transformed Data')
plt.xlabel('Principal Component 1')
plt.ylabel('Principal Component 2')
plt.title('Transformed Data')

# Add legend
plt.legend()
plt.tight_layout()

# Show plot
plt.show()
