import matplotlib.pyplot as plt
from sklearn.datasets import load_iris
import numpy as np

# Step 1: Import the Iris dataset
iris = load_iris()
data = iris.data
target = iris.target

# Step 2: Select samples belonging to any two output classes
class1_index = np.where(target == 0)[0]  # Indices of samples belonging to class 0
class2_index = np.where(target == 1)[0]  # Indices of samples belonging to class 1

# Randomly select 25 samples from each class
class1_samples = np.random.choice(class1_index, size=25, replace=False)
class2_samples = np.random.choice(class2_index, size=25, replace=False)

# Combine the selected samples into a new dataset
new_data = np.concatenate((data[class1_samples], data[class2_samples]), axis=0)
new_target = np.concatenate((target[class1_samples], target[class2_samples]))

# Step 3: Choose two input attributes for the scatter plot
attribute1_index = 0  # Index of the first attribute (change as needed)
attribute2_index = 1  # Index of the second attribute (change as needed)

# Step 4: Plot the scatter plot
plt.figure(figsize=(8, 6))
plt.scatter(new_data[new_target == 0, attribute1_index], new_data[new_target == 0, attribute2_index], label='Class 0')
plt.scatter(new_data[new_target == 1, attribute1_index], new_data[new_target == 1, attribute2_index], label='Class 1')
plt.xlabel(f'Attribute {attribute1_index + 1}')
plt.ylabel(f'Attribute {attribute2_index + 1}')
plt.title('Scatter Plot of Two Input Attributes')
plt.legend()
plt.grid(True)
plt.show()

# Step 5: Examine the scatter plot and find a line that can separate the samples of the two classes
