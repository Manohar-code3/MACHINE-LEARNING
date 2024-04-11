import numpy as np
import pandas as pd

# Step 1: Load an existing dataset
def load_dataset(file_path):
    # Load dataset from file path
    data = pd.read_csv(file_path)
    # Assuming the last column is the target variable
    X = data.iloc[:, :-1].values  # Features
    y = data.iloc[:, -1].values   # Labels
    return X, y

# Step 2: Split the dataset into training and testing sets
def train_test_split(X, y, test_size=0.2, random_state=None):
    if random_state:
        np.random.seed(random_state)
    
    # Shuffle indices
    indices = np.arange(X.shape[0])
    np.random.shuffle(indices)
    
    # Calculate split index
    split_index = int((1 - test_size) * len(indices))
    
    # Split dataset
    X_train, X_test = X[indices[:split_index]], X[indices[split_index:]]
    y_train, y_test = y[indices[:split_index]], y[indices[split_index:]]
    
    return X_train, X_test, y_train, y_test

# Step 3: Implement the K-NN algorithm
def euclidean_distance(x1, x2):
    return np.sqrt(np.sum((x1 - x2)**2))

def knn_predict(X_train, y_train, x_test, k=3):
    distances = []
    for i in range(len(X_train)):
        dist = euclidean_distance(X_train[i], x_test)
        distances.append((dist, y_train[i]))
    
    # Sort distances and get top k neighbors
    distances = sorted(distances)[:k]
    
    # Count the class frequencies
    class_votes = {}
    for d in distances:
        class_votes[d[1]] = class_votes.get(d[1], 0) + 1
    
    # Get the class with the most votes
    predicted_class = max(class_votes, key=class_votes.get)
    
    return predicted_class

# Step 4: Test the model and calculate accuracy and confusion matrix
def calculate_accuracy(y_true, y_pred):
    correct = np.sum(y_true == y_pred)
    total = len(y_true)
    accuracy = correct / total
    return accuracy

def confusion_matrix(y_true, y_pred):
    num_classes = len(np.unique(y_true))
    matrix = np.zeros((num_classes, num_classes))
    for i in range(len(y_true)):
        matrix[y_true[i]][y_pred[i]] += 1
    return matrix

# Step 5: Examine the effect of the value of K on accuracy/performance
def k_vs_accuracy(X_train, y_train, X_test, y_test, k_values):
    accuracies = []
    for k in k_values:
        y_pred = []
        for x in X_test:
            y_pred.append(knn_predict(X_train, y_train, x, k))
        acc = calculate_accuracy(y_test, np.array(y_pred))
        accuracies.append(acc)
    return accuracies

# Main function
def main():
    # Step 1: Load dataset
    file_path = 'E:\MACHINE LEARNING\LAB ML\week7\iris.csv'
    X, y = load_dataset(file_path)

    # Step 2: Split dataset
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

    # Step 3: Test the model and calculate accuracy and confusion matrix
    k_values = [1, 3, 5, 7, 9]  # List of k values to test
    accuracies = k_vs_accuracy(X_train, y_train, X_test, y_test, k_values)

    # Step 4: Find maximum accuracy and corresponding k value
    max_accuracy = max(accuracies)
    best_k = k_values[accuracies.index(max_accuracy)]

    # Print results
    print("Accuracy for different values of k:", accuracies)
    print("Best k value for maximum accuracy:", best_k)

if __name__ == "__main__":
    main()
