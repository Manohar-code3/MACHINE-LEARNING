from sklearn.datasets import load_iris
from sklearn.model_selection import train_test_split
from sklearn.naive_bayes import GaussianNB
from sklearn.metrics import confusion_matrix, accuracy_score, classification_report

# Step 1: Load the Iris dataset
iris = load_iris()
X = iris.data
y = iris.target

# Step 2: No preprocessing needed for the Iris dataset

# Step 3: Split the data into training and testing sets
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Step 4: Train the Naïve Bayes classifier
classifier = GaussianNB()
classifier.fit(X_train, y_train)

# Step 5: Make predictions on the testing data
y_pred = classifier.predict(X_test)

# Step 6: Evaluate the classifier's performance
# Confusion matrix
conf_matrix = confusion_matrix(y_test, y_pred)
print("Confusion Matrix:")
print(conf_matrix)

# Accuracy
accuracy = accuracy_score(y_test, y_pred)
print("Accuracy:", accuracy)

# Other metrics (precision, recall, F1-score)
print("Classification Report:")
print(classification_report(y_test, y_pred, target_names=iris.target_names))
