import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import accuracy_score, confusion_matrix, classification_report, ConfusionMatrixDisplay

# 📌 Load the dataset
column_names = [
    "class", "alcohol", "malic_acid", "ash", "alcalinity_of_ash", "magnesium",
    "total_phenols", "flavanoids", "nonflavanoid_phenols", "proanthocyanins",
    "color_intensity", "hue", "od280/od315_of_diluted_wines", "proline"
]

df = pd.read_csv("wine.data", names=column_names)

# 📌 Separate features and labels
X = df.drop(columns=["class"])
y = df["class"] - 1  # Convert classes from 1-3 to 0-2 for easier indexing

# 📌 Split the dataset into training (80%) and testing (20%) sets
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42, stratify=y)

# 📌 Normalize the data using StandardScaler
scaler = StandardScaler()
X_train = scaler.fit_transform(X_train)
X_test = scaler.transform(X_test)

# 📌 Distance Calculation Functions
def euclidean_distance(x1, x2):
    return np.sqrt(np.sum((x1 - x2) ** 2))

def manhattan_distance(x1, x2):
    return np.sum(np.abs(x1 - x2))

# 📌 Custom k-NN Classifier (Without sklearn)
class KNN:
    def __init__(self, k=3, distance_metric="euclidean"):
        self.k = k
        self.distance_metric = distance_metric

    def fit(self, X_train, y_train):
        self.X_train = X_train
        self.y_train = np.array(y_train)  # Convert Pandas Series to NumPy array ✅

    def predict(self, X_test):
        return np.array([self._predict(x) for x in X_test])

    def _predict(self, x):
        # Calculate distances based on selected metric
        if self.distance_metric == "euclidean":
            distances = [euclidean_distance(x, x_train) for x_train in self.X_train]
        elif self.distance_metric == "manhattan":
            distances = [manhattan_distance(x, x_train) for x_train in self.X_train]
        else:
            raise ValueError("Invalid distance metric!")

        # Get indices of the k nearest neighbors
        k_indices = np.argsort(distances)[:self.k]
        k_nearest_labels = [self.y_train[i] for i in k_indices]  # ✅ Fixed indexing issue

        # Return the most common class among the k neighbors
        return np.bincount(k_nearest_labels).argmax()

# 📌 Test the model with different values of K
k_values = [1, 3, 5, 7, 9]
accuracy_scores = []

for k in k_values:
    model = KNN(k=k, distance_metric="euclidean")
    model.fit(X_train, y_train)
    y_pred = model.predict(X_test)

    acc = accuracy_score(y_test, y_pred)
    accuracy_scores.append(acc)
    print(f"Accuracy for K={k}: {acc:.4f}")

# 📌 Plot Accuracy vs. K Values
plt.figure(figsize=(8,5))
plt.plot(k_values, accuracy_scores, marker='o', linestyle='dashed', color='b')
plt.xlabel("K Value")
plt.ylabel("Accuracy")
plt.title("Model Performance for Different K Values")
plt.grid()
plt.show()

# 📌 Find the best K value and evaluate the model
best_k = k_values[np.argmax(accuracy_scores)]
model = KNN(k=best_k, distance_metric="euclidean")
model.fit(X_train, y_train)
y_pred = model.predict(X_test)

# 📌 Display the Confusion Matrix
cm = confusion_matrix(y_test, y_pred)
disp = ConfusionMatrixDisplay(confusion_matrix=cm)
disp.plot()
plt.show()

# 📌 Print Classification Report
print(f"\nClassification Report for Best K={best_k}:")
print(classification_report(y_test, y_pred))
