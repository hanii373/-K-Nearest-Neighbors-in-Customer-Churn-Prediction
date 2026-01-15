import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.impute import SimpleImputer
from sklearn.neighbors import KNeighborsClassifier
from sklearn.metrics import accuracy_score, classification_report
from sklearn.decomposition import PCA


#  Load dataset
data = pd.read_csv("customer_churn.csv")

# handling missing values
num_cols = ["Tenure", "MonthlyCharges"]
cat_cols = ["ContractType"]

num_imputer = SimpleImputer(strategy="median")
data[num_cols] = num_imputer.fit_transform(data[num_cols])

cat_imputer = SimpleImputer(strategy="most_frequent")
data[cat_cols] = cat_imputer.fit_transform(data[cat_cols])
# Encode categorical variable
data["ContractType"] = data["ContractType"].map({
    "Month-to-month": 0,
    "One year": 1,
    "Two year": 2
})

# Encode target label
data["Churn"] = data["Churn"].map({"No": 0, "Yes": 1})

# Split features and target
X = data.drop("Churn", axis=1)
y = data["Churn"]

# 3. Train-test split
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42, stratify=y
)

#  Feature scaling
scaler = StandardScaler()
X_train_scaled = scaler.fit_transform(X_train)
X_test_scaled = scaler.transform(X_test)

# Distance Functions
def euclidean_distance(a, b):
    return np.sqrt(np.sum((a - b) ** 2))

def manhattan_distance(a, b):
    return np.sum(np.abs(a - b))

# 5. Train KNN model
k_values = range(1, 21)
euclidean_accuracies = []
manhattan_accuracies = []

for k in k_values:
    knn_eu = KNeighborsClassifier(n_neighbors=k, metric="euclidean")
    knn_eu.fit(X_train_scaled, y_train)
    y_pred_eu = knn_eu.predict(X_test_scaled)
    euclidean_accuracies.append(accuracy_score(y_test, y_pred_eu))

    knn_man = KNeighborsClassifier(n_neighbors=k, metric="manhattan")
    knn_man.fit(X_train_scaled, y_train)
    y_pred_man = knn_man.predict(X_test_scaled)
    manhattan_accuracies.append(accuracy_score(y_test, y_pred_man))

# Visualizations
plt.figure(figsize=(8, 5))
plt.plot(k_values, euclidean_accuracies, label="Euclidean", marker="o")
plt.plot(k_values, manhattan_accuracies, label="Manhattan", marker="s")
plt.xlabel("Number of Neighbors (k)")
plt.ylabel("Accuracy")
plt.title("KNN Accuracy vs. k")
plt.legend()
plt.grid()
plt.show()

# Final model evaluation 
best_k = k_values[np.argmax(euclidean_accuracies)]
print("Best k:", best_k)

knn_final = KNeighborsClassifier(n_neighbors=best_k)
knn_final.fit(X_train_scaled, y_train)
y_final_pred = knn_final.predict(X_test_scaled)

print("\nClassification Report:")
print(classification_report(y_test, y_final_pred))
