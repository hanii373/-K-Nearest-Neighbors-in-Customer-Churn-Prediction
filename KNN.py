import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.neighbors import KNeighborsClassifier
from sklearn.metrics import confusion_matrix, classification_report

# 1. Load dataset
# Example columns:
# Tenure, MonthlyCharges, ContractType, Churn
data = pd.read_csv("customer_churn.csv")

# Encode categorical variable
data["ContractType"] = data["ContractType"].map({
    "Month-to-month": 0,
    "One year": 1,
    "Two year": 2
})

# Encode target label
data["Churn"] = data["Churn"].map({"No": 0, "Yes": 1})

# 2. Split features and target
X = data.drop("Churn", axis=1)
y = data["Churn"]

# 3. Train-test split
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42, stratify=y
)

# 4. Feature scaling
scaler = StandardScaler()
X_train_scaled = scaler.fit_transform(X_train)
X_test_scaled = scaler.transform(X_test)

# 5. Train KNN model
knn = KNeighborsClassifier(n_neighbors=5, metric="euclidean")
knn.fit(X_train_scaled, y_train)

# 6. Prediction
y_pred = knn.predict(X_test_scaled)

# 7. Evaluation
print("Confusion Matrix:")
print(confusion_matrix(y_test, y_pred))

print("\nClassification Report:")
print(classification_report(y_test, y_pred))
