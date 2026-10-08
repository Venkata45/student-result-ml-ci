import json
import pickle

from sklearn.model_selection import train_test_split
from sklearn.tree import DecisionTreeClassifier
from sklearn.metrics import accuracy_score

# Student data
X = [
    [80, 90],
    [70, 80],
    [60, 70],
    [50, 60],
    [90, 95],
    [85, 90],
    [40, 50],
    [30, 40],
    [75, 85],
    [65, 75],
    [95, 98],
    [55, 65],
    [45, 55],
    [88, 92],
    [78, 82],
    [35, 45],
    [68, 72],
    [82, 88],
    [92, 96],
    [58, 62]
]

# 1 = Pass, 0 = Fail
y = [
    1, 1, 1, 0, 1,
    1, 0, 0, 1, 1,
    1, 0, 1, 1, 1,
    0, 1, 1, 1, 1
]

# Split data
X_train, X_test, y_train, y_test = train_test_split(
    X,
    y,
    test_size=0.3,
    random_state=42
)

# Create model
model = DecisionTreeClassifier(random_state=42)

# Train model
model.fit(X_train, y_train)

# Predict
y_pred = model.predict(X_test)

# Calculate accuracy
accuracy = accuracy_score(y_test, y_pred)

print("Model Accuracy:", round(accuracy, 4))

# Save model
with open("student_result_model.pkl", "wb") as file:
    pickle.dump(model, file)

# Save metrics
metrics = {
    "accuracy": float(accuracy),
    "training_records": len(X_train),
    "testing_records": len(X_test)
}

with open("metrics.json", "w") as file:
    json.dump(metrics, file, indent=4)

print("Model saved successfully.")
print("metrics.json created successfully.")
