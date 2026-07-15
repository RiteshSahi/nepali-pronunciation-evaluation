import pandas as pd
import joblib

from sklearn.model_selection import train_test_split
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.svm import SVC
from sklearn.metrics import (
    accuracy_score,
    classification_report,
    confusion_matrix
)

# -----------------------------
# Load Dataset
# -----------------------------
data = pd.read_csv("../dataset/pronunciation_dataset.csv")

print(data)

# -----------------------------
# Features and Labels
# -----------------------------
X = data[["dtw", "wer", "cer"]]
y = data["label"]

# -----------------------------
# Split Dataset
# -----------------------------
X_train, X_test, y_train, y_test = train_test_split(
    X,
    y,
    test_size=0.30,
    random_state=42,
    stratify=y
)

# -----------------------------
# SVM Pipeline
# -----------------------------
model = Pipeline([
    ("scaler", StandardScaler()),
    ("svm", SVC(
        kernel="rbf",
        C=1.0,
        gamma="scale"
    ))
])

# -----------------------------
# Train
# -----------------------------
model.fit(X_train, y_train)

# -----------------------------
# Predict
# -----------------------------
prediction = model.predict(X_test)

# -----------------------------
# Evaluation
# -----------------------------
print("\nAccuracy:")
print(accuracy_score(y_test, prediction))

print("\nClassification Report:")
print(classification_report(y_test, prediction))

print("\nConfusion Matrix:")
print(confusion_matrix(y_test, prediction))

# -----------------------------
# Save Model
# -----------------------------
joblib.dump(
    model,
    "../models/svm_pronunciation_model.pkl"
)

print("\nModel Saved Successfully")