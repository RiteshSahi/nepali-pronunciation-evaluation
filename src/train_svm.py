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

# ======================================================
# Load Dataset
# ======================================================

data = pd.read_csv("../dataset/pronunciation_dataset.csv")

print("=" * 60)
print("Dataset")
print("=" * 60)
print(data)

print("\nTotal Samples :", len(data))
print("\nClass Distribution")
print(data["label"].value_counts())

# ======================================================
# Features
# ======================================================

X = data[["dtw", "wer", "cer"]]
y = data["label"]

# ======================================================
# Train / Test Split
# ======================================================

X_train, X_test, y_train, y_test = train_test_split(
    X,
    y,
    test_size=0.20,
    random_state=42,
    stratify=y
)

# ======================================================
# Build Model
# ======================================================

model = Pipeline([
    (
        "scaler",
        StandardScaler()
    ),
    (
        "svm",
        SVC(
            kernel="rbf",
            C=1.0,
            gamma="scale",
            probability=True
        )
    )
])

# ======================================================
# Train
# ======================================================

print("\nTraining SVM...\n")

model.fit(X_train, y_train)

# ======================================================
# Prediction
# ======================================================

prediction = model.predict(X_test)

probability = model.predict_proba(X_test)

# ======================================================
# Evaluation
# ======================================================

accuracy = accuracy_score(
    y_test,
    prediction
)

print("=" * 60)
print("Model Evaluation")
print("=" * 60)

print(f"\nAccuracy : {accuracy:.4f}")

print("\nClassification Report\n")

print(
    classification_report(
        y_test,
        prediction,
        zero_division=0
    )
)

print("Confusion Matrix\n")

print(
    confusion_matrix(
        y_test,
        prediction
    )
)

# ======================================================
# Show Confidence
# ======================================================

print("\nSample Predictions\n")

for i in range(min(10, len(X_test))):

    confidence = max(probability[i]) * 100

    print(
        f"Prediction : {prediction[i]:5s} "
        f"Confidence : {confidence:.2f}%"
    )

# ======================================================
# Save Model
# ======================================================

joblib.dump(
    model,
    "../models/svm_pronunciation_model.pkl"
)

print("\nModel saved successfully.")