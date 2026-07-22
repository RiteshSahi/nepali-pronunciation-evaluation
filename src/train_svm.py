import os
import pandas as pd
import joblib

from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.svm import SVC
from sklearn.metrics import (
    accuracy_score,
    classification_report,
    confusion_matrix
)

# ======================================================
# Dataset Paths
# ======================================================

TRAIN_FILES = []

for user in ["user1", "user2"]:

    file = f"../dataset/{user}_pronunciation_dataset.csv"

    if os.path.exists(file):
        TRAIN_FILES.append(file)

TEST_FILE = "../dataset/user3_pronunciation_dataset.csv"

# ======================================================
# Check Training Files
# ======================================================

if len(TRAIN_FILES) == 0:
    print("No training datasets found.")
    exit()

print("=" * 60)
print("Training Files")
print("=" * 60)

for file in TRAIN_FILES:
    print(file)

if not os.path.exists(TEST_FILE):
    print(f"\nTesting dataset not found:\n{TEST_FILE}")
    exit()

# ======================================================
# Load Dataset
# ======================================================

train_data = pd.concat(
    [pd.read_csv(file) for file in TRAIN_FILES],
    ignore_index=True
)

test_data = pd.read_csv(TEST_FILE)

print("\n" + "=" * 60)
print("Training Dataset")
print("=" * 60)
print(train_data)

print("\nTraining Samples :", len(train_data))

print("\nTraining Class Distribution")
print(train_data["label"].value_counts())

print("\n" + "=" * 60)
print("Testing Dataset")
print("=" * 60)
print(test_data)

print("\nTesting Samples :", len(test_data))

print("\nTesting Class Distribution")
print(test_data["label"].value_counts())

# ======================================================
# Features
# ======================================================

FEATURES = [
    "dtw",
    "duration",
    "zcr",
    "wer",
    "cer"
]

X_train = train_data[FEATURES]
y_train = train_data["label"]

X_test = test_data[FEATURES]
y_test = test_data["label"]

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
            probability=True,
            random_state=42
        )
    )
])

# ======================================================
# Train
# ======================================================

print("\nTraining SVM...\n")

model.fit(
    X_train,
    y_train
)

# ======================================================
# Predict
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

print("\n" + "=" * 60)
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
# Sample Predictions
# ======================================================

print("\nSample Predictions")
print("=" * 60)

for i in range(len(X_test)):

    confidence = max(probability[i]) * 100

    print(
        f"Voice {int(test_data.iloc[i]['voice']):3d} | "
        f"Actual : {y_test.iloc[i]:4s} | "
        f"Predicted : {prediction[i]:4s} | "
        f"Confidence : {confidence:.2f}%"
    )

# ======================================================
# Save Model
# ======================================================

os.makedirs(
    "../models",
    exist_ok=True
)

joblib.dump(
    model,
    "../models/svm_pronunciation_model.pkl"
)

print("\nModel saved successfully.")
print("../models/svm_pronunciation_model.pkl")