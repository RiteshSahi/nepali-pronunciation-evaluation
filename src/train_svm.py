import pandas as pd

from sklearn.model_selection import train_test_split
from sklearn.svm import SVC

from sklearn.metrics import (
    accuracy_score,
    classification_report,
    confusion_matrix
)

import joblib


# -----------------------------
# Load dataset
# -----------------------------

data = pd.read_csv(
    "../dataset/pronunciation_dataset.csv"
)


print(data)


# -----------------------------
# Features and labels
# -----------------------------

X = data[
    [
        "dtw",
        "wer",
        "cer"
    ]
]


y = data["label"]



# -----------------------------
# Split data
# -----------------------------

X_train, X_test, y_train, y_test = train_test_split(

    X,
    y,
    test_size=0.3,
    random_state=42,
    stratify=y

)



# -----------------------------
# Train SVM
# -----------------------------

model = SVC(
    kernel="rbf",
    C=1,
    gamma="scale"
)


model.fit(
    X_train,
    y_train
)



# -----------------------------
# Evaluation
# -----------------------------

prediction = model.predict(
    X_test
)


print("\nAccuracy:")
print(
    accuracy_score(
        y_test,
        prediction
    )
)


print("\nClassification Report:")
print(
    classification_report(
        y_test,
        prediction
    )
)


print("\nConfusion Matrix:")
print(
    confusion_matrix(
        y_test,
        prediction
    )
)



# -----------------------------
# Save model
# -----------------------------

joblib.dump(
    model,
    "../models/svm_pronunciation_model.pkl"
)


print("\nModel Saved Successfully")