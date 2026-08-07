#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""
Overall Visualization for Nepali Pronunciation Evaluation Project

Generates:
1. Performance comparison graph
2. Feature importance graph
3. Dataset class distribution
4. Confusion matrix

All metrics and the feature-importance graph are computed live from
the loaded model and dataset -- nothing is hardcoded -- so the plots
always reflect whichever model/fold you point this script at.
"""

import os
import joblib
import pandas as pd
import matplotlib.pyplot as plt

from sklearn.inspection import permutation_importance
from sklearn.metrics import (
    ConfusionMatrixDisplay,
    accuracy_score,
    f1_score,
    precision_score,
    recall_score,
)


# ---------------------------------------------------
# Which fold / gender to visualize
# ---------------------------------------------------
# The training script now saves one model PER LOUO FOLD, e.g.:
#   ../models/male_model_fold1_test_user1.pkl
#   ../models/male_model_fold2_test_user2.pkl
#   ../models/male_model_fold3_test_user3.pkl
# Pick the fold whose held-out test user you want to visualize.

GENDER = "male"
FOLD_NUM = 1
TEST_USER = "user1"

DATASET_PATH = f"../dataset/{GENDER}/{TEST_USER}_pronunciation_dataset.csv"
MODEL_PATH = f"../models/{GENDER}_model_fold{FOLD_NUM}_test_{TEST_USER}.pkl"

PLOT_DIR = "../plots"

os.makedirs(PLOT_DIR, exist_ok=True)

FEATURES = [
    "dtw",
    "duration",
    "wer",
    "cer",
    "zcr"
]


# ---------------------------------------------------
# Load Dataset and Model
# ---------------------------------------------------

print(f"Loading dataset: {DATASET_PATH}")

if not os.path.exists(DATASET_PATH):
    raise FileNotFoundError(f"Dataset not found: {DATASET_PATH}")

df = pd.read_csv(DATASET_PATH)


print(f"Loading model: {MODEL_PATH}")

if not os.path.exists(MODEL_PATH):
    raise FileNotFoundError(
        f"Model not found: {MODEL_PATH}\n"
        "Did you mean a different FOLD_NUM / TEST_USER / GENDER? "
        "Available models are in ../models/"
    )

model = joblib.load(MODEL_PATH)

print(f"Fold {FOLD_NUM}  |  Test user: {TEST_USER}  |  Gender: {GENDER}")


X = df[FEATURES]
y = df["label"]

y_pred = model.predict(X)


# ===================================================
# 1. Model Performance Graph (computed live)
# ===================================================

print("Creating performance graph...")

metrics = {
    "Accuracy": accuracy_score(y, y_pred) * 100,
    "Precision": precision_score(y, y_pred, pos_label="Good", zero_division=0) * 100,
    "Recall": recall_score(y, y_pred, pos_label="Good", zero_division=0) * 100,
    "F1-score": f1_score(y, y_pred, pos_label="Good", zero_division=0) * 100,
}


plt.figure(figsize=(8, 5))

plt.bar(
    metrics.keys(),
    metrics.values()
)

plt.ylim(0, 100)
plt.ylabel("Score (%)")
plt.xlabel("Evaluation Metrics")
plt.title(f"SVM Model Performance (Fold {FOLD_NUM}, Test: {TEST_USER})")

for i, value in enumerate(metrics.values()):
    plt.text(
        i,
        value + 1,
        f"{value:.2f}%",
        ha="center"
    )

plt.tight_layout()
plt.savefig(f"{PLOT_DIR}/model_performance_fold{FOLD_NUM}.png", dpi=300)
plt.show()


# ===================================================
# 2. Feature Importance (computed live via permutation importance)
# ===================================================

print("Creating feature importance graph...")

perm_result = permutation_importance(
    model,
    X,
    y,
    n_repeats=30,
    random_state=42,
    n_jobs=-1,
)

importance_values = dict(
    sorted(
        zip(FEATURES, perm_result.importances_mean),
        key=lambda item: item[1],
        reverse=True,
    )
)

plt.figure(figsize=(8, 5))

plt.bar(
    [name.upper() for name in importance_values.keys()],
    importance_values.values()
)

plt.ylabel("Permutation Importance")
plt.xlabel("Features")
plt.title(f"Feature Importance (Fold {FOLD_NUM}, Test: {TEST_USER})")
plt.xticks(rotation=45)
plt.tight_layout()
plt.savefig(f"{PLOT_DIR}/feature_importance_fold{FOLD_NUM}.png", dpi=300)
plt.show()


# ===================================================
# 3. Dataset Distribution
# ===================================================

print("Creating dataset distribution graph...")

class_count = df["label"].value_counts()

plt.figure(figsize=(6, 5))

plt.bar(
    class_count.index,
    class_count.values
)

plt.xlabel("Class")
plt.ylabel("Number of Samples")
plt.title(f"Dataset Class Distribution ({TEST_USER})")

for i, value in enumerate(class_count.values):
    plt.text(
        i,
        value + 1,
        str(value),
        ha="center"
    )

plt.tight_layout()
plt.savefig(f"{PLOT_DIR}/dataset_distribution_{TEST_USER}.png", dpi=300)
plt.show()


# ===================================================
# 4. Confusion Matrix
# ===================================================

print("Creating confusion matrix...")

disp = ConfusionMatrixDisplay.from_predictions(
    y,
    y_pred
)

plt.title(f"Confusion Matrix - Fold {FOLD_NUM} (Test: {TEST_USER})")
plt.tight_layout()
plt.savefig(f"{PLOT_DIR}/confusion_matrix_fold{FOLD_NUM}.png", dpi=300)
plt.show()


print("\n===================================")
print("ALL PROJECT GRAPHS GENERATED")
print(f"Saved in: {PLOT_DIR}")
print("===================================")