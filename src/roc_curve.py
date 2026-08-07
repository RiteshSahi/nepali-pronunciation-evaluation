#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""
ROC Curve for Nepali Pronunciation Evaluation SVM Model
"""

import os
import joblib
import pandas as pd
import matplotlib.pyplot as plt

from sklearn.metrics import (
    roc_curve,
    auc,
    accuracy_score
)


# -----------------------------
# Paths
# -----------------------------

DATASET_PATH = "../dataset/male/user1_pronunciation_dataset.csv"
MODEL_PATH = "../models/male_model.pkl"

PLOT_DIR = "../plots"

os.makedirs(PLOT_DIR, exist_ok=True)


# -----------------------------
# Load dataset and model
# -----------------------------

print("Loading dataset...")

df = pd.read_csv(DATASET_PATH)


print("Loading trained model...")

model = joblib.load(MODEL_PATH)


FEATURES = [
    "dtw",
    "duration",
    "wer",
    "cer",
    "zcr"
]


X = df[FEATURES]

y = df["label"]


# -----------------------------
# Convert labels
# Good = 1
# Bad = 0
# -----------------------------

y_binary = (
    y == "Good"
).astype(int)


# -----------------------------
# Prediction probability
# -----------------------------

y_probability = model.predict_proba(X)[:,1]


# -----------------------------
# ROC calculation
# -----------------------------

fpr, tpr, thresholds = roc_curve(
    y_binary,
    y_probability
)


roc_auc = auc(
    fpr,
    tpr
)


# -----------------------------
# Plot ROC Curve
# -----------------------------

plt.figure(figsize=(7,5))


plt.plot(
    fpr,
    tpr,
    label=f"AUC = {roc_auc:.3f}"
)


plt.plot(
    [0,1],
    [0,1],
    linestyle="--"
)


plt.xlabel(
    "False Positive Rate"
)


plt.ylabel(
    "True Positive Rate"
)


plt.title(
    "ROC Curve - SVM Pronunciation Evaluation"
)


plt.legend(
    loc="lower right"
)


plt.grid()


plt.tight_layout()


plt.savefig(
    f"{PLOT_DIR}/roc_curve.png",
    dpi=300
)


plt.show()


print("==============================")
print("ROC Curve Generated")
print(f"AUC Score: {roc_auc:.3f}")
print("==============================")