#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
train_svm.py
============

AI-based Nepali Pronunciation Evaluation - SVM Training Script.

This script:
    1. Asks the user for gender (Male/Female) and resolves the
       gender-specific dataset directory and output model path.
    2. Loads pronunciation datasets for three users.
    3. Combines user2 + user3 datasets for training.
    4. Uses user1 dataset for testing.
    5. Builds a Scikit-learn Pipeline (StandardScaler + SVC).
    6. Performs hyperparameter tuning with GridSearchCV.
    7. Evaluates the best model on the test set.
    8. Computes permutation feature importance.
    9. Prints per-sample predictions with confidence.
    10. Saves the trained model pipeline using joblib.

Author: AI Assistant
Python Version: 3.9+
"""

import os
import sys
from typing import List, Tuple

import numpy as np
import pandas as pd
from sklearn.inspection import permutation_importance
from sklearn.metrics import (
    accuracy_score,
    classification_report,
    confusion_matrix,
    f1_score,
    precision_score,
    recall_score,
)
from sklearn.model_selection import GridSearchCV
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.svm import SVC

import joblib


# --------------------------------------------------------------------------- #
# Constants / Configuration
# --------------------------------------------------------------------------- #

DATASET_ROOT = "../dataset"
MODEL_DIR = "../models"

VALID_GENDERS = ("male", "female")

# NOTE: These four are resolved at runtime, once the user's gender
# selection is known (see `resolve_gender_paths()` and step 1 of
# `main()`), since male and female speakers now use separate dataset
# folders and separate trained models. They are declared here as
# module-level placeholders so every function that references them
# (e.g. `validate_files_exist()`'s error message) keeps working
# unchanged after `main()` assigns them via the `global` statement.
DATASET_DIR = None
MODEL_PATH = None
TRAIN_FILES = None
TEST_FILE = None

FEATURES = [
    "dtw",
    "duration",
    "wer",
    "cer",
    "zcr"
]
TARGET = "label"
REQUIRED_COLUMNS = ["voice"] + FEATURES + [TARGET]

PARAM_GRID = {
    "svc__kernel": ["rbf"],
    "svc__C": [0.1, 1, 10, 50, 100],
    "svc__gamma": ["scale", "auto", 0.1, 0.01, 0.001],
}

CV_FOLDS = 5
SCORING_METRIC = "accuracy"
RANDOM_STATE = 42


# --------------------------------------------------------------------------- #
# Utility / Section Printing Helpers
# --------------------------------------------------------------------------- #

def print_section(title: str) -> None:
    """
    Print a clearly formatted section heading to the console.

    Args:
        title: The title text to display for this section.
    """
    width = 70
    print("\n" + "=" * width)
    print(title.center(width))
    print("=" * width)


def print_subsection(title: str) -> None:
    """
    Print a smaller, sub-section heading to the console.

    Args:
        title: The title text to display for this sub-section.
    """
    print("\n" + "-" * 50)
    print(title)
    print("-" * 50)


# --------------------------------------------------------------------------- #
# Gender Validation
# --------------------------------------------------------------------------- #

def normalize_gender(raw_gender: str) -> str:
    """
    Normalize a user-provided gender value into one of the internal
    keys used for path resolution: "male" or "female".

    Args:
        raw_gender: Raw gender string, any case (e.g. "Male", "female").

    Returns:
        Either "male" or "female".

    Raises:
        SystemExit: If the value isn't recognized.
    """
    cleaned = raw_gender.strip().lower()

    if cleaned not in VALID_GENDERS:
        print_section("ERROR: INVALID GENDER")
        print(f"Invalid gender '{raw_gender}'. Expected 'Male' or 'Female'.")
        sys.exit(1)

    return cleaned


def resolve_gender_paths(gender: str) -> Tuple[str, str, List[str], str]:
    """
    Resolve the dataset directory, model output path, training file
    list, and testing file path for the given (already normalized)
    gender.

    Args:
        gender: Normalized gender string, "male" or "female".

    Returns:
        A tuple of (dataset_dir, model_path, train_files, test_file).
    """
    dataset_dir = os.path.join(DATASET_ROOT, gender)
    model_path = os.path.join(MODEL_DIR, f"{gender}_model.pkl")

    train_files = [
        os.path.join(dataset_dir, "user2_pronunciation_dataset.csv"),
        os.path.join(dataset_dir, "user3_pronunciation_dataset.csv"),
    ]
    test_file = os.path.join(dataset_dir, "user1_pronunciation_dataset.csv")

    return dataset_dir, model_path, train_files, test_file


# --------------------------------------------------------------------------- #
# Data Loading and Validation
# --------------------------------------------------------------------------- #

def validate_files_exist(file_paths: List[str]) -> None:
    """
    Ensure that every required dataset file exists on disk.

    Exits the program with an informative error message if any
    required file is missing.

    Args:
        file_paths: List of file paths that must exist.
    """
    missing_files = [path for path in file_paths if not os.path.isfile(path)]

    if missing_files:
        print_section("ERROR: MISSING DATASET FILE(S)")
        for path in missing_files:
            print(f"  [MISSING] {path}")
        print(
            "\nOne or more required dataset files could not be found.\n"
            "Please make sure the dataset files are placed in the "
            f"'{DATASET_DIR}' directory and try again."
        )
        sys.exit(1)


def validate_columns(df: pd.DataFrame, file_path: str) -> None:
    """
    Validate that a DataFrame contains all required columns.

    Args:
        df: The DataFrame to validate.
        file_path: The source file path (used for error messages).

    Raises:
        SystemExit: If any required column is missing.
    """
    missing_columns = [col for col in REQUIRED_COLUMNS if col not in df.columns]

    if missing_columns:
        print_section("ERROR: MISSING REQUIRED COLUMN(S)")
        print(f"File: {file_path}")
        print(f"Missing columns: {missing_columns}")
        print(f"Required columns: {REQUIRED_COLUMNS}")
        sys.exit(1)


def load_dataset(file_path: str) -> pd.DataFrame:
    """
    Load a single dataset CSV file into a DataFrame, with validation.

    Args:
        file_path: Path to the CSV dataset file.

    Returns:
        A validated pandas DataFrame.
    """
    try:
        df = pd.read_csv(file_path)
    except Exception as exc:  # noqa: BLE001 - broad catch for robust CLI tool
        print_section("ERROR: FAILED TO READ DATASET FILE")
        print(f"File: {file_path}")
        print(f"Reason: {exc}")
        sys.exit(1)

    validate_columns(df, file_path)
    return df


def load_and_combine_datasets(file_paths: List[str]) -> pd.DataFrame:
    """
    Load multiple dataset files and combine them into a single DataFrame.

    Args:
        file_paths: List of CSV file paths to load and combine.

    Returns:
        A combined pandas DataFrame containing rows from all files.
    """
    frames = [load_dataset(path) for path in file_paths]
    combined_df = pd.concat(frames, ignore_index=True)
    return combined_df


# --------------------------------------------------------------------------- #
# Model Training
# --------------------------------------------------------------------------- #

def build_pipeline() -> Pipeline:
    """
    Build the Scikit-learn Pipeline: StandardScaler -> SVC.

    Returns:
        An un-fitted sklearn Pipeline object.
    """
    pipeline = Pipeline(
        steps=[
            ("scaler", StandardScaler()),
            ("svc", SVC(probability=True, random_state=RANDOM_STATE)),
        ]
    )
    return pipeline


def train_model(
    x_train: pd.DataFrame, y_train: pd.Series
) -> GridSearchCV:
    """
    Train an SVM pipeline using GridSearchCV for hyperparameter tuning.

    Args:
        x_train: Training feature matrix.
        y_train: Training target labels.

    Returns:
        A fitted GridSearchCV object containing the best pipeline.
    """
    pipeline = build_pipeline()

    grid_search = GridSearchCV(
        estimator=pipeline,
        param_grid=PARAM_GRID,
        cv=CV_FOLDS,
        scoring=SCORING_METRIC,
        n_jobs=-1,
    )

    try:
        grid_search.fit(x_train, y_train)
    except Exception as exc:  # noqa: BLE001
        print_section("ERROR: MODEL TRAINING FAILED")
        print(f"Reason: {exc}")
        sys.exit(1)

    return grid_search


# --------------------------------------------------------------------------- #
# Evaluation
# --------------------------------------------------------------------------- #

def evaluate_model(
    model: GridSearchCV, x_test: pd.DataFrame, y_test: pd.Series
) -> np.ndarray:
    """
    Evaluate the trained model on the test dataset and print metrics.

    Args:
        model: The fitted GridSearchCV (or pipeline) model.
        x_test: Test feature matrix.
        y_test: Test target labels.

    Returns:
        Array of predicted labels for the test set.
    """
    y_pred = model.predict(x_test)

    accuracy = accuracy_score(y_test, y_pred)
    precision = precision_score(y_test, y_pred, pos_label="Good", zero_division=0)
    recall = recall_score(y_test, y_pred, pos_label="Good", zero_division=0)
    f1 = f1_score(y_test, y_pred, pos_label="Good", zero_division=0)

    print_section("MODEL EVALUATION ON TEST SET (user1)")
    print(f"Accuracy  : {accuracy * 100:.2f}%")
    print(f"Precision : {precision * 100:.2f}%")
    print(f"Recall    : {recall * 100:.2f}%")
    print(f"F1-score  : {f1 * 100:.2f}%")

    print_subsection("Classification Report")
    print(classification_report(y_test, y_pred, zero_division=0))

    print_subsection("Confusion Matrix")
    labels_order = sorted(y_test.unique())
    cm = confusion_matrix(y_test, y_pred, labels=labels_order)
    cm_df = pd.DataFrame(
        cm,
        index=[f"Actual_{lbl}" for lbl in labels_order],
        columns=[f"Predicted_{lbl}" for lbl in labels_order],
    )
    print(cm_df)

    return y_pred


def compute_feature_importance(
    model: GridSearchCV, x_test: pd.DataFrame, y_test: pd.Series
) -> pd.DataFrame:
    """
    Compute permutation feature importance on the test set.

    Args:
        model: The fitted GridSearchCV (or pipeline) model.
        x_test: Test feature matrix.
        y_test: Test target labels.

    Returns:
        A DataFrame with features sorted by importance (descending).
    """
    print_section("FEATURE IMPORTANCE (Permutation Importance)")

    try:
        result = permutation_importance(
            model,
            x_test,
            y_test,
            n_repeats=30,
            random_state=RANDOM_STATE,
            n_jobs=-1,
        )
    except Exception as exc:  # noqa: BLE001
        print(f"Failed to compute permutation importance: {exc}")
        return pd.DataFrame(columns=["feature", "importance_mean", "importance_std"])

    importance_df = pd.DataFrame(
        {
            "feature": FEATURES,
            "importance_mean": result.importances_mean,
            "importance_std": result.importances_std,
        }
    ).sort_values(by="importance_mean", ascending=False).reset_index(drop=True)

    print(importance_df.to_string(index=False))
    return importance_df


def print_predictions(
    model: GridSearchCV,
    test_df: pd.DataFrame,
    x_test: pd.DataFrame,
    y_test: pd.Series,
    y_pred: np.ndarray,
) -> None:
    """
    Print per-sample prediction results including confidence percentage.

    Args:
        model: The fitted GridSearchCV (or pipeline) model.
        test_df: Original test DataFrame (used for the 'voice' column).
        x_test: Test feature matrix.
        y_test: Actual test labels.
        y_pred: Predicted test labels.
    """
    print_section("PER-SAMPLE PREDICTIONS")

    # Get predicted probabilities for confidence calculation.
    probabilities = model.predict_proba(x_test)
    class_labels = model.classes_ if hasattr(model, "classes_") else model.best_estimator_.classes_

    voices = test_df["voice"].reset_index(drop=True)
    y_test_reset = y_test.reset_index(drop=True)

    for idx in range(len(x_test)):
        voice_number = voices.iloc[idx]
        actual_label = y_test_reset.iloc[idx]
        predicted_label = y_pred[idx]

        # Confidence = probability assigned to the predicted class.
        predicted_class_index = list(class_labels).index(predicted_label)
        confidence = probabilities[idx][predicted_class_index] * 100

        print(
            f"Voice {voice_number} | Actual: {actual_label} | "
            f"Predicted: {predicted_label} | Confidence: {confidence:.2f}%"
        )


# --------------------------------------------------------------------------- #
# Model Persistence
# --------------------------------------------------------------------------- #

def save_model(model: GridSearchCV, model_path: str) -> None:
    """
    Save the trained model to disk using joblib, creating the
    destination folder automatically if it does not already exist.

    Args:
        model: The fitted GridSearchCV (or pipeline) model to save.
        model_path: Destination file path for the saved model.
    """
    print_section("SAVING TRAINED MODEL")

    model_dir = os.path.dirname(model_path)
    try:
        os.makedirs(model_dir, exist_ok=True)
        joblib.dump(model, model_path)
        print(f"Model successfully saved to: {model_path}")
    except Exception as exc:  # noqa: BLE001
        print(f"ERROR: Failed to save model. Reason: {exc}")
        sys.exit(1)


# --------------------------------------------------------------------------- #
# Data Preparation Helper
# --------------------------------------------------------------------------- #

def split_features_target(
    df: pd.DataFrame,
) -> Tuple[pd.DataFrame, pd.Series]:
    """
    Split a DataFrame into feature matrix (X) and target vector (y).

    Args:
        df: The source DataFrame containing FEATURES and TARGET columns.

    Returns:
        A tuple of (X, y).
    """
    x = df[FEATURES].copy()
    y = df[TARGET].copy()
    return x, y


# --------------------------------------------------------------------------- #
# Main Execution Flow
# --------------------------------------------------------------------------- #

def main() -> None:
    """
    Main execution entry point for training and evaluating the
    Nepali pronunciation evaluation SVM model.
    """
    print_section("NEPALI PRONUNCIATION EVALUATION - SVM TRAINING")

    # ----------------------------------------------------------------- #
    # 1. Ask for gender and resolve the gender-specific dataset/model
    #    paths (male and female speakers use separate datasets and
    #    separate trained models).
    # ----------------------------------------------------------------- #
    global DATASET_DIR, MODEL_PATH, TRAIN_FILES, TEST_FILE

    gender_input = input("Enter gender (Male/Female): ").strip()
    gender = normalize_gender(gender_input)

    DATASET_DIR, MODEL_PATH, TRAIN_FILES, TEST_FILE = resolve_gender_paths(gender)

    print_subsection("Selected Gender")
    print(gender.capitalize())

    # ----------------------------------------------------------------- #
    # 2. Validate that all required dataset files exist.
    # ----------------------------------------------------------------- #
    all_files = TRAIN_FILES + [TEST_FILE]
    validate_files_exist(all_files)

    # ----------------------------------------------------------------- #
    # 3. Load training and testing datasets.
    # ----------------------------------------------------------------- #
    print_subsection("Training Files")
    for path in TRAIN_FILES:
        print(f"  - {path}")

    train_df = load_and_combine_datasets(TRAIN_FILES)
    test_df = load_dataset(TEST_FILE)

    print(f"\nTraining dataset : {', '.join(TRAIN_FILES)}")
    print(f"Testing dataset  : {TEST_FILE}")
    print(f"\nNumber of training samples : {len(train_df)}")
    print(f"Number of testing samples  : {len(test_df)}")

    print_subsection("Class Distribution (Training Set)")
    print(train_df[TARGET].value_counts().to_string())

    print_subsection("Class Distribution (Testing Set)")
    print(test_df[TARGET].value_counts().to_string())

    # ----------------------------------------------------------------- #
    # 4. Prepare feature matrices and target vectors.
    # ----------------------------------------------------------------- #
    x_train, y_train = split_features_target(train_df)
    x_test, y_test = split_features_target(test_df)

    # ----------------------------------------------------------------- #
    # 5. Train the SVM model using GridSearchCV.
    # ----------------------------------------------------------------- #
    print_section("TRAINING MODEL (GridSearchCV)")
    print("Building pipeline: StandardScaler -> SVC(probability=True)")
    print(f"Parameter grid: {PARAM_GRID}")
    print(f"Cross-validation folds: {CV_FOLDS}")
    print(f"Scoring metric: {SCORING_METRIC}")
    print("Training in progress, please wait...\n")

    grid_search_model = train_model(x_train, y_train)

    print_subsection("Best Hyperparameters")
    print(grid_search_model.best_params_)

    print_subsection("Best Cross-Validation Accuracy")
    print(f"{grid_search_model.best_score_ * 100:.2f}%")

    # ----------------------------------------------------------------- #
    # 6. Evaluate the trained model on the test set.
    # ----------------------------------------------------------------- #
    y_pred = evaluate_model(grid_search_model, x_test, y_test)

    # ----------------------------------------------------------------- #
    # 7. Compute and display permutation feature importance.
    # ----------------------------------------------------------------- #
    compute_feature_importance(grid_search_model, x_test, y_test)

    # ----------------------------------------------------------------- #
    # 8. Print per-sample predictions with confidence.
    # ----------------------------------------------------------------- #
    print_predictions(grid_search_model, test_df, x_test, y_test, y_pred)

    # ----------------------------------------------------------------- #
    # 9. Save the trained model to disk.
    # ----------------------------------------------------------------- #
    save_model(grid_search_model, MODEL_PATH)

    print_section("TRAINING PIPELINE COMPLETED SUCCESSFULLY")


if __name__ == "__main__":
    try:
        main()
    except KeyboardInterrupt:
        print("\n\nProcess interrupted by user. Exiting gracefully.")
        sys.exit(1)
    except Exception as unexpected_error:  # noqa: BLE001
        print_section("UNEXPECTED ERROR")
        print(f"An unexpected error occurred: {unexpected_error}")
        sys.exit(1)