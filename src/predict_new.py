#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
predict.py
==========

Inference script for the AI-based Nepali Pronunciation Evaluation project.

Given a user's recorded audio file and a reference voice id, this script:
    1. Locates the matching reference audio and reference sentence.
    2. Computes DTW distance, duration difference, WER, and CER.
    3. Applies a rule-based safety net for clearly failed utterances.
    4. Otherwise feeds the feature vector into the trained SVM pipeline
       (loaded from ../models/svm_pronunciation_model.pkl) to predict
       "Good" or "Bad" pronunciation, with a confidence percentage.

Python Version: 3.9+
"""

import os
import sys
from typing import Tuple

import joblib
import pandas as pd

from asr import transcribe_audio
from dtw_distance import calculate_dtw_distance
from feature_extraction import get_duration, get_zcr
from text_compare import compare_text

# --------------------------------------------------------------------------- #
# Paths / Configuration
# --------------------------------------------------------------------------- #

# NOTE: These are now resolved per-gender via `get_paths_for_gender()`
# below, since male and female speakers use different reference datasets
# and different trained models. The old fixed constants are kept here
# only as a comment for reference:
#   MODEL_PATH = "../models/svm_pronunciation_model.pkl"
#   REFERENCE_FOLDER = "../dataset/app_reference"
#   SENTENCE_FILE = os.path.join(REFERENCE_FOLDER, "sentences.csv")

GENDER_PATHS = {
    "male": {
        "model_path": "../models/male_model.pkl",
        "reference_folder": "../dataset/male/app_reference",
    },
    "female": {
        "model_path": "../models/female_model.pkl",
        "reference_folder": "../dataset/female/app_reference",
    },
}

# Full feature set this script is capable of producing, in a fixed order.
# The actual columns sent to the model are trimmed/reordered to match
# whatever the loaded model expects (see `resolve_feature_columns`).
ALL_POSSIBLE_FEATURES = ["dtw", "duration", "wer", "cer", "zcr"]

# Rule-based safety-net thresholds. If the recognized speech is empty or
# wildly off from the reference sentence, we skip the SVM entirely and
# label the attempt "Bad" outright, since the model was trained mostly on
# samples within a "normal" WER/CER range and can behave unpredictably on
# extreme, out-of-distribution inputs (e.g. silence, wrong sentence).
WER_REJECT_THRESHOLD = 0.8
CER_REJECT_THRESHOLD = 0.5


# --------------------------------------------------------------------------- #
# Model Loading
# --------------------------------------------------------------------------- #

def load_model(model_path: str):
    """
    Load the trained SVM pipeline (or GridSearchCV object) from disk.

    Args:
        model_path: Path to the joblib-serialized model file.

    Returns:
        The loaded model object.
    """
    if not os.path.exists(model_path):
        print("=" * 60)
        print("ERROR: MODEL FILE NOT FOUND")
        print("=" * 60)
        print(f"Expected model at: {model_path}")
        print(
            "Please run the training script (train_svm.py) first to "
            "generate this file."
        )
        sys.exit(1)

    try:
        loaded_model = joblib.load(model_path)
    except Exception as exc:  # noqa: BLE001
        print("=" * 60)
        print("ERROR: FAILED TO LOAD MODEL")
        print("=" * 60)
        print(f"File: {model_path}")
        print(f"Reason: {exc}")
        sys.exit(1)

    return loaded_model


def resolve_feature_columns(model) -> list:
    """
    Determine which feature columns the loaded model actually expects,
    in the correct order, so the feature vector built at inference time
    always matches what the model was trained on.

    Tries, in order:
        1. `feature_names_in_` on the best estimator / pipeline (set by
           scikit-learn automatically when the model was trained on a
           pandas DataFrame with named columns).
        2. Falling back to `ALL_POSSIBLE_FEATURES` trimmed to the
           model's expected input size (`n_features_in_`), if available.

    Args:
        model: The loaded model (Pipeline or GridSearchCV) object.

    Returns:
        Ordered list of feature column names to use for prediction.

    Raises:
        SystemExit: If the expected feature set/order cannot be
            determined reliably.
    """
    # GridSearchCV wraps the actual pipeline in `best_estimator_`.
    estimator = getattr(model, "best_estimator_", model)

    # Preferred path: scikit-learn stores the exact fitted column names.
    feature_names = getattr(estimator, "feature_names_in_", None)
    if feature_names is not None:
        return list(feature_names)

    # Fallback: infer from expected input width only. This is less safe
    # because it assumes column *order* matches ALL_POSSIBLE_FEATURES,
    # so we warn the user loudly.
    n_features = getattr(estimator, "n_features_in_", None)
    if n_features is not None and n_features <= len(ALL_POSSIBLE_FEATURES):
        print(
            "WARNING: Could not read exact feature names from the model. "
            f"Falling back to the first {n_features} of "
            f"{ALL_POSSIBLE_FEATURES}. Verify this matches your training "
            "script's FEATURES list."
        )
        return ALL_POSSIBLE_FEATURES[:n_features]

    print("=" * 60)
    print("ERROR: UNABLE TO DETERMINE MODEL FEATURE SET")
    print("=" * 60)
    print(
        "The loaded model does not expose 'feature_names_in_' or "
        "'n_features_in_'. Cannot safely build a matching feature vector."
    )
    sys.exit(1)


# --------------------------------------------------------------------------- #
# Helpers
# --------------------------------------------------------------------------- #

def normalize_gender(raw_gender: str) -> str:
    """
    Normalize a user/UI-provided gender value (e.g. Streamlit's
    st.radio(["Male", "Female"]) selection) into one of the internal
    keys used by GENDER_PATHS: "male" or "female".

    Args:
        raw_gender: Raw gender string, any case (e.g. "Male", "female").

    Returns:
        Either "male" or "female".

    Raises:
        ValueError: If the value isn't recognized.
    """
    cleaned = raw_gender.strip().lower()
    if cleaned not in GENDER_PATHS:
        raise ValueError(
            f"Invalid gender '{raw_gender}'. Expected one of: "
            f"{', '.join(sorted(GENDER_PATHS))}."
        )
    return cleaned


def get_paths_for_gender(gender: str) -> Tuple[str, str, str]:
    """
    Resolve the model path, reference folder, and sentence CSV path for
    the given gender.

    Args:
        gender: Raw gender string (e.g. "Male", "female", "MALE").

    Returns:
        A tuple of (model_path, reference_folder, sentence_file).

    Raises:
        ValueError: If the gender is not recognized.
    """
    normalized = normalize_gender(gender)
    paths = GENDER_PATHS[normalized]
    reference_folder = paths["reference_folder"]
    sentence_file = os.path.join(reference_folder, "sentences.csv")
    return paths["model_path"], reference_folder, sentence_file


def normalize_voice_id(raw_voice_id: str) -> str:
    """
    Normalize a user-provided voice id into a bare numeric/string id,
    stripping an optional "voice" prefix in any letter case and
    trimming whitespace.

    Examples:
        "Voice64"  -> "64"
        "voice64"  -> "64"
        "VOICE 64" -> "64"
        "64"       -> "64"

    Args:
        raw_voice_id: The raw user input for the voice id.

    Returns:
        The normalized voice id string.

    Raises:
        ValueError: If the result is empty after normalization.
    """
    cleaned = raw_voice_id.strip()

    lowered = cleaned.lower()
    if lowered.startswith("voice"):
        cleaned = cleaned[len("voice"):]

    cleaned = cleaned.strip()

    if cleaned == "":
        raise ValueError(
            f"Invalid voice id '{raw_voice_id}'. Expected a numeric id, "
            "optionally prefixed with 'Voice' (e.g. 'Voice64' or '64')."
        )

    return cleaned


def load_reference_sentence(voice_id: str, sentence_file: str) -> str:
    """
    Look up the reference sentence text for a given voice id.

    Args:
        voice_id: The normalized voice id (e.g. "64").
        sentence_file: Path to the gender-specific sentences.csv, as
            resolved by `get_paths_for_gender()`.

    Returns:
        The reference sentence text.

    Raises:
        FileNotFoundError: If the sentence CSV file does not exist.
        ValueError: If no matching row is found for the voice id.
    """
    if not os.path.exists(sentence_file):
        raise FileNotFoundError(
            f"Reference sentence file not found: {sentence_file}"
        )

    try:
        sentence_df = pd.read_csv(sentence_file)
    except Exception as exc:  # noqa: BLE001
        raise RuntimeError(
            f"Failed to read reference sentence file '{sentence_file}': {exc}"
        ) from exc

    if "audio_id" not in sentence_df.columns or "sentence" not in sentence_df.columns:
        raise ValueError(
            f"'{sentence_file}' must contain 'audio_id' and 'sentence' "
            "columns."
        )

    row = sentence_df[sentence_df["audio_id"] == f"Voice{voice_id}"]

    if row.empty:
        raise ValueError(f"Reference sentence not found for Voice{voice_id}")

    return row.iloc[0]["sentence"]


# --------------------------------------------------------------------------- #
# Prediction
# --------------------------------------------------------------------------- #

def predict(
    audio_file: str, voice_id: str, gender: str
) -> Tuple[str, float, dict]:
    """
    Run the full prediction pipeline for a single user recording.

    Args:
        audio_file: Path to the user's recorded .wav file.
        voice_id: Reference voice id (raw user input, will be normalized).
        gender: Raw gender string from the UI or CLI (e.g. "Male"/
            "Female"). Used to resolve the gender-appropriate model,
            reference folder, and sentences.csv for this speaker. The
            model is loaded internally so both the CLI (`main()`) and
            the Streamlit app can call this function the exact same
            way, without either caller having to pre-load a model.

    Returns:
        A tuple of (prediction, confidence_percent, feature_values) where
        feature_values is a dict mapping feature name -> computed value.

    Raises:
        FileNotFoundError: If the user audio or reference audio is missing.
        ValueError: If the reference sentence cannot be found, or the
            gender value is not recognized.
        RuntimeError: If any feature-extraction step fails.
    """
    voice_id = normalize_voice_id(voice_id)

    # ----------------------------------------------------------------- #
    # Resolve gender-specific model path / reference folder / sentence
    # file, then load the model.
    # ----------------------------------------------------------------- #
    model_path, reference_folder, sentence_file = get_paths_for_gender(gender)
    model = load_model(model_path)

    # ----------------------------------------------------------------- #
    # Resolve audio file paths.
    # ----------------------------------------------------------------- #
    if not os.path.exists(audio_file):
        raise FileNotFoundError(f"User audio file not found: {audio_file}")

    reference_audio = os.path.join(reference_folder, f"Voice{voice_id}.wav")
    if not os.path.exists(reference_audio):
        raise FileNotFoundError(f"Reference audio file not found: {reference_audio}")

    # ----------------------------------------------------------------- #
    # DTW distance.
    # ----------------------------------------------------------------- #
    try:
        dtw_score = calculate_dtw_distance(reference_audio, audio_file)
    except Exception as exc:  # noqa: BLE001
        raise RuntimeError(f"DTW calculation failed: {exc}") from exc

    # ----------------------------------------------------------------- #
    # Duration difference.
    # ----------------------------------------------------------------- #
    try:
        reference_duration = get_duration(reference_audio)
        user_duration = get_duration(audio_file)
        duration_diff = abs(reference_duration - user_duration)
    except Exception as exc:  # noqa: BLE001
        raise RuntimeError(f"Duration extraction failed: {exc}") from exc


    # ----------------------------------------------------------------- #
    # ZCR difference.
    # ----------------------------------------------------------------- #
    try:
        reference_zcr = get_zcr(reference_audio)
        user_zcr = get_zcr(audio_file)
        zcr_diff = abs(reference_zcr - user_zcr)
    except Exception as exc:  # noqa: BLE001
        raise RuntimeError(f"ZCR extraction failed: {exc}") from exc

    # ----------------------------------------------------------------- #
    # ASR transcription.
    # ----------------------------------------------------------------- #
    try:
        recognized_text = transcribe_audio(audio_file)
    except Exception as exc:  # noqa: BLE001
        raise RuntimeError(f"Speech recognition (ASR) failed: {exc}") from exc

    # ----------------------------------------------------------------- #
    # Reference sentence lookup.
    # ----------------------------------------------------------------- #
    reference_text = load_reference_sentence(voice_id, sentence_file)

    # ----------------------------------------------------------------- #
    # WER / CER.
    # ----------------------------------------------------------------- #
    try:
        wer_score, cer_score = compare_text(reference_text, recognized_text)
    except Exception as exc:  # noqa: BLE001
        raise RuntimeError(f"WER/CER computation failed: {exc}") from exc

    # ----------------------------------------------------------------- #
    # Debug output.
    # ----------------------------------------------------------------- #
    print("\n" + "=" * 60)
    print("ASR RESULT")
    print("=" * 60)
    print("Reference:")
    print(reference_text)
    print()
    print("Recognized:")
    print(recognized_text)
    print()
    print(f"WER : {wer_score:.4f}")
    print(f"CER : {cer_score:.4f}")
    print(f"ZCR : {zcr_diff:.4f}")

    # ----------------------------------------------------------------- #
    # Assemble all computed feature values into a single lookup dict.
    # ----------------------------------------------------------------- #
    computed_values = {
        "dtw": dtw_score,
        "duration": duration_diff,
        "wer": wer_score,
        "cer": cer_score,
        "zcr": zcr_diff,
    }
    # ----------------------------------------------------------------- #
    # Rule-based safety net for clearly failed utterances.
    # ----------------------------------------------------------------- #
    if (
        recognized_text.strip() == ""
        or wer_score > WER_REJECT_THRESHOLD
        or cer_score > CER_REJECT_THRESHOLD
    ):
        prediction = "Bad"
        confidence = 100.0
        method = "rule-based (utterance mismatch)"

    else:
        # ------------------------------------------------------------- #
        # Build the feature vector in the exact order the model expects.
        # ------------------------------------------------------------- #
        feature_columns = resolve_feature_columns(model)

        missing = [col for col in feature_columns if col not in computed_values]
        if missing:
            raise RuntimeError(
                f"Cannot build feature vector: missing computed value(s) "
                f"for {missing}. The loaded model expects features "
                f"{feature_columns}, but this script could not compute "
            )

        features_df = pd.DataFrame(
            [[computed_values[col] for col in feature_columns]],
            columns=feature_columns,
        )

        try:
            prediction = model.predict(features_df)[0]
            probability = model.predict_proba(features_df)[0]
            confidence = float(probability.max()) * 100.0
        except Exception as exc:  # noqa: BLE001
            raise RuntimeError(f"Model prediction failed: {exc}") from exc

        method = "SVM"

    print(f"Method     : {method}")

    return prediction, confidence, computed_values


# --------------------------------------------------------------------------- #
# Terminal Entry Point
# --------------------------------------------------------------------------- #

def main() -> None:
    """
    Command-line entry point: prompts the user for an audio file and
    reference voice id, runs the prediction pipeline, and prints results.
    """
    print("=" * 60)
    print("Pronunciation Prediction")
    print("=" * 60)

    gender_input = input("Enter gender (Male/Female): ").strip()

    audio_file = os.path.expanduser(
        input("Enter user audio path (.wav): ").strip()
    )
    voice_id_input = input("Enter reference voice id: ").strip()

    try:
        prediction, confidence, feature_values = predict(
            audio_file, voice_id_input, gender_input
        )
    except (FileNotFoundError, ValueError, RuntimeError) as exc:
        print("\n" + "=" * 60)
        print("ERROR")
        print("=" * 60)
        print(str(exc))
        sys.exit(1)
    except KeyboardInterrupt:
        print("\n\nProcess interrupted by user. Exiting gracefully.")
        sys.exit(1)

    print("\n" + "=" * 60)
    print("INPUT FEATURES")
    print("=" * 60)
    for name, value in feature_values.items():
        print(f"{name.upper():<10}: {value:.4f}")

    print("\n" + "=" * 60)
    print("RESULT")
    print("=" * 60)
    print(f"Prediction : {prediction}")
    print(f"Confidence : {confidence:.2f}%")


if __name__ == "__main__":
    try:
        main()
    except KeyboardInterrupt:
        print("\n\nProcess interrupted by user. Exiting gracefully.")
        sys.exit(1)
    except Exception as unexpected_error:  # noqa: BLE001
        print("=" * 60)
        print("UNEXPECTED ERROR")
        print("=" * 60)
        print(f"An unexpected error occurred: {unexpected_error}")
        sys.exit(1)