import os
import joblib
import pandas as pd
from dtw_distance import calculate_dtw_distance
from asr import transcribe_audio
from text_compare import compare_text
from feature_extraction import get_duration


# ======================================================
# Paths
# ======================================================

MODEL_PATH = "../models/svm_pronunciation_model.pkl"

REFERENCE_FOLDER = "../dataset/app_reference/"

SENTENCE_FILE = os.path.join(
    REFERENCE_FOLDER,
    "sentences.csv"
)


# ======================================================
# Load Model
# ======================================================

model = joblib.load(MODEL_PATH)


# ======================================================
# Prediction Function
# ======================================================

def predict(audio_file, voice_id):

    # Accept both "64" and "Voice64"
    voice_id = str(voice_id).replace("Voice", "")

    # --------------------------------------------------
    # Reference Audio
    # --------------------------------------------------

    reference_audio = os.path.join(
    "../dataset",
    gender,
    "app_reference",
    f"Voice{voice_id}.wav"
)

    if not os.path.exists(audio_file):
        raise FileNotFoundError(audio_file)

    if not os.path.exists(reference_audio):
        raise FileNotFoundError(reference_audio)

    # --------------------------------------------------
    # DTW
    # --------------------------------------------------

    dtw_score = calculate_dtw_distance(
        reference_audio,
        audio_file
    )

    # --------------------------------------------------
    # Duration Difference
    # --------------------------------------------------

    reference_duration = get_duration(
        reference_audio
    )

    user_duration = get_duration(
        audio_file
    )

    duration_diff = abs(
        reference_duration -
        user_duration
    )

    # --------------------------------------------------
    # ASR
    # --------------------------------------------------

    recognized_text = transcribe_audio(
        audio_file
    )

    # --------------------------------------------------
    # Reference Sentence
    # --------------------------------------------------

    sentence_df = pd.read_csv(
        SENTENCE_FILE
    )

    row = sentence_df[
        sentence_df["audio_id"] == f"Voice{voice_id}"
    ]

    if row.empty:
        raise ValueError(
            f"Reference sentence not found for Voice{voice_id}"
        )

    reference_text = row.iloc[0]["sentence"]

    # --------------------------------------------------
    # WER / CER
    # --------------------------------------------------

    wer_score, cer_score = compare_text(
        reference_text,
        recognized_text
    )

    # --------------------------------------------------
    # Debug
    # --------------------------------------------------

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

    # --------------------------------------------------
    # Feature Vector
    # --------------------------------------------------

    features = pd.DataFrame(
        [[
            dtw_score,
            duration_diff,
            wer_score,
            cer_score
        ]],
        columns=[
            "dtw",
            "duration",
            "wer",
            "cer"
        ]
    )

    # --------------------------------------------------
    # Prediction
    # --------------------------------------------------

    # Safety rule: if the recognized text is essentially wrong
    # (empty transcription, or very high word/character error rate),
    # reject immediately as "Bad" without trusting the SVM.
    # The SVM was trained mostly on samples with WER/CER inside a
    # "normal" range, so it can behave unpredictably on extreme,
    # out-of-range inputs like silence or a completely different
    # sentence being spoken. This rule acts as a safety net that
    # catches those cases before they ever reach the model.

    if recognized_text.strip() == "" or wer_score > 0.8 or cer_score > 0.5:

        prediction = "Bad"
        confidence = 100.0
        method = "rule-based (utterance mismatch)"

    else:

        prediction = model.predict(
            features
        )[0]

        probability = model.predict_proba(
            features
        )[0]

        confidence = probability.max() * 100
        method = "SVM"

    print(f"Method     : {method}")

    # --------------------------------------------------
    # Return
    # --------------------------------------------------

    return (
        prediction,
        confidence,
        dtw_score,
        duration_diff,
        wer_score,
        cer_score
    )


# ======================================================
# Terminal Test
# ======================================================

if __name__ == "__main__":

    print("=" * 60)
    print("Pronunciation Prediction")
    print("=" * 60)

    audio_file = os.path.expanduser(
        input("Enter user audio path (.wav): ").strip()
    )

    voice_id = input(
        "Enter reference voice id: "
    ).strip()

    prediction, confidence, dtw, duration, wer, cer = predict(
        audio_file,
        voice_id
    )

    print("\n" + "=" * 60)
    print("INPUT FEATURES")
    print("=" * 60)

    print(f"DTW       : {dtw:.4f}")
    print(f"Duration  : {duration:.4f}")
    print(f"WER       : {wer:.4f}")
    print(f"CER       : {cer:.4f}")

    print("\n" + "=" * 60)
    print("RESULT")
    print("=" * 60)

    print(f"Prediction : {prediction}")
    print(f"Confidence : {confidence:.2f}%")