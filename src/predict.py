import os
import joblib
import pandas as pd
import librosa

from preprocess import preprocess_audio
from dtw_distance import calculate_dtw_distance

# ======================================================
# Paths
# ======================================================

MODEL_PATH = "../models/svm_pronunciation_model.pkl"
REFERENCE_FOLDER = "../dataset/app_reference"

# ======================================================
# Load Model
# ======================================================

model = joblib.load(MODEL_PATH)


# ======================================================
# Prediction Function
# ======================================================

def predict(audio_file, voice_id):

    # --------------------------------------------------
    # Reference File
    # --------------------------------------------------

    reference_file = os.path.join(
        REFERENCE_FOLDER,
        f"{voice_id}.wav"
    )

    # --------------------------------------------------
    # Check Files
    # --------------------------------------------------

    if not os.path.exists(audio_file):
        raise FileNotFoundError(audio_file)

    if not os.path.exists(reference_file):
        raise FileNotFoundError(reference_file)

    # --------------------------------------------------
    # DTW
    # --------------------------------------------------

    dtw_score = calculate_dtw_distance(
        reference_file,
        audio_file
    )

    # --------------------------------------------------
    # Duration
    # --------------------------------------------------

    audio, sr = preprocess_audio(
        audio_file
    )

    duration = len(audio) / sr

  
    # --------------------------------------------------
    # ASR Features
    # --------------------------------------------------

    # Replace these later with actual WER/CER
    wer = 0.0
    cer = 0.0

    # --------------------------------------------------
    # Feature Vector
    # --------------------------------------------------

    features = pd.DataFrame(
        [[
            dtw_score,
            duration,
            wer,
            cer
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

    prediction = model.predict(
        features
    )[0]

    probability = model.predict_proba(
        features
    )[0]

    confidence = probability.max() * 100

    # --------------------------------------------------
    # Return
    # --------------------------------------------------

    return (
        prediction,
        confidence,
        dtw_score,
        duration,
        wer,
        cer
    )


# ======================================================
# Terminal Testing
# ======================================================

if __name__ == "__main__":

    print("=" * 60)
    print("Pronunciation Prediction")
    print("=" * 60)

    audio_file = os.path.expanduser(
        input("Enter user audio path (.wav): ").strip()
    )

    voice_id = input(
        "Enter reference voice number: "
    ).strip()

    prediction, confidence, dtw, duration, wer, cer = predict(
        audio_file,
        voice_id
    )

    print("\n" + "=" * 60)
    print("Input Features")
    print("=" * 60)

    print(f"DTW      : {dtw:.4f}")
    print(f"Duration : {duration:.4f}")
    print(f"WER      : {wer:.4f}")
    print(f"CER      : {cer:.4f}")

    print("\n" + "=" * 60)
    print("Prediction")
    print("=" * 60)

    print(f"Pronunciation : {prediction}")
    print(f"Confidence    : {confidence:.2f}%")