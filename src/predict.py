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


print("=" * 60)
print("Pronunciation Prediction")
print("=" * 60)


# ======================================================
# Input Audio
# ======================================================

audio_file = os.path.expanduser(
    input("Enter user audio path (.wav): ").strip()
)


voice_id = input(
    "Enter reference voice number: "
).strip()


# ======================================================
# Reference File
# ======================================================

reference_file = os.path.join(
    REFERENCE_FOLDER,
    f"Voice{int(voice_id):02d}.wav"
)


# ======================================================
# Check Files
# ======================================================

if not os.path.exists(audio_file):
    print("\n❌ Audio file not found:")
    print(audio_file)
    exit()


if not os.path.exists(reference_file):
    print("\n❌ Reference file not found:")
    print(reference_file)
    exit()


print("\nAudio:")
print(audio_file)

print("\nReference:")
print(reference_file)


# ======================================================
# DTW Feature
# ======================================================

dtw_score = calculate_dtw_distance(
    reference_file,
    audio_file
)


# ======================================================
# Duration Feature
# ======================================================

audio, sr = preprocess_audio(
    audio_file
)

duration = len(audio) / sr


# ======================================================
# ZCR Feature
# ======================================================

zcr = librosa.feature.zero_crossing_rate(
    audio
).mean()


# ======================================================
# ASR Features
# ======================================================

# Temporary values
# Replace with wav2vec2 ASR WER/CER later

wer = 0.0
cer = 0.0


# ======================================================
# Create Feature Vector
# ======================================================

features = pd.DataFrame(
    [[
        dtw_score,
        duration,
        zcr,
        wer,
        cer
    ]],
    columns=[
        "dtw",
        "duration",
        "zcr",
        "wer",
        "cer"
    ]
)


print("\n" + "=" * 60)
print("Input Features")
print("=" * 60)

print(features)


# ======================================================
# Prediction
# ======================================================

prediction = model.predict(
    features
)


probability = model.predict_proba(
    features
)


confidence = max(probability[0]) * 100


# ======================================================
# Output
# ======================================================

print("\n" + "=" * 60)
print("Result")
print("=" * 60)

print(
    "Pronunciation:",
    prediction[0]
)

print(
    f"Confidence: {confidence:.2f}%"
)

print("=" * 60)