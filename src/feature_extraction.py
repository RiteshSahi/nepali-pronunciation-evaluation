import os
import numpy as np
import librosa

from preprocess import preprocess_audio

N_MFCC = 13


# =====================================================
# Extract MFCC Features
# =====================================================

def extract_mfcc(audio, sample_rate, n_mfcc=N_MFCC):
    """
    Extract 39-dimensional MFCC features:
    13 MFCC + 13 Delta + 13 Delta-Delta
    """

    mfcc = librosa.feature.mfcc(
        y=audio,
        sr=sample_rate,
        n_mfcc=n_mfcc
    )

    delta = librosa.feature.delta(
        mfcc
    )

    delta2 = librosa.feature.delta(
        mfcc,
        order=2
    )

    features = np.vstack([
        mfcc,
        delta,
        delta2
    ])

    return features


# =====================================================
# CMVN Normalization
# =====================================================

def apply_cmvn(features):
    """
    Cepstral Mean and Variance Normalization
    """

    mean = np.mean(
        features,
        axis=1,
        keepdims=True
    )

    std = np.std(
        features,
        axis=1,
        keepdims=True
    )

    std[std == 0] = 1

    return (features - mean) / std


# =====================================================
# Complete Feature Extraction Pipeline
# =====================================================

def extract_features(file_path):
    """
    Complete feature extraction pipeline.

    Steps:
    1. Preprocess audio
    2. Extract MFCC
    3. Extract Delta and Delta-Delta
    4. Apply CMVN
    """

    audio, sr = preprocess_audio(
        file_path
    )

    features = extract_mfcc(
        audio,
        sr
    )

    features = apply_cmvn(
        features
    )

    return features


# =====================================================
# Get Audio Duration
# =====================================================

def get_duration(file_path):

    audio, sr = preprocess_audio(
        file_path
    )

    duration = len(audio) / sr

    return duration


# =====================================================
# Zero Crossing Rate
# =====================================================

def get_zcr(audio_path):

    y, sr = librosa.load(
        audio_path,
        sr=16000
    )

    zcr = librosa.feature.zero_crossing_rate(
        y
    )[0]

    return np.mean(zcr)


# =====================================================
# Test
# =====================================================

if __name__ == "__main__":

    file_path = input(
        "Enter audio path: "
    ).strip()


    if not os.path.exists(file_path):

        print("Audio file not found.")
        exit()


    features = extract_features(
        file_path
    )


    print("\n" + "=" * 50)
    print("Feature Extraction")
    print("=" * 50)

    print(f"File           : {file_path}")
    print(f"Feature Shape  : {features.shape}")
    print(f"Feature Dim    : {features.shape[0]}")
    print(f"Frames         : {features.shape[1]}")
    print(f"Mean           : {np.mean(features):.4f}")
    print(f"Std            : {np.std(features):.4f}")

    print("=" * 50)