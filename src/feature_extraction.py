import numpy as np
import librosa

from preprocess import preprocess_audio


N_MFCC = 13


def extract_mfcc(audio, sample_rate, n_mfcc=N_MFCC):

    # MFCC features
    mfcc = librosa.feature.mfcc(
        y=audio,
        sr=sample_rate,
        n_mfcc=n_mfcc
    )

    # First order derivative
    delta = librosa.feature.delta(
        mfcc
    )

    # Second order derivative
    delta2 = librosa.feature.delta(
        mfcc,
        order=2
    )

    # Combine MFCC + Delta + Delta-Delta
    combined = np.vstack([
        mfcc,
        delta,
        delta2
    ])

    return combined



def apply_cmvn(features):

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

    # Avoid division by zero
    std[std == 0] = 1

    normalized = (
        features - mean
    ) / std

    return normalized



def extract_features(file_path):

    # Apply noise reduction only for user recordings
    use_noise_reduction = "user_recordings" in file_path

    audio, sr = preprocess_audio(
        file_path,
        apply_noise_reduction=use_noise_reduction
    )

    # Extract 39-dimensional MFCC features
    features = extract_mfcc(
        audio,
        sr
    )

    # Apply CMVN
    features = apply_cmvn(
        features
    )

    return features


if __name__ == "__main__":

    file_path = "../dataset/app_reference/Voice71.wav"

    features = extract_features(
        file_path
    )

    print("Feature Extraction Successful")
    print(f"Feature Shape : {features.shape}")
    print(f"Mean          : {np.mean(features):.4f}")
    print(f"Std           : {np.std(features):.4f}")