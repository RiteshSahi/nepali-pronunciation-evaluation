import os
import librosa
import librosa.display
import numpy as np
import soundfile as sf
import matplotlib.pyplot as plt

TARGET_SR = 16000


# =====================================================
# Load Audio
# =====================================================

def load_audio(file_path):
    """
    Load audio as mono and resample to 16 kHz.
    """
    audio, sr = librosa.load(
        file_path,
        sr=TARGET_SR,
        mono=True
    )

    return audio, sr


# =====================================================
# Remove Silence
# =====================================================

def remove_silence(audio, top_db=30):
    """
    Remove leading and trailing silence.
    """
    audio, _ = librosa.effects.trim(
        audio,
        top_db=top_db
    )

    return audio


# =====================================================
# Normalize Audio
# =====================================================

def normalize_audio(audio):
    """
    Normalize waveform to [-1, 1].
    """
    peak = np.max(np.abs(audio))

    if peak == 0:
        return audio

    return audio / peak


# =====================================================
# Preprocess Pipeline
# =====================================================

def preprocess_audio(file_path, save_output=False):
    """
    Complete preprocessing pipeline.

    Steps:
    1. Load audio
    2. Remove silence
    3. Normalize
    """

    audio, sr = load_audio(file_path)

    audio = remove_silence(audio)

    audio = normalize_audio(audio)

    if save_output:

        os.makedirs(
            "../temp",
            exist_ok=True
        )

        sf.write(
            "../temp/preprocessed.wav",
            audio,
            sr
        )

    return audio, sr


# =====================================================
# Test
# =====================================================

if __name__ == "__main__":

    file_path = input(
        "Enter audio file path: "
    ).strip()

    if not os.path.exists(file_path):
        print("Audio file not found.")
        exit()

    # Original audio
    original_audio, _ = load_audio(file_path)

    # Preprocessed audio
    audio, sr = preprocess_audio(
        file_path,
        save_output=True
    )

    print("\n" + "=" * 60)
    print("Preprocessing Successful")
    print("=" * 60)
    print(f"File        : {file_path}")
    print(f"Sample Rate : {sr}")
    print(f"Duration    : {len(audio)/sr:.2f} sec")
    print(f"Samples     : {len(audio)}")
    print(f"Max Value   : {np.max(audio):.4f}")
    print(f"Min Value   : {np.min(audio):.4f}")
    print("=" * 60)

    print("\nSaved:")
    print("../temp/preprocessed.wav")

    # =====================================================
    # Plot Waveforms
    # =====================================================

    plt.figure(figsize=(12, 6))

    # Original
    plt.subplot(2, 1, 1)
    librosa.display.waveshow(original_audio, sr=sr)
    plt.title("Original Audio")

    # Preprocessed
    plt.subplot(2, 1, 2)
    librosa.display.waveshow(audio, sr=sr)
    plt.title("Preprocessed Audio")

    plt.tight_layout()
    plt.show()