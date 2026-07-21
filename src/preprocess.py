import os
import librosa
import numpy as np
import noisereduce as nr

TARGET_SR = 16000


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


def reduce_noise(audio, sr):
    """
    Reduce background noise.
    """
    return nr.reduce_noise(
        y=audio,
        sr=sr
    )


def remove_silence(audio, top_db=30):
    """
    Remove leading and trailing silence.
    """
    audio, _ = librosa.effects.trim(
        audio,
        top_db=top_db
    )
    return audio


def normalize_audio(audio):
    """
    Normalize audio amplitude.
    """
    peak = np.max(np.abs(audio))

    if peak == 0:
        return audio

    return audio / peak


def pre_emphasis(audio, coefficient=0.97):
    """
    Apply pre-emphasis filter.
    """
    emphasized = np.append(
        audio[0],
        audio[1:] - coefficient * audio[:-1]
    )

    return emphasized


def preprocess_audio(file_path, apply_noise_reduction=True):
    """
    Complete preprocessing pipeline.
    """

    audio, sr = load_audio(file_path)

    if apply_noise_reduction:
        audio = reduce_noise(audio, sr)

    audio = remove_silence(audio)

    audio = normalize_audio(audio)

    audio = pre_emphasis(audio)

    return audio, sr


if __name__ == "__main__":

    file_path = input(
        "Enter audio file path: "
    ).strip()

    if not os.path.exists(file_path):
        print("Audio file not found.")
        exit()

    audio, sr = preprocess_audio(
        file_path,
        apply_noise_reduction=True
    )

    print("\n" + "=" * 50)
    print("Preprocessing Successful")
    print("=" * 50)
    print(f"File         : {file_path}")
    print(f"Sample Rate  : {sr}")
    print(f"Duration     : {len(audio) / sr:.2f} seconds")
    print(f"Samples      : {len(audio)}")
    print(f"Max Value    : {np.max(audio):.4f}")
    print(f"Min Value    : {np.min(audio):.4f}")
    print("=" * 50)