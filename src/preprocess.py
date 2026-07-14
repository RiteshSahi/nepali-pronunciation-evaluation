import librosa
import numpy as np
import noisereduce as nr

TARGET_SR = 16000


def load_audio(file_path):
    audio, sr = librosa.load(
        file_path,
        sr=TARGET_SR,
        mono=True
    )
    return audio, sr


def reduce_noise(audio, sr):
    return nr.reduce_noise(
        y=audio,
        sr=sr
    )


def remove_silence(audio, top_db=30):
    audio, _ = librosa.effects.trim(
        audio,
        top_db=top_db
    )
    return audio


def normalize_audio(audio):
    peak = np.max(np.abs(audio))

    if peak == 0:
        return audio

    return audio / peak


def pre_emphasis(audio, coefficient=0.97):
    emphasized = np.append(
        audio[0],
        audio[1:] - coefficient * audio[:-1]
    )

    return emphasized


def preprocess_audio(file_path, apply_noise_reduction=False):

    audio, sr = load_audio(file_path)

    if apply_noise_reduction:
        audio = reduce_noise(audio, sr)

    audio = remove_silence(audio)

    audio = normalize_audio(audio)

    audio = pre_emphasis(audio)

    return audio, sr


if __name__ == "__main__":

    file_path = "../dataset/user_recordings/good_voice/user_Voice71.wav"

    audio, sr = preprocess_audio(
        file_path,
        apply_noise_reduction=True
    )

    print("Preprocessing Successful")
    print(f"Sample Rate : {sr}")
    print(f"Duration    : {len(audio) / sr:.2f} seconds")
    print(f"Samples     : {len(audio)}")
    print(f"Max Value   : {np.max(audio):.4f}")
    print(f"Min Value   : {np.min(audio):.4f}")