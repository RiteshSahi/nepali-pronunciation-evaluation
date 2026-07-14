import librosa
from feature_extraction import extract_features

def calculate_dtw_distance(file1, file2):

    mfcc1 = extract_features(file1)
    mfcc2 = extract_features(file2)

    D, wp = librosa.sequence.dtw(
        X=mfcc1,
        Y=mfcc2,
        metric="euclidean",
        global_constraints=True,
        band_rad=0.25
    )

    raw_distance = D[-1, -1]

    path_length = max(len(wp), 1)

    normalized_distance = raw_distance / path_length

    return raw_distance, normalized_distance


if __name__ == "__main__":
    base = "../dataset/app_reference/"

    test_pairs = [
        ("Voice71.wav", "Voice71.wav"),    # identical -> baseline, expect 0
        ("Voice83.wav", "Voice141.wav"),   # short vs short
        ("Voice102.wav", "Voice167.wav"),  # short vs short
        ("Voice97.wav", "Voice149.wav"),   # long vs long
        ("Voice121.wav", "Voice137.wav"),  # long vs long
        ("Voice83.wav", "Voice97.wav"),    # short vs long
        ("Voice141.wav", "Voice149.wav"),  # short vs long
    ]

    for f1, f2 in test_pairs:
        raw, norm = calculate_dtw_distance(base + f1, base + f2)
        print(f"{f1} vs {f2} -> Raw: {raw:.4f} | Normalized: {norm:.4f}")