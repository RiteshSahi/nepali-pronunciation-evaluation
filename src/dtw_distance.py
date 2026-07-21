import os
import librosa

from feature_extraction import extract_features


def calculate_dtw_distance(reference_file, user_file):
    """
    Calculate normalized DTW distance between
    reference audio and user audio.
    """

    reference_features = extract_features(reference_file)
    user_features = extract_features(user_file)

    distance_matrix, warping_path = librosa.sequence.dtw(
        X=reference_features,
        Y=user_features,
        metric="euclidean",
        global_constraints=True,
        band_rad=0.25
    )

    raw_distance = distance_matrix[-1, -1]

    normalized_distance = raw_distance / max(len(warping_path), 1)

    return normalized_distance


if __name__ == "__main__":

    reference = input("Reference audio path: ").strip()
    user = input("User audio path      : ").strip()

    if not os.path.exists(reference):
        print("Reference audio not found.")
        exit()

    if not os.path.exists(user):
        print("User audio not found.")
        exit()

    distance = calculate_dtw_distance(
        reference,
        user
    )

    print("\n" + "=" * 50)
    print("DTW Evaluation")
    print("=" * 50)
    print(f"Normalized DTW Distance : {distance:.4f}")
    print("=" * 50)