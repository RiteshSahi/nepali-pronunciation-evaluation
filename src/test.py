from dtw_distance import calculate_dtw_distance
import os

# ----------------------------------
# Change only this
# ----------------------------------
MODE = "bad"      # "good" or "bad"
# ----------------------------------

reference_folder = "../dataset/app_reference/"

voice_ids = [
    71,
    76,
    80,
    83,
    97,
    102,
    109,
    113,
    117,
    121,
    123,
    137,
    141,
    149,
    151,
    167
]

if MODE == "good":
    user_folder = "../dataset/user_recordings/good_voice/"
    prefix = "user_Voice"
else:
    user_folder = "../dataset/user_recordings/bad_voice/"
    prefix = "user_badVoice"

print("=" * 50)
print(f"Pronunciation Evaluation using DTW ({MODE.upper()})")
print("=" * 50)

for vid in voice_ids:

    reference_file = os.path.join(
        reference_folder,
        f"Voice{vid}.wav"
    )

    user_file = os.path.join(
        user_folder,
        f"{prefix}{vid}.wav"
    )

    if not os.path.exists(user_file):
        print(f"Voice{vid:03d} -> Recording not found")
        continue

    raw_distance, normalized_distance = calculate_dtw_distance(
        reference_file,
        user_file
    )

    print(
        f"Voice{vid:03d} -> DTW Distance: {normalized_distance:.4f}"
    )

print("=" * 50)
print("Evaluation Completed")
print("=" * 50)