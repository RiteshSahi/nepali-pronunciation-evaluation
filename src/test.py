import os

from dtw_distance import calculate_dtw_distance

# =====================================================
# Select User
# =====================================================

USER = input(
    "Enter user folder (user1/user2/user3): "
).strip()

MODE = input(
    "Enter mode (good/bad): "
).strip().lower()

# =====================================================
# Paths
# =====================================================

REFERENCE_FOLDER = "../dataset/app_reference/"

if MODE == "good":

    USER_FOLDER = (
        f"../dataset/user_recordings/{USER}/good_voice"
    )

elif MODE == "bad":

    USER_FOLDER = (
        f"../dataset/user_recordings/{USER}/bad_voice"
    )

else:

    print("Invalid mode.")
    exit()


print("=" * 60)
print(f"Pronunciation Evaluation ({USER} - {MODE.upper()})")
print("=" * 60)

# =====================================================
# Automatically detect recordings
# =====================================================

audio_files = sorted([
    f for f in os.listdir(USER_FOLDER)
    if f.endswith(".wav")
])

if len(audio_files) == 0:
    print("No recordings found.")
    exit()

# =====================================================
# Evaluate
# =====================================================

for filename in audio_files:

    number = "".join(
        filter(str.isdigit, filename)
    )

    reference_file = os.path.join(
        REFERENCE_FOLDER,
        f"Voice{number}.wav"
    )

    user_file = os.path.join(
        USER_FOLDER,
        filename
    )

    if not os.path.exists(reference_file):

        print(f"Voice{number} -> Reference missing")
        continue

    dtw_distance = calculate_dtw_distance(
        reference_file,
        user_file
    )

    print(
        f"Voice{int(number):03d} -> DTW Distance : {dtw_distance:.4f}"
    )

print("=" * 60)
print("Evaluation Completed")
print("=" * 60)