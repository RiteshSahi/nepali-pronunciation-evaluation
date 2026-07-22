import os
import re
import pandas as pd

from jiwer import wer, cer
from dtw_distance import calculate_dtw_distance
from feature_extraction import (
    get_duration,
    get_zcr
)


# =====================================================
# Select User
# =====================================================

USER = input("Enter user folder (user1/user2/user3): ").strip()


# =====================================================
# Paths
# =====================================================

REFERENCE_FOLDER = "../dataset/app_reference/"

GOOD_AUDIO_FOLDER = f"../dataset/user_recordings/{USER}/good_voice/"
BAD_AUDIO_FOLDER = f"../dataset/user_recordings/{USER}/bad_voice/"

GOOD_ASR_FOLDER = f"../dataset/data/asr_output/{USER}/good/"
BAD_ASR_FOLDER = f"../dataset/data/asr_output/{USER}/bad/"

OUTPUT_FILE = f"../dataset/{USER}_pronunciation_dataset.csv"


# =====================================================
# Load Sentences
# =====================================================

reference_df = pd.read_csv(
    "../dataset/app_reference/sentences.csv"
)


# =====================================================
# Text Processing
# =====================================================

def normalize_text(text):

    text = text.lower()

    text = re.sub(
        r"[।,!?]",
        "",
        text
    )

    text = re.sub(
        r"\s+",
        " ",
        text
    )

    return text.strip()


def read_text(file_path):

    with open(
        file_path,
        "r",
        encoding="utf-8"
    ) as f:

        return f.read()


# =====================================================
# WER CER
# =====================================================

def get_text_scores(reference, recognized):

    reference = normalize_text(reference)
    recognized = normalize_text(recognized)

    return (
        wer(reference, recognized),
        cer(reference, recognized)
    )


# =====================================================
# Dataset
# =====================================================

dataset = []


def process_recordings(label, audio_folder, asr_folder):

    print(f"\nProcessing {label}")

    if not os.path.exists(audio_folder):
        print(audio_folder, "does not exist")
        return

    audio_files = sorted([
        f for f in os.listdir(audio_folder)
        if f.endswith(".wav")
    ])

    for filename in audio_files:

        number = "".join(filter(str.isdigit, filename))

        voice_id = f"Voice{number}"

        row = reference_df[
            reference_df["audio_id"] == voice_id
        ]

        if row.empty:
            print(f"{voice_id} -> Reference Missing")
            continue

        reference_sentence = row.iloc[0]["sentence"]

        reference_audio = os.path.join(
            REFERENCE_FOLDER,
            f"{voice_id}.wav"
        )

        user_audio = os.path.join(
            audio_folder,
            filename
        )

        text_file = os.path.join(
            asr_folder,
            filename.replace(".wav", ".txt")
        )

        if not os.path.exists(reference_audio):
            print(f"{voice_id} -> Reference Audio Missing")
            continue

        if not os.path.exists(text_file):
            print(f"{voice_id} -> ASR Missing")
            continue

        # -----------------------------
        # DTW
        # -----------------------------

        dtw_score = calculate_dtw_distance(
            reference_audio,
            user_audio
        )

        # -----------------------------
        # Duration
        # -----------------------------

        reference_duration = get_duration(
            reference_audio
        )

        user_duration = get_duration(
            user_audio
        )

        duration_diff = abs(
            reference_duration -
            user_duration
        )
        #ZCR
        reference_zcr = get_zcr(
            reference_audio
        )

        user_zcr = get_zcr(
            user_audio
        )

        zcr_diff = abs(
            reference_zcr - user_zcr
        )
        # -----------------------------
        # ASR
        # -----------------------------

        recognized = read_text(
            text_file
        )

        word_error, char_error = get_text_scores(
            reference_sentence,
            recognized
        )

        # -----------------------------
        # Save
        # -----------------------------

        dataset.append({

            "voice": int(number),

            "dtw": round(dtw_score, 4),

            "duration": round(duration_diff, 4),
            
            "zcr": round(zcr_diff, 4),

            "wer": round(word_error, 4),

            "cer": round(char_error, 4),

            "label": label

        })

        print(f"{voice_id} -> Completed")


# =====================================================
# Process
# =====================================================

process_recordings(
    "Good",
    GOOD_AUDIO_FOLDER,
    GOOD_ASR_FOLDER
)

process_recordings(
    "Bad",
    BAD_AUDIO_FOLDER,
    BAD_ASR_FOLDER
)


# =====================================================
# Save Dataset
# =====================================================

if len(dataset) == 0:
    print("No recordings found.")
    exit()

result = pd.DataFrame(dataset)

result = result.sort_values(
    by=["label", "voice"]
).reset_index(drop=True)

result.to_csv(
    OUTPUT_FILE,
    index=False
)

print("\n" + "=" * 60)
print("Dataset Created Successfully")
print("=" * 60)

print(result)

print("\nSaved to:")
print(OUTPUT_FILE)
