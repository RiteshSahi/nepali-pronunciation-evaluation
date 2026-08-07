import os
import re
import pandas as pd

from jiwer import wer, cer

from dtw_distance import calculate_dtw_distance
from feature_extraction import (
    get_duration,
    get_zcr
)
from asr_mms import transcribe


# =====================================================
# Gender Validation
# =====================================================
# Normalizes user/UI-provided gender input ("Male", "female", "MALE", ...)
# into the lowercase folder name used under dataset/. Mirrors the same
# validation pattern used in predict.py so both scripts stay consistent.

VALID_GENDERS = ["male", "female"]


def normalize_gender(raw_gender):

    cleaned = raw_gender.strip().lower()

    if cleaned not in VALID_GENDERS:
        print(f"\nInvalid gender '{raw_gender}'. Expected 'Male' or 'Female'.")
        exit()

    return cleaned


# =====================================================
# Select Gender and User
# =====================================================

GENDER = normalize_gender(
    input("Enter gender (Male/Female): ").strip()
)

USER = input("Enter user folder (user1/user2/user3): ").strip()


# =====================================================
# Paths
# =====================================================

REFERENCE_FOLDER = f"../dataset/{GENDER}/app_reference/"

GOOD_AUDIO_FOLDER = f"../dataset/{GENDER}/{USER}/good_voice/"
BAD_AUDIO_FOLDER = f"../dataset/{GENDER}/{USER}/bad_voice/"

OUTPUT_FILE = f"../dataset/{GENDER}/{USER}_pronunciation_dataset.csv"


# =====================================================
# Validate Reference Folder / Sentences File
# =====================================================
# Without these, nothing downstream can run, so we fail fast with a
# clear message instead of letting pd.read_csv() raise a raw traceback.

if not os.path.exists(REFERENCE_FOLDER):
    print(f"\nReference folder not found: {REFERENCE_FOLDER}")
    exit()

SENTENCE_FILE = os.path.join(REFERENCE_FOLDER, "sentences.csv")

if not os.path.exists(SENTENCE_FILE):
    print(f"\nsentences.csv not found: {SENTENCE_FILE}")
    exit()


# =====================================================
# Load Reference Sentences
# =====================================================

reference_df = pd.read_csv(SENTENCE_FILE)


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


# =====================================================
# WER / CER
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


# =====================================================
# Process Recordings
# =====================================================

def process_recordings(label, audio_folder):

    print(f"\nProcessing {label} Recordings")

    if not os.path.exists(audio_folder):
        print(f"{audio_folder} does not exist.")
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
            print(f"{voice_id} -> Reference sentence missing")
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

        if not os.path.exists(reference_audio):
            print(f"{voice_id} -> Reference audio missing")
            continue

        try:

            # ---------------------------------------
            # DTW
            # ---------------------------------------

            dtw_score = calculate_dtw_distance(
                reference_audio,
                user_audio
            )

            # ---------------------------------------
            # Duration Difference
            # ---------------------------------------

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

            # ---------------------------------------
            # ZCR Difference
            # ---------------------------------------

            reference_zcr = get_zcr(
                reference_audio
            )

            user_zcr = get_zcr(
                user_audio
            )

            zcr_diff = abs(
                reference_zcr -
                user_zcr
            )

            # ---------------------------------------
            # MMS ASR
            # ---------------------------------------

            recognized = transcribe(user_audio)

            print(f"{voice_id}")
            print("Reference :", reference_sentence)
            print("Recognized:", recognized)

            # ---------------------------------------
            # WER / CER
            # ---------------------------------------

            word_error, char_error = get_text_scores(
                reference_sentence,
                recognized
            )

            # ---------------------------------------
            # Save
            # ---------------------------------------

            dataset.append({

                "voice": int(number),

                "dtw": round(dtw_score, 4),

                "duration": round(duration_diff, 4),

                "zcr": round(zcr_diff, 4),

                "wer": round(word_error, 4),

                "cer": round(char_error, 4),

                "label": label

            })

            print(
                f"{voice_id} -> "
                f"DTW={dtw_score:.4f}, "
                f"WER={word_error:.4f}, "
                f"CER={char_error:.4f}"
            )

            print("-" * 60)

        except Exception as e:

            print(f"{voice_id} -> Error: {e}")


# =====================================================
# Process Good and Bad
# =====================================================

process_recordings(
    "Good",
    GOOD_AUDIO_FOLDER
)

process_recordings(
    "Bad",
    BAD_AUDIO_FOLDER
)


# =====================================================
# Save Dataset
# =====================================================

if len(dataset) == 0:

    print("\nNo recordings processed.")

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