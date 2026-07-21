import os
import re
import pandas as pd
from jiwer import wer, cer

# ----------------------------------
# User Selection
# ----------------------------------

USER = input("Enter user folder (user1/user2/user3): ").strip()
MODE = input("Enter mode (good/bad): ").strip().lower()

# ----------------------------------
# Load Reference Sentences
# ----------------------------------

reference_df = pd.read_csv(
    "../dataset/app_reference/sentences.csv"
)

# ----------------------------------
# Select ASR Folder
# ----------------------------------

if MODE == "good":

    PREFIX = "user_Voice"

    ASR_FOLDER = (
        f"../dataset/data/asr_output/{USER}/good/"
    )

else:

    PREFIX = "user_badVoice"

    ASR_FOLDER = (
        f"../dataset/data/asr_output/{USER}/bad/"
    )

# ----------------------------------
# Text Processing
# ----------------------------------

def normalize_text(text):

    text = text.lower()

    text = re.sub(
        r"[।,!?;:\"'()\-\[\]{}]",
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


print("=" * 60)
print(f"ASR Evaluation ({USER} - {MODE.upper()})")
print("=" * 60)

# ----------------------------------
# Automatically Detect ASR Files
# ----------------------------------

text_files = sorted([
    f for f in os.listdir(ASR_FOLDER)
    if f.endswith(".txt")
])

for filename in text_files:

    number = "".join(filter(str.isdigit, filename))

    voice_id = f"Voice{number}"

    row = reference_df[
        reference_df["audio_id"] == voice_id
    ]

    if row.empty:
        print(f"{voice_id} -> Reference not found")
        continue

    reference = row.iloc[0]["sentence"]

    asr_file = os.path.join(
        ASR_FOLDER,
        filename
    )

    recognized = read_text(asr_file)

    reference = normalize_text(reference)
    recognized = normalize_text(recognized)

    try:

        word_error = wer(reference, recognized)
        char_error = cer(reference, recognized)

        word_accuracy = max(
            0,
            (1 - word_error) * 100
        )

        char_accuracy = max(
            0,
            (1 - char_error) * 100
        )

        print("\n" + "-" * 60)
        print(voice_id)
        print("-" * 60)

        print("Reference:")
        print(reference)

        print("\nRecognized:")
        print(recognized)

        print(f"\nWER                : {word_error:.4f}")
        print(f"Word Accuracy      : {word_accuracy:.2f}%")

        print(f"CER                : {char_error:.4f}")
        print(f"Character Accuracy : {char_accuracy:.2f}%")

    except Exception as e:

        print(f"{voice_id} -> Error : {e}")

print("\n" + "=" * 60)
print("Evaluation Completed")
print("=" * 60)