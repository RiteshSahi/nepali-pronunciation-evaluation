import os
import re
import pandas as pd
from jiwer import wer, cer

# ----------------------------------
# Change only this
# ----------------------------------
MODE = "bad"      # "good" or "bad"
# ----------------------------------


def normalize_text(text):

    text = text.lower()

    # Remove Nepali punctuation
    text = re.sub(r"[।,!?;:\"'()\-\[\]{}]", "", text)

    # Remove extra spaces
    text = re.sub(r"\s+", " ", text)

    return text.strip()


def read_text(file_path):

    with open(file_path, "r", encoding="utf-8") as f:
        return f.read()


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


# Load reference sentences
df = pd.read_csv("../dataset/app_reference/sentences.csv")

if MODE == "good":
    prefix = "user_Voice"
    asr_folder = "../dataset/data/asr_output/good/"
else:
    prefix = "user_badVoice"
    asr_folder = "../dataset/data/asr_output/bad/"


print("=" * 60)
print(f"ASR Evaluation ({MODE.upper()})")
print("=" * 60)

for vid in voice_ids:

    voice_id = f"Voice{vid}"

    # Find reference sentence
    row = df[df["audio_id"] == voice_id]

    if row.empty:
        print(f"{voice_id} -> Reference sentence not found")
        continue

    reference = row.iloc[0]["sentence"]

    # ASR text file
    asr_file = os.path.join(
        asr_folder,
        f"{prefix}{vid}.txt"
    )

    if not os.path.exists(asr_file):
        print(f"{voice_id} -> ASR file not found")
        continue

    recognized = read_text(asr_file)

    # Normalize
    reference = normalize_text(reference)
    recognized = normalize_text(recognized)

    # Calculate WER and CER
    try:

        word_error = wer(reference, recognized)
        char_error = cer(reference, recognized)

        word_accuracy = max(0, (1 - word_error) * 100)
        char_accuracy = max(0, (1 - char_error) * 100)

        print("\n" + "-" * 60)
        print(voice_id)
        print("-" * 60)

        print("Reference :")
        print(reference)

        print("\nRecognized:")
        print(recognized)

        print(f"\nWER                 : {word_error:.4f}")
        print(f"Word Accuracy       : {word_accuracy:.2f}%")

        print(f"CER                 : {char_error:.4f}")
        print(f"Character Accuracy  : {char_accuracy:.2f}%")

    except Exception as e:
        print(f"{voice_id} -> Error: {e}")

print("\n" + "=" * 60)
print("Evaluation Completed")
print("=" * 60)