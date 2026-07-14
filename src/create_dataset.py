import os
import re
import pandas as pd

from jiwer import wer, cer
from dtw_distance import calculate_dtw_distance


# -----------------------------
# Paths
# -----------------------------

REFERENCE_FOLDER = "../dataset/app_reference/"

GOOD_AUDIO_FOLDER = "../dataset/user_recordings/good_voice/"
BAD_AUDIO_FOLDER = "../dataset/user_recordings/bad_voice/"

GOOD_ASR_FOLDER = "../dataset/data/asr_output/good/"
BAD_ASR_FOLDER = "../dataset/data/asr_output/bad/"

OUTPUT_FILE = "../dataset/pronunciation_dataset.csv"


# -----------------------------
# Voice IDs
# -----------------------------

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
    123,
    141
]


# -----------------------------
# Load reference sentences
# -----------------------------

df = pd.read_csv(
    "../dataset/app_reference/sentences.csv"
)


# -----------------------------
# Text functions
# -----------------------------

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



def read_text(file):

    with open(
        file,
        "r",
        encoding="utf-8"
    ) as f:
        return f.read()



# -----------------------------
# Feature extraction
# -----------------------------

def get_text_scores(reference, recognized):

    reference = normalize_text(reference)
    recognized = normalize_text(recognized)

    word_error = wer(
        reference,
        recognized
    )

    char_error = cer(
        reference,
        recognized
    )

    return word_error, char_error



# -----------------------------
# Process recordings
# -----------------------------

dataset = []


def process_recordings(
        mode,
        audio_folder,
        asr_folder,
        prefix
):

    print("\nProcessing", mode)

    for vid in voice_ids:

        voice_id = f"Voice{vid}"


        # Reference sentence
        row = df[
            df["audio_id"] == voice_id
        ]

        if row.empty:
            print(
                voice_id,
                "Reference missing"
            )
            continue


        reference = row.iloc[0]["sentence"]


        # Audio file

        audio_file = os.path.join(
            audio_folder,
            f"{prefix}{vid}.wav"
        )


        # ASR file

        text_file = os.path.join(
            asr_folder,
            f"{prefix}{vid}.txt"
        )


        if not os.path.exists(audio_file):

            print(
                voice_id,
                "Audio missing"
            )
            continue


        if not os.path.exists(text_file):

            print(
                voice_id,
                "ASR missing"
            )
            continue


        # DTW

        _, dtw_score = calculate_dtw_distance(
            REFERENCE_FOLDER + f"Voice{vid}.wav",
            audio_file
        )


        # WER CER

        recognized = read_text(
            text_file
        )

        word_error, char_error = get_text_scores(
            reference,
            recognized
        )


        dataset.append({

            "voice": vid,

            "dtw": round(
                dtw_score,
                4
            ),

            "wer": round(
                word_error,
                4
            ),

            "cer": round(
                char_error,
                4
            ),

            "label": mode

        })


        print(
            voice_id,
            "completed"
        )



# -----------------------------
# Run
# -----------------------------

process_recordings(
    "Good",
    GOOD_AUDIO_FOLDER,
    GOOD_ASR_FOLDER,
    "user_Voice"
)


process_recordings(
    "Bad",
    BAD_AUDIO_FOLDER,
    BAD_ASR_FOLDER,
    "user_badVoice"
)



# -----------------------------
# Save CSV
# -----------------------------

result = pd.DataFrame(
    dataset
)


result.to_csv(
    OUTPUT_FILE,
    index=False
)


print("\n" + "="*50)
print("Dataset Created Successfully")
print("="*50)

print(result)

print("\nSaved at:")
print(OUTPUT_FILE)