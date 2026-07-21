import os
import torch
import librosa
from transformers import AutoProcessor, AutoModelForCTC

MODEL_NAME = "anish-shilpakar/wav2vec2-nepali"

# ----------------------------------
# Load Model
# ----------------------------------

processor = AutoProcessor.from_pretrained(MODEL_NAME)
model = AutoModelForCTC.from_pretrained(MODEL_NAME)
model.eval()


def transcribe_audio(file_path):

    audio, _ = librosa.load(
        file_path,
        sr=16000,
        mono=True
    )

    inputs = processor(
        audio,
        sampling_rate=16000,
        return_tensors="pt"
    )

    with torch.no_grad():
        logits = model(inputs.input_values).logits

    predicted_ids = torch.argmax(
        logits,
        dim=-1
    )

    text = processor.batch_decode(
        predicted_ids
    )[0]

    return text


# ----------------------------------
# Voice IDs
# ----------------------------------

voice_numbers = [
    64, 67, 71, 72, 76, 78,
    80, 83, 84, 85, 92, 94,
    95, 97, 102, 109, 113,
    117, 118, 121, 123,
    137, 141, 149, 151, 167
]


# ----------------------------------
# Select User
# ----------------------------------

user_name = input(
    "Enter user folder (user1/user2/user3): "
).strip()


datasets = [

    {
        "name": "GOOD",
        "input_folder": f"../dataset/user_recordings/{user_name}/good_voice",
        "output_folder": f"../dataset/data/asr_output/{user_name}/good"
    },

    {
        "name": "BAD",
        "input_folder": f"../dataset/user_recordings/{user_name}/bad_voice",
        "output_folder": f"../dataset/data/asr_output/{user_name}/bad"
    }

]


print("=" * 60)
print("Generating ASR Text Files")
print("=" * 60)

for data in datasets:

    print(f"\nProcessing {data['name']} Recordings")
    print("-" * 60)

    os.makedirs(
        data["output_folder"],
        exist_ok=True
    )

    for number in voice_numbers:

        audio_file = os.path.join(
            data["input_folder"],
            f"Voice{number}.wav"
        )

        output_file = os.path.join(
            data["output_folder"],
            f"Voice{number}.txt"
        )

        if not os.path.exists(audio_file):

            print(f"Voice{number} -> Not Found")
            continue

        try:

            text = transcribe_audio(audio_file)

            with open(
                output_file,
                "w",
                encoding="utf-8"
            ) as f:
                f.write(text)

            print(f"Voice{number} -> Saved")

        except Exception as e:

            print(f"Voice{number} -> Error: {e}")


print("\n" + "=" * 60)
print("ASR Generation Completed")
print("=" * 60)