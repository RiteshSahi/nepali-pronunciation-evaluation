import os
import torch
import librosa
from transformers import AutoProcessor, AutoModelForCTC


MODEL_NAME = "anish-shilpakar/wav2vec2-nepali"

# Load model only once
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
        logits = model(
            inputs.input_values
        ).logits

    predicted_ids = torch.argmax(
        logits,
        dim=-1
    )

    text = processor.batch_decode(
        predicted_ids
    )[0]

    return text


voice_numbers = [
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


datasets = [
    {
        "name": "GOOD",
        "input_folder": "../dataset/user_recordings/good_voice",
        "output_folder": "../dataset/data/asr_output/good",
        "prefix": "user_Voice"
    },
    {
        "name": "BAD",
        "input_folder": "../dataset/user_recordings/bad_voice",
        "output_folder": "../dataset/data/asr_output/bad",
        "prefix": "user_badVoice"
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
            f"{data['prefix']}{number}.wav"
        )

        output_file = os.path.join(
            data["output_folder"],
            f"{data['prefix']}{number}.txt"
        )

        if not os.path.exists(audio_file):
            print(f"Voice{number} -> Recording not found")
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