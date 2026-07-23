import torch
import librosa
from transformers import AutoProcessor, AutoModelForCTC

MODEL_NAME = "anish-shilpakar/wav2vec2-nepali"

# -----------------------------------------------------
# Load Model (loads only once)
# -----------------------------------------------------

processor = AutoProcessor.from_pretrained(MODEL_NAME)
model = AutoModelForCTC.from_pretrained(MODEL_NAME)
model.eval()


# -----------------------------------------------------
# Transcribe Single Audio
# -----------------------------------------------------

def transcribe_audio(audio_file):

    audio, _ = librosa.load(
        audio_file,
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

    return text.strip()