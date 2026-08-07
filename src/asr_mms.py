import os
import warnings

# Optional: Use locally cached model only
os.environ["HF_HUB_OFFLINE"] = "1"
os.environ["TRANSFORMERS_OFFLINE"] = "1"

from urllib3.exceptions import NotOpenSSLWarning
warnings.filterwarnings("ignore", category=NotOpenSSLWarning)

import torch
import librosa
from transformers import AutoProcessor, Wav2Vec2ForCTC

# ============================================================
# Configuration
# ============================================================

MODEL_ID = "facebook/mms-1b-all"
LANGUAGE = "npi"
SAMPLE_RATE = 16000

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# ============================================================
# Load model once
# ============================================================

print("Loading MMS model...")

processor = AutoProcessor.from_pretrained(MODEL_ID)
processor.tokenizer.set_target_lang(LANGUAGE)

model = Wav2Vec2ForCTC.from_pretrained(MODEL_ID)
model.load_adapter(LANGUAGE)
model.to(device)
model.eval()

print("MMS model loaded successfully.")

# ============================================================
# Transcription Function
# ============================================================

def transcribe(audio_path):
    """
    Transcribe a Nepali speech audio file using MMS.

    Parameters
    ----------
    audio_path : str
        Path to the WAV audio file.

    Returns
    -------
    str
        Recognized Nepali text.
    """

    if not os.path.exists(audio_path):
        raise FileNotFoundError(f"Audio file not found:\n{audio_path}")

    speech, sr = librosa.load(
        audio_path,
        sr=SAMPLE_RATE,
        mono=True,
    )

    inputs = processor(
        speech,
        sampling_rate=SAMPLE_RATE,
        return_tensors="pt",
    )

    inputs = {k: v.to(device) for k, v in inputs.items()}

    with torch.no_grad():
        logits = model(**inputs).logits

    predicted_ids = torch.argmax(logits, dim=-1)

    transcription = processor.batch_decode(
        predicted_ids,
        skip_special_tokens=True,
    )[0]

    return transcription.strip()


# ============================================================
# Test
# ============================================================

if __name__ == "__main__":

    audio = input("Audio path: ").strip()

    text = transcribe(audio)

    print("\n==============================")
    print("Recognized Text")
    print("==============================")
    print(text)
