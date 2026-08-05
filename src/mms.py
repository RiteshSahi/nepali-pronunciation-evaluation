import torch
import librosa
from transformers import AutoProcessor, Wav2Vec2ForCTC

MODEL_ID = "facebook/mms-1b-all"

print("Loading model...")

processor = AutoProcessor.from_pretrained(MODEL_ID)
processor.tokenizer.set_target_lang("npi")

model = Wav2Vec2ForCTC.from_pretrained(MODEL_ID)
model.load_adapter("npi")
model.eval()

audio_path = input("Audio path: ")

speech, sr = librosa.load(audio_path, sr=16000)

inputs = processor(
    speech,
    sampling_rate=16000,
    return_tensors="pt"
)

with torch.no_grad():
    outputs = model(**inputs).logits

predicted_ids = torch.argmax(outputs, dim=-1)

transcription = processor.batch_decode(
    predicted_ids,
    skip_special_tokens=True
)[0]

print("\nTranscription:")
print(transcription)
