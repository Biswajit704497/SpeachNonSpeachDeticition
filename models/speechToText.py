import whisper

model = whisper.load_model("base")  # small / base / medium

def audio_to_text(audio_path):
    result = model.transcribe(audio_path)
    return result["text"]
