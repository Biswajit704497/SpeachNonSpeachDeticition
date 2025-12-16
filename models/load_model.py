import joblib
MODEL_PATH = "C:\\Users\\ytsub\\Desktop\\github\\SpeachNonSpeachDeticition\\routes\\speech_non_speech_rf.joblib"
import os
_model=None
def load_model():
    """Load the model if not already loaded."""
    global _model
    if _model is None:
        if os.path.exists(MODEL_PATH):
            _model = joblib.load(MODEL_PATH)
        else:
            raise FileNotFoundError(f"Model file not found at: {MODEL_PATH}")
    return _model