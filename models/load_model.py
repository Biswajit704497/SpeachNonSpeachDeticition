import joblib
import os

# Get the directory of this file (models/)
MODEL_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
MODEL_PATH = os.path.join(MODEL_DIR, "routes", "speech_non_speech_rf.joblib")

_model = None

def load_model():
    """Load the model if not already loaded."""
    global _model
    if _model is None:
        if os.path.exists(MODEL_PATH):
            _model = joblib.load(MODEL_PATH)
        else:
            raise FileNotFoundError(f"Model file not found at: {MODEL_PATH}")
    return _model