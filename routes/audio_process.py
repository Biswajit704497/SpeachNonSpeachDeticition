from flask import Flask, Blueprint, render_template, request, jsonify
import os
import numpy as np
import librosa
import joblib
import tempfile
from werkzeug.utils import secure_filename

audio_bp = Blueprint("audio_bp", __name__)

# Configure upload settings
UPLOAD_FOLDER = tempfile.gettempdir()
ALLOWED_EXTENSIONS = {'wav', 'mp3', 'm4a', 'flac', 'ogg'}
MAX_FILE_SIZE = 100 * 1024 * 1024  # 100MB

# Model path
MODEL_PATH = "C:\\Users\\Hp\\Desktop\\college project\\SpeachNonSpeachDeticition\\routes\\speech_non_speech_rf.joblib"

# Load model once when module is imported
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

def allowed_file(filename):
    """Check if file extension is allowed."""
    return '.' in filename and filename.rsplit('.', 1)[1].lower() in ALLOWED_EXTENSIONS

def extract_features(audio, sr, n_mfcc=13):
    """Extract features from audio (same as predict_from_mic.py).
    
    Returns concatenation of mean and std for:
    - MFCCs
    - MFCC deltas (1st derivative)
    - MFCC delta-deltas (2nd derivative)
    
    For n_mfcc=13 this produces 13 * 2 * 3 = 78 features.
    """
    audio = audio.astype('float32')
    
    # Pad very short audio to avoid errors
    if audio.size < 2048:
        audio = np.pad(audio, (0, 2048 - audio.size), mode='constant')
    
    # Compute MFCCs
    mfcc = librosa.feature.mfcc(y=audio, sr=sr, n_mfcc=n_mfcc)
    # Deltas
    delta = librosa.feature.delta(mfcc)
    delta2 = librosa.feature.delta(mfcc, order=2)
    
    # Aggregate: mean and std for each set
    def agg_feats(mat):
        return np.hstack([np.mean(mat, axis=1), np.std(mat, axis=1)])
    
    feats_mfcc = agg_feats(mfcc)
    feats_delta = agg_feats(delta)
    feats_delta2 = agg_feats(delta2)
    
    features = np.hstack([feats_mfcc, feats_delta, feats_delta2]).astype(np.float32)
    return features

@audio_bp.route("/audio")
def audio():
    return render_template("audio_page.html")

@audio_bp.route("/predict", methods=["POST","GET"])
def predict():
    """Handle audio file upload and return prediction."""
    try:
        # Check if file is present
        if 'audio' not in request.files:
            return jsonify({'error': 'No audio file provided'}), 400
        
        file = request.files['audio']
        
        if file.filename == '':
            return jsonify({'error': 'No file selected'}), 400
        
        # Validate file
        if not allowed_file(file.filename):
            return jsonify({'error': 'File type not allowed. Allowed types: WAV, MP3, M4A, FLAC, OGG'}), 400
        
        # Save temporary file
        filename = secure_filename(file.filename)
        filepath = os.path.join(UPLOAD_FOLDER, filename)
        file.save(filepath)
        
        try:
            # Load audio with librosa
            audio, sr = librosa.load(filepath, sr=22050, mono=True, duration=30)
            
            # Extract features
            features = extract_features(audio, sr, n_mfcc=13)
            
            # Load model and predict
            model = load_model()
            
            # Reshape for prediction (model expects 2D array)
            X = features.reshape(1, -1)
            
            # Make prediction
            prediction = model.predict(X)[0]
            
            # Map prediction to human-readable label
            # Model uses: 0 = Non-Speech, 1 = Speech
            def map_class_to_label(class_val):
                """Map numeric class to readable label."""

                class_val = int(class_val) if isinstance(class_val, (int, float, str)) else class_val
                if class_val == 0:
                    return "Non-Speech"
                elif class_val == 1:
                    return "Speech"
                else:
                    return str(class_val)
            
            # Get label for prediction
            result_label = map_class_to_label(prediction)
            
            
            # Get probabilities if available
            result = {
                'success': True,
                'prediction': str(prediction),
                'label': result_label
            }
            
            if hasattr(model, 'predict_proba'):
                proba = model.predict_proba(X)[0]
                if hasattr(model, 'classes_'):
                    # Map class values to labels
                    probabilities_dict = {}
                    classes_list = list(model.classes_)
                    
                    for i, class_val in enumerate(classes_list):
                        label = map_class_to_label(class_val)
                        probabilities_dict[label] = float(proba[i])
                    
                    result['probabilities'] = probabilities_dict
                    
                    # Get confidence for the predicted class
                    pred_idx = list(model.classes_).index(prediction) if prediction in model.classes_ else 0
                    result['confidence'] = float(proba[pred_idx])
                else:
                    result['probabilities'] = proba.tolist()
            
        finally:
            # Clean up temporary file
            if os.path.exists(filepath):
                os.remove(filepath)
        
        return jsonify(result)
        
    except FileNotFoundError as e:
        return jsonify({'error': f'Model not found: {str(e)}'}), 500
    except Exception as e:
        return jsonify({'error': f'Prediction error: {str(e)}'}), 500