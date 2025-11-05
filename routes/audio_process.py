from flask import Flask, Blueprint, render_template, request, jsonify
import os
import numpy as np
import librosa
import joblib
import tempfile
from werkzeug.utils import secure_filename
from models.file_allow import allowed_file
from models.load_model import load_model
from models.features_extract import extract_features

audio_bp = Blueprint("audio_bp", __name__)

# Configure upload settings
UPLOAD_FOLDER = tempfile.gettempdir()
MAX_FILE_SIZE = 100 * 1024 * 1024  # 100MB

@audio_bp.route("/audio")
def audio():
    return render_template("audio_page.html")

@audio_bp.route("/predict", methods=["POST"])
def predict():
    _model = None
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