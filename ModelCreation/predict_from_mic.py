#!/usr/bin/env python3
"""
Record audio from the default microphone, extract simple features (MFCC mean/std),
load a scikit-learn model saved with joblib, and predict speech vs non-speech.

Assumptions:
- The saved model is a scikit-learn style estimator saved via joblib (joblib.dump).
- The model expects a 1D feature vector built from: [mfcc_mean (n_mfcc), mfcc_std (n_mfcc)]
  (i.e. 2 * n_mfcc features). If your model uses different features, update
  the feature extraction block below.

Install required packages:
    pip install numpy sounddevice librosa joblib soundfile scikit-learn

Usage:
    python predict_from_mic.py --model path/to/model.joblib --duration 3

"""
import argparse
import os
import sys
import time
from datetime import datetime

import joblib
import numpy as np
import sounddevice as sd
import soundfile as sf
import librosa


def record_audio(duration=3, fs=22050, channels=1, out_path=None):
    """Record audio from default microphone for `duration` seconds.
    Returns numpy array (mono) and sample rate.
    Optionally saves recorded WAV to out_path.
    """
    print(f"Recording {duration}s of audio (sr={fs})...")
    recording = sd.rec(int(duration * fs), samplerate=fs, channels=channels, dtype='float32')
    sd.wait()
    # If multi-channel, convert to mono by averaging channels
    if recording.ndim > 1:
        audio = np.mean(recording, axis=1)
    else:
        audio = recording

    audio = audio.flatten()

    if out_path:
        try:
            parent = os.path.dirname(out_path)
            if parent:
                os.makedirs(parent, exist_ok=True)
            sf.write(out_path, audio, fs)
            print(f"Saved recording to {out_path}")
        except Exception as e:
            print("Warning: failed to save recording:", e)

    return audio, fs


def extract_features(audio, sr, n_mfcc=13):
    """Extract features from audio.

    By default this returns the concatenation of mean and std for:
      - MFCCs
      - MFCC deltas (1st derivative)
      - MFCC delta-deltas (2nd derivative)

    For n_mfcc=13 this produces 13 * 2 * 3 = 78 features which matches
    models that were trained with MFCC + delta + delta2 mean/std.

    If your model expects a different feature set, change this function.
    """
    audio = audio.astype('float32')

    # pad very short audio to avoid errors
    if audio.size < 2048:
        audio = np.pad(audio, (0, 2048 - audio.size), mode='constant')

    # compute MFCCs
    mfcc = librosa.feature.mfcc(y=audio, sr=sr, n_mfcc=n_mfcc)
    # deltas
    delta = librosa.feature.delta(mfcc)
    delta2 = librosa.feature.delta(mfcc, order=2)

    # aggregate: mean and std for each set
    def agg_feats(mat):
        return np.hstack([np.mean(mat, axis=1), np.std(mat, axis=1)])

    feats_mfcc = agg_feats(mfcc)
    feats_delta = agg_feats(delta)
    feats_delta2 = agg_feats(delta2)

    features = np.hstack([feats_mfcc, feats_delta, feats_delta2]).astype(np.float32)
    return features


def load_model(path):
    if not os.path.exists(path):
        raise FileNotFoundError(f"Model file not found: {path}")
    model = joblib.load(path)
    return model


def predict(model, features):
    X = features.reshape(1, -1)
    result = {}
    try:
        pred = model.predict(X)
        result['prediction'] = pred[0]
    except Exception as e:
        result['prediction'] = None
        result['error'] = f"predict failed: {e}"

    # Try to get probabilities if available
    try:
        if hasattr(model, 'predict_proba'):
            proba = model.predict_proba(X)[0]
            # Map classes to probabilities if possible
            if hasattr(model, 'classes_'):
                result['proba'] = dict(zip(map(str, model.classes_), proba.tolist()))
            else:
                result['proba'] = proba.tolist()
        elif hasattr(model, 'decision_function'):
            dec = model.decision_function(X)
            result['decision'] = dec.tolist()
    except Exception as e:
        result['proba_error'] = str(e)

    return result


def main():
    parser = argparse.ArgumentParser(description='Record from mic and predict speech vs non-speech using a joblib model')
    parser.add_argument('--model', '-m', required=True, help='Path to joblib model file')
    parser.add_argument('--duration', '-d', type=float, default=3.0, help='Recording duration in seconds (default: 3)')
    parser.add_argument('--sr', type=int, default=22050, help='Sampling rate to record at (default: 22050)')
    parser.add_argument('--n_mfcc', type=int, default=13, help='Number of MFCCs to compute (default: 13)')
    parser.add_argument('--save', '-s', default=r'C:\Users\Hp\Desktop\recordings\my_clip.wav', help='Optional path to save the recorded WAV (e.g. out.wav). Defaults to C:\\Users\\Hp\\Desktop\\recordings\\my_clip.wav')
    parser.add_argument('--labels', help='Optional comma-separated labels in the order of model.classes_. Example: "no speak,suman speak". If omitted the script will default to mapping class 0 -> "no speak" and class 1 -> "suman speak" when possible.')
    args = parser.parse_args()

    try:
        model = load_model(args.model)
    except Exception as e:
        print("Failed to load model:", e)
        sys.exit(2)

    audio, sr = record_audio(duration=args.duration, fs=args.sr, channels=1, out_path=args.save)

    features = extract_features(audio, sr, n_mfcc=args.n_mfcc)

    # If model expects a different feature length, warn user
    try:
        expected = getattr(model, 'n_features_in_', None)
        if expected is not None and expected != features.size:
            print(f"Warning: model expects {expected} features but produced {features.size}.")
            print("If prediction fails, update the feature extraction to match the training features.")
    except Exception:
        pass

    res = predict(model, features)

    print('\n=== Prediction Result ===')
    if 'error' in res:
        print('Prediction error:', res['error'])
    else:
        pred = res.get('prediction')
        # Map numeric class to human-readable label
        label_map = None
        # If user provided labels, use them (must match model.classes_ order)
        if args.labels:
            parts = [p.strip() for p in args.labels.split(',') if p.strip()]
            try:
                classes = list(model.classes_)
                if len(parts) == len(classes):
                    label_map = dict(zip(map(str, classes), parts))
                else:
                    print('Warning: number of labels provided does not match model.classes_. Using default mapping or numeric classes.')
            except Exception:
                print('Warning: could not read model.classes_. Using numeric classes.')

        # If no custom labels provided, attempt sensible default: 0 -> 'no speak', 1 -> 'suman speak'
        if label_map is None:
            try:
                classes = list(model.classes_)
                # Map only when classes are exactly [0,1] or ['0','1'] etc.
                if len(classes) == 2 and set(map(str, classes)) == set(['0', '1']):
                    label_map = {str(classes[0]): 'no speak' if str(classes[0]) == '0' else 'suman speak',
                                 str(classes[1]): 'suman speak' if str(classes[1]) == '1' else 'no speak'}
                else:
                    # fallback: do not map
                    label_map = None
            except Exception:
                label_map = None

        if label_map and str(pred) in label_map:
            print('Prediction:', label_map[str(pred)], f"(raw: {pred})")
        else:
            print('Prediction:', pred)

        if 'proba' in res:
            print('Probabilities:')
            for cls, p in res['proba'].items():
                label = label_map.get(str(cls)) if label_map and str(cls) in label_map else cls
                print(f'  {label}: {p:.3f}')
        if 'decision' in res:
            print('Decision function:', res['decision'])


if __name__ == '__main__':
    main()
