import librosa 
import numpy as np
import os
import joblib


def predict_audio_mfcc(audio_path, model_path, sr=16000):
    import librosa
    import numpy as np
    import joblib

    # Load audio
    y, _ = librosa.load(audio_path, sr=sr)

    # Extract MFCC (SAME parameters as training)
    mfcc = librosa.feature.mfcc(
        y=y,
        sr=sr,
        n_mfcc=13,
        n_fft=int(sr * 0.025),
        hop_length=int(sr * 0.010)
    )

    mfcc = mfcc.T  # SAME as saved training data

    # SAME feature reduction as training (13 MFCC * 6 stats = 78 features)
    from scipy.stats import skew, kurtosis
    features = np.hstack([
        np.mean(mfcc, axis=0),
        np.std(mfcc, axis=0),
        np.min(mfcc, axis=0),
        np.max(mfcc, axis=0),
        skew(mfcc, axis=0),
        kurtosis(mfcc, axis=0)
    ]).reshape(1, -1)

    model = joblib.load(model_path)
    pred = model.predict(features)[0]

    return "Speech" if pred == 1 else "Non-Speech"

audio_path = "C:\\Users\\ytsub\\OneDrive\\ドキュメント\\Sound Recordings\\Recording.m4a"
model_path = "C:\\Users\\ytsub\\Desktop\\github\\SpeachNonSpeachDeticition\\models\\speech_non_speech_rf_old.joblib"

result = predict_audio_mfcc(audio_path,model_path )
print(result)