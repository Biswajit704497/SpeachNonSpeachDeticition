import librosa
import numpy as np

def extract_speech_non_speech_features(file_path):
    y, sr = librosa.load(file_path, sr=22050)
    y, _ = librosa.effects.trim(y)

    features = []

    # MFCC
    mfcc = librosa.feature.mfcc(y=y, sr=sr, n_mfcc=20)
    features.extend(np.mean(mfcc, axis=1))
    features.extend(np.std(mfcc, axis=1))

    # Zero Crossing Rate
    zcr = librosa.feature.zero_crossing_rate(y)
    features.append(np.mean(zcr))

    # RMS Energy
    rms = librosa.feature.rms(y=y)
    features.append(np.mean(rms))

    # Spectral Centroid
    centroid = librosa.feature.spectral_centroid(y=y, sr=sr)
    features.append(np.mean(centroid))

    # Spectral Bandwidth
    bandwidth = librosa.feature.spectral_bandwidth(y=y, sr=sr)
    features.append(np.mean(bandwidth))

    # Spectral Roll-off
    rolloff = librosa.feature.spectral_rolloff(y=y, sr=sr)
    features.append(np.mean(rolloff))

    return np.array(features)


def extract_emotion_features(file_path):
    y, sr = librosa.load(file_path, sr=22050)
    y, _ = librosa.effects.trim(y)

    features = []

    # MFCC
    mfcc = librosa.feature.mfcc(y=y, sr=sr, n_mfcc=40)
    mfcc_delta = librosa.feature.delta(mfcc)
    mfcc_delta2 = librosa.feature.delta(mfcc, order=2)

    for feature in [mfcc, mfcc_delta, mfcc_delta2]:
        features.extend(np.mean(feature, axis=1))
        features.extend(np.std(feature, axis=1))

    # RMS Energy
    rms = librosa.feature.rms(y=y)
    features.append(np.mean(rms))

    # Pitch
    pitch = librosa.yin(y, fmin=50, fmax=400, sr=sr)
    pitch = pitch[pitch > 0]
    features.append(np.mean(pitch) if len(pitch) > 0 else 0)
    features.append(np.std(pitch) if len(pitch) > 0 else 0)

    # Spectral Centroid
    centroid = librosa.feature.spectral_centroid(y=y, sr=sr)
    features.append(np.mean(centroid))

    # Zero Crossing Rate
    zcr = librosa.feature.zero_crossing_rate(y)
    features.append(np.mean(zcr))

    return np.array(features)

def extract_gender_features(file_path):
    y, sr = librosa.load(file_path, sr=22050)
    y, _ = librosa.effects.trim(y)

    features = []

    # MFCC
    mfcc = librosa.feature.mfcc(y=y, sr=sr, n_mfcc=20)
    features.extend(np.mean(mfcc, axis=1))
    features.extend(np.std(mfcc, axis=1))

    # Pitch (Fundamental Frequency)
    pitch = librosa.yin(y, fmin=50, fmax=300, sr=sr)
    pitch = pitch[pitch > 0]  # remove unvoiced frames
    features.append(np.mean(pitch) if len(pitch) > 0 else 0)
    features.append(np.std(pitch) if len(pitch) > 0 else 0)

    # Spectral Centroid
    centroid = librosa.feature.spectral_centroid(y=y, sr=sr)
    features.append(np.mean(centroid))

    # Spectral Bandwidth
    bandwidth = librosa.feature.spectral_bandwidth(y=y, sr=sr)
    features.append(np.mean(bandwidth))

    return np.array(features)

