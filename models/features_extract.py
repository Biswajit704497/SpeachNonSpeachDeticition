import librosa
import numpy as np
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
