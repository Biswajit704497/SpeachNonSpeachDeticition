import os
from typing import List, Tuple
import numpy as np
import librosa


def find_audio_files(root_dir: str, subfolders: List[str]) -> List[str]:
    """Recursively find audio files in the given root_dir inside any of the subfolders.
    Accepts common audio extensions.
    """
    exts = {'.wav', '.flac', '.mp3', '.m4a', '.ogg'}
    files = []
    for sub in subfolders:
        path = os.path.join(root_dir, sub)
        if not os.path.exists(path):
            continue
        for root, _, filenames in os.walk(path):
            for fn in filenames:
                if os.path.splitext(fn)[1].lower() in exts:
                    files.append(os.path.join(root, fn))
    return files


def load_audio(path: str, sr: int = 16000, duration: float | None = None) -> Tuple[np.ndarray, int]:
    """Load an audio file with librosa.
    If duration is set, it will load up to that many seconds (truncating/padding handled later).
    """
    audio, sr = librosa.load(path, sr=sr, mono=True, duration=duration)
    return audio, sr


def extract_mfcc_features(y: np.ndarray, sr: int, n_mfcc: int = 13, max_len: int = 16000*3) -> np.ndarray:
    """Extract frame-level MFCCs + deltas, then return summary statistics (mean, std) as a 1D feature vector.
    max_len is used only if you want to pad/truncate raw audio before computing features.
    """
    # Optionally pad/truncate to max_len samples for consistent analysis
    if max_len is not None:
        if len(y) < max_len:
            y = np.pad(y, (0, max_len - len(y)))
        else:
            y = y[:max_len]

    mfcc = librosa.feature.mfcc(y=y, sr=sr, n_mfcc=n_mfcc)
    delta = librosa.feature.delta(mfcc)
    delta2 = librosa.feature.delta(mfcc, order=2)

    # Aggregate per-coefficient statistics
    features = []
    for mat in (mfcc, delta, delta2):
        mean = np.mean(mat, axis=1)
        std = np.std(mat, axis=1)
        features.append(mean)
        features.append(std)

    feat = np.concatenate(features)
    return feat
