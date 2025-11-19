"""
Data preparation script.
Scans the user-provided dataset root for speech and noise folders, extracts features and saves a features.npz file.
"""
import os
from pathlib import Path
import numpy as np
from tqdm import tqdm

from utils import find_audio_files, load_audio, extract_mfcc_features


def prepare_features(dataset_root: str,
                     speech_folders=('speech', 'speach'),
                     noise_folders=('noise', 'noice'),
                     sr: int = 16000,
                     max_duration_seconds: int = 3,
                     n_mfcc: int = 13,
                     out_dir: str | None = None):
    """Scan dataset and prepare features.
    Saves npz with X (features) and y (labels: 1 for speech, 0 for noise) to out_dir/data/features.npz by default.
    """
    root = Path(dataset_root)
    if not root.exists():
        raise FileNotFoundError(f"Dataset root not found: {dataset_root}")

    # Find files
    speech_files = find_audio_files(dataset_root, list(speech_folders))
    noise_files = find_audio_files(dataset_root, list(noise_folders))

    print(f"Found {len(speech_files)} speech files and {len(noise_files)} noise files.")

    files = [(p, 1) for p in speech_files] + [(p, 0) for p in noise_files]

    if not files:
        raise RuntimeError("No audio files found. Check the dataset path and folder names (speech/noise).")

    X = []
    y = []
    max_len = sr * max_duration_seconds

    for path, label in tqdm(files, desc="Extracting features"):
        try:
            audio, _ = load_audio(path, sr=sr, duration=max_duration_seconds)
            feat = extract_mfcc_features(audio, sr=sr, n_mfcc=n_mfcc, max_len=max_len)
            X.append(feat)
            y.append(label)
        except Exception as e:
            print(f"Failed to process {path}: {e}")

    X = np.vstack(X)
    y = np.array(y, dtype=np.int64)

    if out_dir is None:
        out_dir = Path(__file__).resolve().parent / 'data'
    out_dir = Path(out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    out_path = out_dir / 'features.npz'
    np.savez_compressed(out_path, X=X, y=y)
    print(f"Saved features to {out_path}")


if __name__ == '__main__':
    import argparse

    parser = argparse.ArgumentParser(description='Prepare features for speech/non-speech classification')
    parser.add_argument('--dataset', type=str, required=True, help='Path to dataset root (e.g. D:\\dataset\\musan)')
    parser.add_argument('--out', type=str, default=None, help='Output directory for features (default: ModelCreation/data)')
    parser.add_argument('--sr', type=int, default=16000, help='Sampling rate to use')
    parser.add_argument('--max-dur', type=int, default=3, help='Max duration (seconds) to use per file')
    args = parser.parse_args()

    prepare_features(args.dataset, sr=args.sr, max_duration_seconds=args.max_dur, out_dir=args.out)
