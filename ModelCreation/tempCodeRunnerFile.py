"""
Train a classifier on precomputed features and save the trained model.
"""
from pathlib import Path
import numpy as np
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix
import joblib


def train_and_save(features_npz: str | Path,
                   out_model_path: str | Path | None = None,
                   test_size: float = 0.2,
                   random_state: int = 42):
    features_npz = Path(features_npz)
    if not features_npz.exists():
        raise FileNotFoundError(f"Features file not found: {features_npz}")

    data = np.load(features_npz)
    X = data['X']
    y = data['y']

    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=test_size, random_state=random_state, stratify=y)

    clf = RandomForestClassifier(n_estimators=200, random_state=random_state, n_jobs=-1)
    print("Training RandomForest...")
    clf.fit(X_train, y_train)

    y_pred = clf.predict(X_test)
    acc = accuracy_score(y_test, y_pred)
    print(f"Test accuracy: {acc:.4f}")
    print(classification_report(y_test, y_pred))
    print("Confusion matrix:\n", confusion_matrix(y_test, y_pred))

    if out_model_path is None:
        out_model_path = Path(__file__).resolve().parent / 'models'
    else:
        out_model_path = Path(out_model_path)
    out_model_path.mkdir(parents=True, exist_ok=True)
    model_path = out_model_path / 'speech_non_speech_rf.joblib'

    joblib.dump(clf, model_path)
    print(f"Saved model to {model_path}")
    return model_path


if __name__ == '__main__':
    import argparse

    parser = argparse.ArgumentParser(description='Train speech/non-speech classifier from features')
    # Default to ModelCreation/data/features.npz if not provided
    default_features = Path(__file__).resolve().parent / 'data' / 'features.npz'
    parser.add_argument('--features', type=str, default=str(default_features), required=False,
                        help=f'Path to features.npz (default: {default_features})')
    parser.add_argument('--out', type=str, default=None, help='Directory where to save model')
    args = parser.parse_args()

    features_path = Path(args.features)
    if not features_path.exists():
        raise FileNotFoundError(f"Features file not found: {features_path} -- run data_prep.py first or provide --features")

    print(f"Using features file: {features_path}")
    train_and_save(features_path, out_model_path=args.out)
