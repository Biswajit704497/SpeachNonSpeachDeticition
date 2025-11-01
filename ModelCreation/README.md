Predict from Microphone
=======================

This small utility records audio from the default microphone, extracts MFCC-based features, loads a scikit-learn model saved with joblib, and prints a speech / non-speech prediction.

Quick start
-----------

1. Install dependencies (preferably in a virtualenv):

```bash
pip install numpy sounddevice librosa joblib soundfile scikit-learn
```

2. Run the recorder and prediction script:

```bash
python predict_from_mic.py --model path/to/your_model.joblib --duration 3 --n_mfcc 13
```

Notes & assumptions
-------------------
- The script extracts MFCC mean and std (13 MFCCs by default) and concatenates them into a feature vector of length 26. If your trained model used different features (chroma, spectral contrast, delta features, different frame sizes), you must update `predict_from_mic.py` to match the same features.
- If the model has `predict_proba` and `classes_`, the script will print probabilities for each class.
- On Windows you may need to install system dependencies for `sounddevice` and `librosa` (portaudio / soundfile). See each package's docs if you encounter installation errors.

Troubleshooting
---------------
- "Model expects X features but produced Y": update `n_mfcc` or change `extract_features()` to produce the same features used during training.
- If recording fails, check your microphone permissions and try a different sampling rate (e.g. 44100).

If you'd like, I can adapt the script to stream and do continuous predictions or to use a different feature set (MFCC deltas, chroma, etc.).
