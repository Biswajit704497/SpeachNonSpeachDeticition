Speech / Non-speech classification utilities

Overview
--------
This folder contains simple scripts to prepare features from an audio dataset and train a RandomForest classifier to distinguish speech vs. non-speech (noise).

Files
-----
- `utils.py` - helper functions: find audio files, load audio, and extract MFCC-based features.
- `data_prep.py` - scans dataset folders (speech/noise), extracts features, and saves `ModelCreation/data/features.npz`.
- `train_model.py` - trains a RandomForest classifier from the features and saves the model to `ModelCreation/models/`.
- `requirements.txt` - Python packages used by the scripts.

Quick usage (PowerShell on Windows)
----------------------------------
1. (Optional) Create and activate a virtualenv. In PowerShell:

```powershell
python -m venv .venv; .\.venv\Scripts\Activate.ps1
```

2. Install packages for this folder:

```powershell
pip install -r ModelCreation\requirements.txt
```

3. Prepare features (replace dataset path):

```powershell
python ModelCreation\data_prep.py --dataset "D:\\SpeachNonSpeachDEtection\\dataset\\musan"
```

This will save `ModelCreation/data/features.npz`.

4. Train the model:

```powershell
python ModelCreation\train_model.py --features ModelCreation\data\features.npz
```

A model file `speech_non_speech_rf.joblib` will be saved in `ModelCreation/models/`.

Notes and assumptions
---------------------
- The scripts look for speech folders named `speech` (also accepts `speach`) and noise folders named `noise` (also accepts `noice`). If your dataset uses different folder names, pass a dataset root that contains these folders or modify `data_prep.py`.
- Feature extraction uses MFCC summary statistics (mean/std) over a 3-second window by default. This is simple and fast. For better performance, consider training a CNN on spectrogram patches or using longer context.

Next steps
----------
- Add a small notebook demonstrating end-to-end usage (optional).
- Add model evaluation plots and a small inference script to run on arbitrary wav files.
