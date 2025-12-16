import joblib
import numpy as np
from extract_feature import extract_features
import librosa

filepath = "C:\\Users\\ytsub\\Desktop\\github\\SpeachNonSpeachDeticition\\static\\uploads\\1762358453_recording.wav"
model = joblib.load("speech_non_speech_rf.joblib")
print(type(model))
dummy_input = np.zeros((1, model.n_features_in_))
# print("Expected feature shape:", dummy_input.shape)


audio, sr = librosa.load(filepath, sr=22050, mono=True, duration=30)
print(audio, sr)


features = extract_features(audio, sr, n_mfcc=13)
predict = model.predict([features])
print(predict)