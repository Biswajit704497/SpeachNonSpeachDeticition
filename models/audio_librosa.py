import librosa
import matplotlib.pyplot as plt
import numpy as np

#create a window
plt.figure(figsize=(10, 4))

#load the audio model
waveform, sample_rate = librosa.load('C:\\Users\\Hp\\Desktop\\college project\\SpeachNonSpeachDeticition\\static\\uploads\\1762358453_recording.wav')

librosa.display.waveshow(waveform, sr=sample_rate, color="blue")


# stft = librosa.stft(waveform)
# spectrogram = librosa.amplitude_to_db(np.abs(stft))
# librosa.display.specshow(spectrogram, sr=sample_rate, x_axis='time', y_axis='log')

plt.tight_layout()

plt.show()