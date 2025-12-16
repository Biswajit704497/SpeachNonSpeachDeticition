from flask import Blueprint, render_template, request
from packages.audioModel import extract_speech_non_speech_features
from models.speechToText import audio_to_text
from models.extract_feature import extract_features
import uuid
import os
import joblib
import librosa
model = joblib.load("C:\\Users\\ytsub\\Desktop\\github\\SpeachNonSpeachDeticition\\routes\\speech_non_speech_rf.joblib")
audio_bp = Blueprint("audio_bp", __name__)

upload_folder = "static/uploads"

@audio_bp.route("/audio_route", methods=["GET", "POST"])
def audio_route():
    if request.method == "POST":
        file = request.files['audio']
        print(file)
        if('audio' not in request.files):
            print("file not found")

        elif('audio' in request.files):
            fileName = str(uuid.uuid4()) + '.mp3'
            file_path = os.path.join(upload_folder, fileName)
            file.save(file_path)
            # feature = extract_speech_non_speech_features(file_path)
            # print(feature)
            
            print("file uplaod successfully")
            
            # audio, sr = librosa.load(file_path, sr=22050, mono=True, duration=30)
            
            # features = extract_features(audio, sr, n_mfcc=13)
            # predict = model.predict([features])
            # print(predict)
            
           
            
            
            # transcript = audio_to_text(file_path)
            return render_template("audio.html",audio_url=f"uploads/{fileName}",)


    return render_template("audio.html")
