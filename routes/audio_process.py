from flask import Flask, Blueprint, render_template, request, jsonify

audio_bp = Blueprint("audio_bp", __name__)

# Configure upload settings


@audio_bp.route("/audio")
def audio():
    return render_template("audio_page.html")
