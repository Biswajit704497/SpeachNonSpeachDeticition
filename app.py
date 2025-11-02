from flask import Flask
from routes.home_bp import home_bp
from routes.audio_process import audio_bp
import os
app = Flask(__name__)
app.secret_key = "asdniubasdb45454d15545415256v5vv5v5dsdf5@4ds5"

app.register_blueprint( home_bp)
app.register_blueprint(audio_bp)

if __name__ == "__main__":
    port = int(os.environ.get("PORT", 5000))
    app.run(host="0.0.0.0", port=port, debug=True)
