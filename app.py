from flask import Flask
from routes.home_bp import home_bp
app = Flask(__name__)
app.secret_key = "asdniubasdb45454d15545415256v5vv5v5dsdf5@4ds5"

app.register_blueprint( home_bp)

if __name__ == "__main__":
    app.run(debug=True)