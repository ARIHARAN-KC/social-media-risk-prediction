import os
from flask import Flask, request, jsonify, render_template
from flask_cors import CORS
from config import Config
from src.predict import predict_risk

app = Flask(__name__)
app.config.from_object(Config)

CORS(app)

# UI ROUTE
@app.route("/")
def home():
    return render_template("index.html")

# API ROUTE

@app.route("/predict", methods=["POST"])
def predict():
    try:
        data = request.get_json()

        if not data or "text" not in data:
            return jsonify({"error": "Missing text field"}), 400

        result = predict_risk(data["text"])
        return jsonify(result)

    except Exception as e:
        return jsonify({"error": str(e)}), 500

@app.route("/how-it-works")
def how_it_works():
    return render_template("how_it_works.html")

if __name__ == "__main__":
    app.run(debug=True)