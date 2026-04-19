"""
Main web application for the ML Client.
"""

import os
import random

import requests
from flask import Flask, redirect, render_template, request

app = Flask(__name__)

# get the secret key from env
app.config["SECRET_KEY"] = os.getenv("SECRET_KEY", "dev-secret")

ML_CLIENT_PREDICT_URL = os.getenv(
    "ML_CLIENT_PREDICT_URL", "http://localhost:8000/predict"
)
ML_CLIENT_HISTORY_URL = os.getenv(
    "ML_CLIENT_HISTORY_URL", "http://localhost:8000/history"
)
ML_CLIENT_TIMEOUT_SECONDS = float(os.getenv("ML_CLIENT_TIMEOUT_SECONDS", "30"))

things = [
    "airplane",
    "banana",
    "computer",
    "dog",
    "elephant",
    "fish",
    "garden",
    "helmet",
    "ice cream",
    "jail",
    "key",
    "lantern",
    "motorbike",
    "necklace",
    "onion",
    "penguin",
    "raccoon",
    "sandwich",
    "table",
    "underwear",
    "vase",
    "watermelon",
    "yoga",
    "zigzag",
]


def get_random_thing():
    """Return a random item from the list of things."""
    return random.choice(things)


@app.route("/", methods=["GET", "POST"])
def index():
    # pylint: disable=too-many-return-statements
    """Render main page or handle image prediction upload."""
    if request.method == "GET":
        thing = get_random_thing()
        return render_template("index.html", thing=thing)

    if request.method == "POST":
        draw_instruction = request.headers.get("Draw-Instruction", "unknown")
        image_bytes = request.get_data()
        if not image_bytes:
            return "no image bytes provided", 400

        try:
            response = requests.post(
                ML_CLIENT_PREDICT_URL,
                params={"source": "web-app", "instruction": draw_instruction},
                data=image_bytes,
                headers={"Content-Type": "application/octet-stream"},
                timeout=ML_CLIENT_TIMEOUT_SECONDS,
            )
        except requests.RequestException:
            return "unable to reach ml service", 502

        if response.status_code != 200:
            return f"ml service error ({response.status_code})", 502

        try:
            payload = response.json()
        except ValueError:
            return "invalid response from ml service", 502

        predictions = payload.get("predictions", [])
        if not predictions:
            return "unknown", 200

        top_prediction = predictions[0]
        label = top_prediction.get("label", "unknown")
        confidence = top_prediction.get("confidence")
        if isinstance(confidence, (int, float)):
            return f"{label} ({confidence:.1%})", 200
        return str(label), 200

    return "method not allowed", 405


@app.route("/history", methods=["GET"])
def history():
    """Retrieve and display the prediction history."""
    records = []
    try:
        response = requests.get(
            ML_CLIENT_HISTORY_URL,
            timeout=ML_CLIENT_TIMEOUT_SECONDS,
        )
        if response.status_code == 200:
            payload = response.json()
            records = payload.get("records", [])
    except requests.RequestException:
        pass
    return render_template("history.html", records=records)


@app.route("/history/<string:record_id>/delete", methods=["POST"])
def delete_history(record_id):
    """Delete a prediction record and redirect to history."""
    try:
        requests.delete(
            f"{ML_CLIENT_HISTORY_URL}/{record_id}",
            timeout=ML_CLIENT_TIMEOUT_SECONDS,
        )
    except requests.RequestException:
        pass

    # After deleting (or failing), redirect to the history page
    return redirect("/history")


if __name__ == "__main__":
    app.run(debug=True)
