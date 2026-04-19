from io import BytesIO
from flask import Flask, flash, redirect, render_template, request, url_for, session
import asyncio, os
import random

app = Flask(__name__)

#get the secret key from env
app.config["SECRET_KEY"] = os.getenv("SECRET_KEY")

things = [
    "circle",
    "square",
    "triangle",
    "rectangle",
    "oval",
    "star",
    "heart",
    "diamond",
    "line",
    "zigzag",
    "spiral",
    "cube",
    "sphere",
    "pyramid",
    "cone",
    "cylinder",
    "cross",
    "arrow",
    "wave",
    "loop",
    "person",
    "face",
    "eye",
    "nose",
    "mouth",
    "ear",
    "hand",
    "foot",
    "finger",
    "hair",
    "hat",
    "shirt",
    "pants",
    "shoe",
    "dress",
    "flower",
    "tree",
    "leaf",
    "grass",
    "bush",
    "sun",
    "moon",
    "cloud",
    "rain",
    "snow",
    "rainbow",
    "starfish",
    "fish",
    "bird",
    "cat",
    "dog",
    "horse",
    "cow",
    "pig",
    "sheep",
    "duck",
    "frog",
    "bug",
    "butterfly",
    "spider",
    "bee",
    "ant",
    "house",
    "door",
    "window",
    "roof",
    "fence",
    "road",
    "bridge",
    "car",
    "truck",
    "bus",
    "train",
    "airplane",
    "boat",
    "rocket",
    "bike",
    "wheel",
    "cup",
    "plate",
    "spoon",
    "fork",
    "knife",
    "bottle",
    "apple",
    "banana",
    "cake",
    "cookie",
    "icecream",
    "candy",
    "ball",
    "kite",
    "balloon",
    "gift",
    "book",
    "pencil",
    "crayon",
    "paint",
    "brush",
    "clock",
]

def getRandomThing():
    return random.choice(things)

@app.route("/", methods=["GET", "POST"])
def index():
    if request.method == "GET":
        thing = getRandomThing()
        return render_template("index.html", thing=thing)
    if request.method == "POST":
        image_bytes = request.get_data()
        # TODO: do ai image recognition
        image_description = "[temp description of doodle]"
        return image_description, 200

@app.route("/history", methods=["GET"])
def index():
    if request.method == "GET":
        # TODO
        return render_template("history.html")

if __name__ == "__main__":
    app.run(debug=True)
