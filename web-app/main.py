from io import BytesIO
from flask import Flask, flash, redirect, render_template, request, url_for, session
import asyncio, os

app = Flask(__name__)

#get the secret key from env
app.config["SECRET_KEY"] = os.getenv("SECRET_KEY")

@app.route("/", methods=["GET", "POST"])
def index():
    if request.method == "GET":
        return render_template("index.html")
    if request.method == "POST":
        image_bytes = request.get_data()
        # TODO: do ai image recognition
        image_description = "a cool doodle"
        return image_description, 200

if __name__ == "__main__":
    app.run(debug=True)
