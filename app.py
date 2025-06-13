from flask import Flask, render_template, request
from classify_image import moderate_image
from text_moderation_response import moderate_text
from video_classification import moderate_video
import os

app = Flask(__name__)
UPLOAD_FOLDER = "uploads"
os.makedirs(UPLOAD_FOLDER, exist_ok=True)
app.config['UPLOAD_FOLDER'] = UPLOAD_FOLDER

@app.route("/", methods=["GET", "POST"])
def index():
    result = None
    if request.method == "POST":
        moderation_type = request.form["type"]

        if moderation_type == "text":
            text = request.form.get("text_input", "")
            result = moderate_text(text)

        elif moderation_type == "image":
            image = request.files["image_file"]
            image_path = os.path.join(app.config["UPLOAD_FOLDER"], image.filename)
            image.save(image_path)
            result = moderate_image(image_path)

        elif moderation_type == "video":
            video = request.files["video_file"]
            video_path = os.path.join(app.config["UPLOAD_FOLDER"], video.filename)
            video.save(video_path)
            result = moderate_video(video_path)

    return render_template("index.html", result=result)

if __name__ == "__main__":
    app.run(debug=True)
