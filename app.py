from flask import Flask, render_template, request
from image_classifiction import moderate_image
from text_moderation_response import moderate_text
from video_classification import moderate_video
from io import BytesIO
import os

app = Flask(__name__)

@app.route("/", methods=["GET", "POST"])
def index():
    result = None
    if request.method == "POST":
        moderation_type = request.form["type"]

        if moderation_type == "text":
            text = request.form.get("text_input", "")
            result = moderate_text(text)

        elif moderation_type == "image":
            image_file = request.files["image_file"]
            if image_file:
                # Read image into memory
                image_bytes = image_file.read()
                result = moderate_image(BytesIO(image_bytes))

        elif moderation_type == "video":
            video_file = request.files["video_file"]
            if video_file:
                # Read video into memory
                video_bytes = video_file.read()
                result = moderate_video(BytesIO(video_bytes))

    return render_template("index.html", result=result)

if __name__ == "__main__":
    port = int(os.environ.get("PORT", 5000))
    app.run(debug=False, host="0.0.0.0", port=port)
