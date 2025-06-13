# 🛡️ Content Moderation System (Text, Image, Video)

A Flask-based web application for automatic moderation of **text**, **images**, and **videos**. The system classifies content into:
- ✅ Safe
- ⚠️ Potentially Harmful
- ❌ Prohibited

It uses pre-trained models and NLP/computer vision techniques to detect nudity, violence, hate symbols, toxic language, and more.

---

## 🔧 Tech Stack

- **Flask** – Web framework
- **Torch + Transformers** – CLIP model for image classification
- **OpenCV** – Video frame extraction
- **scikit-learn + NLTK** – Text moderation using TF-IDF and Random Forest
- **HTML + CSS** – Frontend UI (with a responsive layout)

---

## 🚀 Features

- ✅ Upload and classify text, image, or video content
- ✅ Returns category and confidence level
- ✅ Clean UI with Bootstrap/CSS
- ✅ Image & text pre-trained model support

---

## 📂 Project Structure

content_moderation_system/
├── app.py
├── classify_image.py
├── text_moderation_response.py
├── video_classification.py
├── txt_training.py                    # Uses final_dataset.csv
├── final_dataset.csv                  # CSV used for model training
├── models/
│   ├── traditional_moderation_model.pkl
│   ├── image_model.pkl
│   └── video_model.pkl
├── static/
│   └── style.css
├── templates/
│   └── index.html
├── requirements.txt
├── Procfile                          
├── README.md
├── .gitignore                       
└── venv/                    

--

## 🔍 How It Works

This project automatically moderates user-generated text, image, and video content using trained machine learning and deep learning models. Here's how each type of content is handled:

📝 Text Moderation
- ✅ Preprocessing: Cleans and tokenizes user input using NLTK (stopword removal, lemmatization).
- ✅ Feature Extraction: Converts text into TF-IDF vectors.
- ✅ Classification: Predicts label using a trained RandomForestClassifier.
- ✅ Output: Displays moderation label such as Safe, Hate Speech, or Abusive.

🖼️ Image Moderation
- ✅ Model: Uses OpenAI’s pre-trained CLIP model.
- ✅ Process: Compares uploaded image with moderation categories (e.g., nudity, violence, hate symbols).
- ✅ Confidence Check: Applies a threshold (e.g., 60%) to determine label.
- ✅ Output: Displays the most relevant label and confidence score.

🎞️ Video Moderation
- ✅ Frame Extraction: Uses OpenCV to extract frames from the uploaded video.
- ✅ Per-Frame Classification: Each frame is analyzed using the image moderation pipeline.
- ✅ Aggregation: Final result is based on the most frequent frame label.
- ✅ Output: Returns the dominant classification for the video.


