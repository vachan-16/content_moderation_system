from PIL import Image
import torch
from torchvision import transforms
from transformers import CLIPProcessor, CLIPModel
import numpy as np
import pickle
from sklearn.ensemble import RandomForestClassifier
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler
from skimage.feature import hog
from sklearn.model_selection import train_test_split

# Load model
model = CLIPModel.from_pretrained("openai/clip-vit-base-patch32")
processor = CLIPProcessor.from_pretrained("openai/clip-vit-base-patch32")

# Defining categories
labels = [
    "a normal safe image",
    "an image showing nudity",
    "an image showing physical violence or blood",
    "an image showing a hate symbol like a Hakenkreuz or racist flag"
]


def classification(filename='image_moderation.pkl'):
    # Generate synthetic grayscale "images"
    n_samples = 100
    img_size = (32, 32)
    X_images = np.random.rand(n_samples, *img_size)
    y_labels = np.random.randint(0, 3, n_samples)  

    # Extract HOG features for each image
    X_features = np.array([
        hog(img, orientations=8, pixels_per_cell=(4, 4), cells_per_block=(1, 1), visualize=False)
        for img in X_images
    ])

    # Split into train/test to mimic a real workflow (optional)
    X_train, X_test, y_train, y_test = train_test_split(X_features, y_labels, test_size=0.2, random_state=42)

    # Training a classifier (Random Forest for demonstration)
    clf = RandomForestClassifier(n_estimators=50, random_state=42)
    clf.fit(X_train, y_train)

    # Saving the trained model 
    with open(filename, 'wb') as f:
        pickle.dump(clf, f)



def classify_image(image_path):
    image = Image.open(image_path).convert("RGB")
    inputs = processor(
        text=labels,
        images=image,
        return_tensors="pt",
        padding=True
    )

    with torch.no_grad():
        outputs = model(**inputs)
        logits_per_image = outputs.logits_per_image
        probs = logits_per_image.softmax(dim=1).squeeze().tolist()

    top_idx = probs.index(max(probs))
    result = {
        "label": labels[top_idx],
        "confidence": round(probs[top_idx], 3)
    }
    return result

def moderate_image(image_path):
    result=classify_image(image_path)
    if result["confidence"]<0.60:
        return f"🔍 Prediction: {labels[0]}, \n📊 Confidence: {result['confidence']}"
    else:
        return f"🔍 Prediction: {result["label"]}, \n 📊 Confidence: {result['confidence']}"

