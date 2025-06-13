import cv2
import numpy as np
from PIL import Image

class ImageClassifier:
    def __init__(self):
        # Color ranges for detection (HSV format)
        self.skin_lower = np.array([0, 48, 80], dtype=np.uint8)
        self.skin_upper = np.array([20, 255, 255], dtype=np.uint8)
        self.blood_lower = np.array([0, 50, 50], dtype=np.uint8)
        self.blood_upper = np.array([10, 255, 255], dtype=np.uint8)
        self.extreme_dark_lower = np.array([0, 0, 0], dtype=np.uint8)
        self.extreme_dark_upper = np.array([180, 255, 30], dtype=np.uint8)
        
        # Thresholds for classification
        self.skin_threshold = 20  # % of image
        self.blood_threshold = 5   # % of image
        self.dark_threshold = 30   # % of image
        
    def analyze_image(self, image_path):
        """Analyze image content and return classification"""
        try:
            # Read and convert image
            img = cv2.imread(image_path)
            if img is None:
                return "safe", {"error": "Could not read image"}
                
            hsv = cv2.cvtColor(img, cv2.COLOR_BGR2HSV)
            height, width = img.shape[:2]
            total_pixels = height * width
            
            # 1. Skin detection
            skin_mask = cv2.inRange(hsv, self.skin_lower, self.skin_upper)
            skin_pixels = np.sum(skin_mask > 0)
            skin_percent = (skin_pixels / total_pixels) * 100
            
            # 2. Blood-like color detection
            blood_mask = cv2.inRange(hsv, self.blood_lower, self.blood_upper)
            blood_pixels = np.sum(blood_mask > 0)
            blood_percent = (blood_pixels / total_pixels) * 100
            
            # 3. Extreme dark areas detection
            dark_mask = cv2.inRange(hsv, self.extreme_dark_lower, self.extreme_dark_upper)
            dark_pixels = np.sum(dark_mask > 0)
            dark_percent = (dark_pixels / total_pixels) * 100
            
            # 4. Edge detection (for violent content)
            gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
            edges = cv2.Canny(gray, 100, 200)
            edge_pixels = np.sum(edges > 0)
            edge_percent = (edge_pixels / total_pixels) * 100
            
            analysis = {
                'skin_percent': round(skin_percent, 2),
                'blood_percent': round(blood_percent, 2),
                'dark_percent': round(dark_percent, 2),
                'edge_percent': round(edge_percent, 2)
            }
            
            # Classification logic
            if skin_percent > self.skin_threshold and edge_percent > 5:
                return "potentially_harmful", analysis
            elif blood_percent > self.blood_threshold:
                return "potentially_harmful", analysis
            elif dark_percent > self.dark_threshold and edge_percent > 3:
                return "potentially_harmful", analysis
            else:
                return "safe", analysis
                
        except Exception as e:
            return "safe", {"error": str(e)}

from PIL import Image
from io import BytesIO
# adjust if needed

def moderate_image(image_io):
    image = Image.open(image_io)
    classifier = ImageClassifier()
    result, analysis = classifier.analyze_image(image)

    # print(f"Classification: {result}")
    # print("Analysis Details:")
    # for k, v in analysis.items():
    #     print(f"{k}: {v}")

    return result


# Example usage
if __name__ == "__main__":
    result = moderate_image("DSC_2312.jpg")