import cv2
import os
import numpy as np
from collections import Counter

class VideoClassifier:
    def __init__(self):
        # Define classification thresholds 
        self.safe_keywords = ['educational', 'tutorial', 'nature', 'science']
        self.harmful_keywords = ['violence', 'fight', 'weapon', 'danger']
        self.prohibited_keywords = ['nudity', 'porn', 'hate speech', 'illegal']
        
        # Visual detection thresholds (HSV color ranges)
        self.skin_lower = np.array([0, 48, 80], dtype=np.uint8)
        self.skin_upper = np.array([20, 255, 255], dtype=np.uint8)
        self.blood_lower = np.array([0, 50, 50], dtype=np.uint8)
        self.blood_upper = np.array([10, 255, 255], dtype=np.uint8)
        
    def classify_video(self, video_path, metadata=None):
        """
        Classify a video into safe, potentially_harmful, or prohibited categories
        
        Args:
            video_path: Path to the video file
            metadata: Optional dictionary containing video metadata (title, description, etc.)
            
        Returns:
            str: Classification result ('safe', 'potentially_harmful', or 'prohibited')
            dict: Analysis details
        """
        analysis = {
            'keyword_matches': [],
            'skin_pixels_percent': 0,
            'blood_pixels_percent': 0,
            'motion_intensity': 0,
            'brightness_changes': 0
        }
        
        # 1. Analyze metadata if available
        if metadata:
            text = f"{metadata.get('title', '')} {metadata.get('description', '')}".lower()
            
            # Check for prohibited keywords
            prohibited_matches = [kw for kw in self.prohibited_keywords if kw in text]
            if prohibited_matches:
                analysis['keyword_matches'] = prohibited_matches
                return 'prohibited', analysis
                
            # Check for harmful keywords
            harmful_matches = [kw for kw in self.harmful_keywords if kw in text]
            if harmful_matches:
                analysis['keyword_matches'] = harmful_matches
                return 'potentially_harmful', analysis
                
            # Check for safe keywords
            safe_matches = [kw for kw in self.safe_keywords if kw in text]
            if safe_matches:
                analysis['keyword_matches'] = safe_matches
                return 'safe', analysis
        
        # 2. Analyze video content
        try:
            cap = cv2.VideoCapture(video_path)
            total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
            sample_frames = min(100, total_frames)  # Analyze up to 100 frames
            
            skin_pixels = []
            blood_pixels = []
            prev_frame = None
            brightness_changes = []
            
            for i in range(sample_frames):
                ret, frame = cap.read()
                if not ret:
                    break
                    
                if i % (total_frames // sample_frames) == 0:
                    # Convert to HSV for color analysis
                    hsv = cv2.cvtColor(frame, cv2.COLOR_BGR2HSV)
                    
                    # Skin detection
                    skin_mask = cv2.inRange(hsv, self.skin_lower, self.skin_upper)
                    skin_pixels.append(np.sum(skin_mask > 0) / (frame.shape[0] * frame.shape[1]))
                    
                    # Blood-like color detection
                    blood_mask = cv2.inRange(hsv, self.blood_lower, self.blood_upper)
                    blood_pixels.append(np.sum(blood_mask > 0) / (frame.shape[0] * frame.shape[1]))
                    
                    # Motion detection
                    if prev_frame is not None:
                        diff = cv2.absdiff(cv2.cvtColor(prev_frame, cv2.COLOR_BGR2GRAY), 
                                         cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY))
                        analysis['motion_intensity'] += np.mean(diff)
                        
                        # Brightness changes
                        brightness = np.mean(frame)
                        brightness_changes.append(abs(brightness - np.mean(prev_frame)))
                    
                    prev_frame = frame
            
            # Calculate averages
            if skin_pixels:
                analysis['skin_pixels_percent'] = np.mean(skin_pixels) * 100
            if blood_pixels:
                analysis['blood_pixels_percent'] = np.mean(blood_pixels) * 100
            if brightness_changes:
                analysis['brightness_changes'] = np.mean(brightness_changes)
            if sample_frames > 1:
                analysis['motion_intensity'] /= sample_frames - 1
                
            cap.release()
            
            # Make classification based on content analysis
            if analysis['skin_pixels_percent'] > 20:  # High skin pixel percentage
                return 'potentially_harmful', analysis
            elif analysis['blood_pixels_percent'] > 5:  # Blood-like colors detected
                return 'potentially_harmful', analysis
            elif analysis['motion_intensity'] > 30:  # High motion intensity
                return 'potentially_harmful', analysis
            elif analysis['brightness_changes'] > 20:  # Rapid brightness changes
                return 'potentially_harmful', analysis
            else:
                return 'safe', analysis
                
        except Exception as e:
            print(f"Error analyzing video: {e}")
            return 'safe', analysis  # Default to safe if analysis fails


def moderate_video(uploaded_image):
    classifier = VideoClassifier()
    

    result, analysis = classifier.classify_video(uploaded_image)
    return f"Classification: {result}"
    # print(f"Analysis: {analysis}")