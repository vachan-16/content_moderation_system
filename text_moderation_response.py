import joblib
import nltk
import string
import os
import re
from nltk.corpus import stopwords
from nltk.tokenize import word_tokenize
from nltk.stem import WordNetLemmatizer
import pandas as pd
def setup_nltk():
    """Download NLTK data if not already present"""
    try:
        nltk.data.find('tokenizers/punkt')
    except LookupError:
        nltk.download('punkt', quiet=True)
        nltk.download('stopwords', quiet=True)
        nltk.download('wordnet', quiet=True)
        nltk.download('punkt_tab', quiet=True)

def preprocess_text(text):
    """Enhanced text preprocessing function"""
    if pd.isna(text) or text == '':
        return ''
    
    text = str(text).lower()
    text = re.sub(r'http\S+|www\S+|https\S+', '', text, flags=re.MULTILINE)
    text = re.sub(r'\S+@\S+', '', text)
    text = re.sub(r'\s+', ' ', text)
    text = text.translate(str.maketrans('', '', string.punctuation))
    text = re.sub(r'\d+', '', text)
    
    try:
        tokens = word_tokenize(text)
    except:
        tokens = text.split()
    
    stop_words = set(stopwords.words('english'))
    tokens = [word for word in tokens if word and word not in stop_words and len(word) > 2]
    
    lemmatizer = WordNetLemmatizer()
    tokens = [lemmatizer.lemmatize(word) for word in tokens]
    
    return ' '.join(tokens)

class TextModerationModel:
    """Class for loading and using the trained model"""
    
    def __init__(self, model_path='traditional_moderation_model.pkl'):
        """Load the trained model"""
        try:
            base_dir = os.path.dirname(__file__)
            full_model_path = os.path.join(base_dir, 'models', model_path)
            self.model_data = joblib.load(full_model_path)
            self.model = self.model_data['model']
            self.vectorizer = self.model_data['vectorizer']
            self.label_mapping = self.model_data['label_mapping']
            self.reverse_mapping = {v: k for k, v in self.label_mapping.items()}
            
            # Setup NLTK for preprocessing
            setup_nltk()
            
        except FileNotFoundError:
            print("❌ Model file not found! Please train the model first.")
            raise
        except Exception as e:
            print(f"❌ Error loading model: {e}")
            raise
    
    def predict_single(self, text):
        """Predict moderation label for a single text"""
        # Preprocess
        processed = preprocess_text(text)
        
        if not processed:
            return {
                'prediction': 'safe',
                'confidence': 0.5,
                'probabilities': {'safe': 0.5, 'potentially_harmful': 0.25, 'prohibited': 0.25},
                'warning': 'Empty text after preprocessing'
            }
        
        # Vectorize
        text_vec = self.vectorizer.transform([processed])
        
        # Predict
        prediction = self.model.predict(text_vec)[0]
        probabilities = self.model.predict_proba(text_vec)[0]
        
        return {
            'prediction': self.reverse_mapping[prediction],
            'confidence': max(probabilities),
            'probabilities': {self.reverse_mapping[i]: float(prob) for i, prob in enumerate(probabilities)},
            'processed_text': processed
        }
    
    def predict_batch(self, texts):
        """Predict moderation labels for multiple texts"""
        results = []
        for text in texts:
            results.append(self.predict_single(text))
        return results
    
    def is_safe(self, text, threshold=0.7):
        """Quick check if text is safe"""
        result = self.predict_single(text)
        return result['prediction'] == 'safe' and result['confidence'] >= threshold



def moderate_text(user_input):
    """Main function demonstrating usage"""
    

    # Load model
    moderator = TextModerationModel()
    
    # Test examples
    # test_texts = [
    #     "This is a great product, I love it!",
    #     "I hate this stupid thing, it's garbage",
    #     "You should kill yourself, nobody likes you",
    #     "The weather is nice today",
    #     "This person is annoying but whatever"
    # ]
    

    # for text in test_texts:
    #     result = moderator.predict_single(text)
    #     print(f"Text: {text}")
    #     print(f"Prediction: {result['prediction']} (confidence: {result['confidence']:.2f})")
    #     print(f"Safe check: {moderator.is_safe(text)}")
    #     print("-" * 40)
    
    # Batch prediction
    # print("\n--- Batch Predictions ---")
    # batch_results = moderator.predict_batch(test_texts)
    # for i, result in enumerate(batch_results):
    #     print(f"{i+1}. {result['prediction']} ({result['confidence']:.2f})")
    
    # # Interactive mode
    # print("\n--- Interactive Mode ---")
    # print("Enter text to moderate (or 'quit' to exit):")
    

        
    if user_input:
        result = moderator.predict_single(user_input)
        return f"🔍 Prediction: {result['prediction']}\n 📊 Confidence: {result['confidence']:.2f}"






# --------------------------------------------------------------------------------------------------------------------

# ================================================================
# PART 4: DEPLOYMENT FUNCTIONS (for web apps, APIs, etc.)
# ================================================================

# Global model instance (load once when app starts)
# _model_instance = None

# def get_model():
#     """Get singleton model instance"""
#     global _model_instance
#     if _model_instance is None:
#         _model_instance = TextModerationModel()
#     return _model_instance

# def moderate_text_api(text):
#     """API-ready function for text moderation"""
#     try:
#         model = get_model()
#         result = model.predict_single(text)
        
#         return {
#             'status': 'success',
#             'text': text,
#             'moderation': result['prediction'],
#             'confidence': result['confidence'],
#             'is_safe': model.is_safe(text),
#             'probabilities': result['probabilities']
#         }
#     except Exception as e:
#         return {
#             'status': 'error',
#             'error': str(e),
#             'text': text
#         }

# ================================================================
# QUICK START GUIDE
# ================================================================

"""
QUICK START GUIDE:

1. FIRST TIME SETUP:
   - Run: python train_model.py
   - This creates 'moderation_model.pkl'

2. FOR PREDICTIONS:
   - Use TextModerationModel class
   - Or call moderate_text_api() function

3. EXAMPLES:
   
   # Simple prediction
   moderator = TextModerationModel()
   result = moderator.predict_single("Your text here")
   
   # Quick safety check
   is_safe = moderator.is_safe("Your text here")
   
   # API-style usage
   api_result = moderate_text_api("Your text here")

4. RETRAINING:
   - Only run train_model() when you have new data
   - Or want to change model parameters

5. DEPLOYMENT:
   - Use get_model() to load once
   - Use moderate_text_api() for web APIs
"""