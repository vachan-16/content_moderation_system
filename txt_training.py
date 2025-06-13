import pandas as pd
from sklearn.model_selection import train_test_split
import nltk
from nltk.corpus import stopwords
from nltk.tokenize import word_tokenize
from nltk.stem import WordNetLemmatizer
import re
import string
import numpy as np
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import classification_report, confusion_matrix
import joblib
import warnings
warnings.filterwarnings('ignore')

# Download NLTK data
nltk.download('punkt', quiet=True)
nltk.download('stopwords', quiet=True)
nltk.download('wordnet', quiet=True)
nltk.download('punkt_tab', quiet=True)

# Load dataset
df = pd.read_csv('final_dataset.csv')

# Data validation
print(f"Dataset shape: {df.shape}")
print(f"Columns: {df.columns.tolist()}")
print(f"\nLabel distribution:")
print(df['label'].value_counts())

# Handle missing values
df = df.dropna(subset=['comment_text', 'label'])
print(f"After removing NaN: {df.shape}")

# Convert labels to numerical values
label_mapping = {'safe': 0, 'potentially_harmful': 1, 'prohibited': 2}
df['label_encoded'] = df['label'].map(label_mapping)

# Check for unmapped labels
if df['label_encoded'].isna().any():
    print("Warning: Some labels couldn't be mapped!")
    print(df[df['label_encoded'].isna()]['label'].unique())
    df = df.dropna(subset=['label_encoded'])

# Stratified split to maintain class balance
print("\nSplitting data...")
train_df, test_df = train_test_split(
    df, test_size=0.2, random_state=42, stratify=df['label_encoded']
)
train_df, val_df = train_test_split(
    train_df, test_size=0.25, random_state=42, stratify=train_df['label_encoded']
)

print(f"Train: {len(train_df)}, Val: {len(val_df)}, Test: {len(test_df)}")

def preprocess_text(text):
    """Enhanced text preprocessing function"""
    # Handle missing values
    if pd.isna(text) or text == '':
        return ''
    
    # Ensure text is string
    text = str(text)
    
    # Lowercase
    text = text.lower()
    
    # Remove URLs
    text = re.sub(r'http\S+|www\S+|https\S+', '', text, flags=re.MULTILINE)
    
    # Remove email addresses
    text = re.sub(r'\S+@\S+', '', text)
    
    # Remove extra whitespace
    text = re.sub(r'\s+', ' ', text)
    
    # Remove punctuation
    text = text.translate(str.maketrans('', '', string.punctuation))
    
    # Remove digits 
    text = re.sub(r'\d+', '', text)
    
    # Tokenization
    try:
        tokens = word_tokenize(text)
    except:
        # Fallback if tokenization fails
        tokens = text.split()
    
    # Remove stopwords and filter out empty tokens
    stop_words = set(stopwords.words('english'))
    tokens = [word for word in tokens if word and word not in stop_words and len(word) > 2]
    
    # Lemmatization
    lemmatizer = WordNetLemmatizer()
    tokens = [lemmatizer.lemmatize(word) for word in tokens]
    
    return ' '.join(tokens)

# Apply preprocessing
print("\nPreprocessing text...")
train_df = train_df.copy()
val_df = val_df.copy()
test_df = test_df.copy()

train_df['processed_text'] = train_df['comment_text'].apply(preprocess_text)
val_df['processed_text'] = val_df['comment_text'].apply(preprocess_text)
test_df['processed_text'] = test_df['comment_text'].apply(preprocess_text)

# Remove empty processed texts
train_df = train_df[train_df['processed_text'].str.len() > 0]
val_df = val_df[val_df['processed_text'].str.len() > 0]
test_df = test_df[test_df['processed_text'].str.len() > 0]

print(f"After preprocessing - Train: {len(train_df)}, Val: {len(val_df)}, Test: {len(test_df)}")

# Traditional ML Pipeline
print("\n" + "="*50)
print("TRADITIONAL ML PIPELINE (TF-IDF + Random Forest)")
print("="*50)

# Vectorization
print("Vectorizing text...")
tfidf = TfidfVectorizer(
    max_features=5000,
    ngram_range=(1, 2),
    min_df=2,
    max_df=0.95
)

X_train_tfidf = tfidf.fit_transform(train_df['processed_text'])
X_val_tfidf = tfidf.transform(val_df['processed_text'])
X_test_tfidf = tfidf.transform(test_df['processed_text'])

# Model training
print("Training Random Forest...")
rf = RandomForestClassifier(
    n_estimators=100,
    class_weight='balanced',
    random_state=42,
    n_jobs=-1
)
rf.fit(X_train_tfidf, train_df['label_encoded'])

# Evaluation
print("\n--- Validation Results ---")
val_preds = rf.predict(X_val_tfidf)
print(classification_report(val_df['label_encoded'], val_preds, 
                          target_names=['safe', 'potentially_harmful', 'prohibited']))

print("\n--- Test Results ---")
test_preds = rf.predict(X_test_tfidf)
print(classification_report(test_df['label_encoded'], test_preds,
                          target_names=['safe', 'potentially_harmful', 'prohibited']))

# Save the traditional ML model
print("\nSaving traditional ML model...")
joblib.dump({'model': rf, 'vectorizer': tfidf, 'label_mapping': label_mapping}, 
           'traditional_moderation_model.pkl')

# BERT Pipeline
print("\n" + "="*50)
print("BERT PIPELINE (Optional)")
print("="*50)

try:
    from transformers import BertTokenizer, BertForSequenceClassification
    from transformers import Trainer, TrainingArguments
    import torch
    from torch.utils.data import Dataset
    
    # Check if CUDA is available
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"Using device: {device}")
    
    class TextDataset(Dataset):
        def __init__(self, texts, labels, tokenizer, max_length=128):
            self.encodings = tokenizer(
                texts, 
                truncation=True, 
                padding=True, 
                max_length=max_length,
                return_tensors='pt'
            )
            self.labels = labels
        
        def __getitem__(self, idx):
            item = {key: val[idx] for key, val in self.encodings.items()}
            item['labels'] = torch.tensor(self.labels[idx], dtype=torch.long)
            return item
        
        def __len__(self):
            return len(self.labels)
    
    # Initialize tokenizer and model
    print("Loading BERT model...")
    tokenizer = BertTokenizer.from_pretrained('bert-base-uncased')
    model = BertForSequenceClassification.from_pretrained(
        'bert-base-uncased', 
        num_labels=3
    ).to(device)
    
    print("Preparing BERT datasets...")
    train_dataset = TextDataset(
        train_df['comment_text'].tolist(), 
        train_df['label_encoded'].tolist(), 
        tokenizer, 
        128
    )
    val_dataset = TextDataset(
        val_df['comment_text'].tolist(), 
        val_df['label_encoded'].tolist(), 
        tokenizer, 
        128
    )
    
    # Training arguments
    training_args = TrainingArguments(
        output_dir='./bert_results',
        num_train_epochs=3,
        per_device_train_batch_size=8,  # Reduced batch size for memory
        per_device_eval_batch_size=16,
        warmup_steps=500,
        weight_decay=0.01,
        logging_dir='./bert_logs',
        logging_steps=100,
        evaluation_strategy='epoch',
        save_strategy='epoch',
        load_best_model_at_end=True,
        metric_for_best_model='eval_loss',
        greater_is_better=False
    )
    
    # Trainer
    trainer = Trainer(
        model=model,
        args=training_args,
        train_dataset=train_dataset,
        eval_dataset=val_dataset
    )
    
    # Train
    print("Training BERT model...")
    trainer.train()
    
    # Save BERT model
    print("Saving BERT model...")
    model.save_pretrained('./bert_moderation_model')
    tokenizer.save_pretrained('./bert_moderation_model')
    
except ImportError:
    print("Transformers library not installed. Skipping BERT training.")
    print("Install with: pip install transformers torch")
except Exception as e:
    print(f"Error in BERT training: {str(e)}")

print("\n" + "="*50)
print("PIPELINE COMPLETE!")
print("="*50)
print("Traditional ML model saved as 'traditional_moderation_model.pkl'")
print("BERT model saved in './bert_moderation_model/' directory")
