import pandas as pd
import numpy as np
import re
import nltk
from nltk.corpus import stopwords
from nltk.stem import WordNetLemmatizer
from sklearn.feature_extraction.text import TfidfVectorizer, CountVectorizer
from sklearn.model_selection import train_test_split
import joblib

class TextPreprocessor:
    def __init__(self):
        nltk.download('stopwords')
        nltk.download('wordnet')
        self.stop_words = set(stopwords.words('english'))
        self.lemmatizer = WordNetLemmatizer()
        self.vectorizer = None
        
    def clean_text(self, text):
        """Clean and preprocess text data"""
        if not isinstance(text, str):
            return ""
        
        # Convert to lowercase
        text = text.lower()
        
        # Remove special characters and digits
        text = re.sub(r'[^a-zA-Z\s]', '', text)
        
        # Remove extra whitespaces
        text = ' '.join(text.split())
        
        # Tokenize and remove stopwords
        tokens = text.split()
        tokens = [word for word in tokens if word not in self.stop_words]
        
        # Lemmatization
        tokens = [self.lemmatizer.lemmatize(word) for word in tokens]
        
        return ' '.join(tokens)
    
    def load_and_preprocess_data(self, file_path):
        """Load dataset and preprocess text"""
        df = pd.read_csv(file_path)
        
        # Clean comments
        df['cleaned_comment'] = df['comment_text'].apply(self.clean_text)
        
        # Extract labels
        labels = ['toxic', 'severe_toxic', 'obscene', 'threat', 'insult', 'identity_hate']
        
        return df, labels
    
    def create_features(self, texts, method='tfidf', max_features=5000):
        """Create feature vectors from text"""
        if method == 'tfidf':
            self.vectorizer = TfidfVectorizer(max_features=max_features, stop_words='english')
        else:
            self.vectorizer = CountVectorizer(max_features=max_features, stop_words='english')
            
        features = self.vectorizer.fit_transform(texts)
        return features
    
    def save_vectorizer(self, file_path):
        """Save fitted vectorizer"""
        if self.vectorizer:
            joblib.dump(self.vectorizer, file_path)
    
    def load_vectorizer(self, file_path):
        """Load fitted vectorizer"""
        self.vectorizer = joblib.load(file_path)
