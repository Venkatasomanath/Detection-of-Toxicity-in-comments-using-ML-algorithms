import pandas as pd
import numpy as np
from sklearn.ensemble import RandomForestClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.svm import SVC
from sklearn.naive_bayes import MultinomialNB
from sklearn.tree import DecisionTreeClassifier
from sklearn.neighbors import KNeighborsClassifier
from sklearn.multioutput import MultiOutputClassifier
from sklearn.metrics import accuracy_score, classification_report, hamming_loss
from sklearn.model_selection import cross_val_score
import joblib

class ToxicityClassifier:
    def __init__(self):
        self.models = {
            'random_forest': RandomForestClassifier(n_estimators=100, random_state=42),
            'logistic_regression': LogisticRegression(random_state=42),
            'svm': SVC(probability=True, random_state=42),
            'naive_bayes': MultinomialNB(),
            'decision_tree': DecisionTreeClassifier(random_state=42),
            'knn': KNeighborsClassifier(n_neighbors=3)
        }
        self.trained_models = {}
        self.best_model = None
        
    def train_models(self, X_train, y_train, X_test, y_test):
        """Train all models and evaluate performance"""
        results = {}
        
        for name, model in self.models.items():
            print(f"Training {name}...")
            
            # Use MultiOutputClassifier for multi-label classification
            multi_model = MultiOutputClassifier(model)
            multi_model.fit(X_train, y_train)
            
            # Make predictions
            y_pred = multi_model.predict(X_test)
            
            # Calculate metrics
            accuracy = accuracy_score(y_test, y_pred)
            h_loss = hamming_loss(y_test, y_pred)
            
            results[name] = {
                'model': multi_model,
                'accuracy': accuracy,
                'hamming_loss': h_loss,
                'predictions': y_pred
            }
            
            self.trained_models[name] = multi_model
            
            print(f"{name} - Accuracy: {accuracy:.4f}, Hamming Loss: {h_loss:.4f}")
        
        return results
    
    def find_best_model(self, results):
        """Find the best performing model"""
        best_accuracy = 0
        best_model_name = None
        
        for name, result in results.items():
            if result['accuracy'] > best_accuracy:
                best_accuracy = result['accuracy']
                best_model_name = name
        
        self.best_model = self.trained_models[best_model_name]
        print(f"Best model: {best_model_name} with accuracy: {best_accuracy:.4f}")
        
        return best_model_name
    
    def save_model(self, model_name, file_path):
        """Save trained model"""
        if model_name in self.trained_models:
            joblib.dump(self.trained_models[model_name], file_path)
            print(f"Model saved to {file_path}")
    
    def load_model(self, model_name, file_path):
        """Load trained model"""
        self.trained_models[model_name] = joblib.load(file_path)
