import streamlit as st
import pandas as pd
import numpy as np
import joblib
from sklearn.feature_extraction.text import TfidfVectorizer
import plotly.graph_objects as go
import plotly.express as px

class ToxicityDetectionApp:
    def __init__(self):
        self.models = {}
        self.vectorizer = None
        self.labels = ['toxic', 'severe_toxic', 'obscene', 'threat', 'insult', 'identity_hate']
        
    def load_models(self):
        """Load pre-trained models and vectorizer"""
        try:
            self.vectorizer = joblib.load('models/tfidf_vectorizer.pkl')
            self.models['logistic_regression'] = joblib.load('models/logistic_regression_model.pkl')
            self.models['random_forest'] = joblib.load('models/random_forest_model.pkl')
            return True
        except:
            return False
    
    def predict_toxicity(self, text, model_name='logistic_regression'):
        """Predict toxicity for given text"""
        if model_name not in self.models:
            return None
            
        # Preprocess and vectorize text
        cleaned_text = self.preprocess_text(text)
        features = self.vectorizer.transform([cleaned_text])
        
        # Make prediction
        predictions = self.models[model_name].predict_proba(features)
        
        results = {}
        for i, label in enumerate(self.labels):
            results[label] = predictions[i][0][1]  # Probability of toxic class
        
        return results
    
    def preprocess_text(self, text):
        """Basic text preprocessing"""
        import re
        text = text.lower()
        text = re.sub(r'[^a-zA-Z\s]', '', text)
        return text
    
    def run(self):
        """Run the Streamlit application"""
        st.set_page_config(page_title="Toxicity Detection", page_icon="🚫", layout="wide")
        
        st.title("🚫 Toxicity Detection in Comments")
        st.markdown("Detect toxic content in text comments using Machine Learning")
        
        # Load models
        if not self.load_models():
            st.error("Models not found. Please train the models first.")
            return
        
        # Sidebar
        st.sidebar.header("Configuration")
        model_choice = st.sidebar.selectbox(
            "Select Model",
            list(self.models.keys())
        )
        
        # Main content
        col1, col2 = st.columns([2, 1])
        
        with col1:
            st.subheader("Enter Text to Analyze")
            user_input = st.text_area("Type your comment here:", height=150)
            
            if st.button("Analyze Toxicity"):
                if user_input.strip():
                    with st.spinner("Analyzing..."):
                        results = self.predict_toxicity(user_input, model_choice)
                        
                        if results:
                            # Display results
                            st.subheader("Toxicity Analysis Results")
                            
                            # Create gauge chart
                            fig = go.Figure()
                            
                            for i, (label, score) in enumerate(results.items()):
                                fig.add_trace(go.Indicator(
                                    mode = "gauge+number",
                                    value = score * 100,
                                    title = {'text': label.upper()},
                                    domain = {'row': i // 3, 'column': i % 3},
                                    gauge = {
                                        'axis': {'range': [0, 100]},
                                        'bar': {'color': "darkblue"},
                                        'steps': [
                                            {'range': [0, 30], 'color': "lightgreen"},
                                            {'range': [30, 70], 'color': "yellow"},
                                            {'range': [70, 100], 'color': "red"}
                                        ]
                                    }
                                ))
                            
                            fig.update_layout(
                                grid = {'rows': 2, 'columns': 3, 'pattern': "independent"},
                                height=500
                            )
                            
                            st.plotly_chart(fig, use_container_width=True)
                            
                            # Overall toxicity score
                            max_toxicity = max(results.values())
                            st.metric("Overall Toxicity Score", f"{max_toxicity*100:.2f}%")
                            
                else:
                    st.warning("Please enter some text to analyze.")
        
        with col2:
            st.subheader("Model Information")
            st.info(f"Current Model: {model_choice.upper()}")
            
            st.subheader("Toxicity Labels")
            for label in self.labels:
                st.write(f"• {label.replace('_', ' ').title()}")
            
            st.subheader("About")
            st.markdown("""
            This system detects various types of toxic content:
            - **Toxic**: Generally rude, disrespectful comments
            - **Severe Toxic**: Very hateful, aggressive comments  
            - **Obscene**: Offensive, vulgar language
            - **Threat**: Intent to inflict physical harm
            - **Insult**: Insulting, inflammatory comments
            - **Identity Hate**: Hate speech against protected groups
            """)

if __name__ == "__main__":
    app = ToxicityDetectionApp()
    app.run()
