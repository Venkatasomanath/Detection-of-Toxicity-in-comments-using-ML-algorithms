# Detection-of-Toxicity-in-comments-using-ML-algorithms
## Project Overview
Advanced machine learning system for detecting toxic content in online comments using multiple classification algorithms. Achieved 89.46% accuracy in identifying various types of toxic content including hate speech, threats, and obscene language.

## Features
- Multi-label classification (6 toxicity types)
- Multiple ML algorithms comparison
- Real-time toxicity prediction
- Interactive web interface
- Comprehensive model evaluation

## Models Implemented
- Logistic Regression
- Random Forest
- Support Vector Machines
- Naive Bayes
- Decision Trees
- K-Nearest Neighbors

## Installation

```bash
git clone https://github.com/yourusername/toxicity-detection-ml.git
cd toxicity-detection-ml
pip install -r requirements.txt
Usage
Training Models
python
python src/train_models.py
Web Application
bash
streamlit run src/app.py
Jupyter Notebook
bash
jupyter notebook notebooks/toxicity_analysis.ipynb
Dataset
The project uses the Jigsaw Toxic Comments dataset from Kaggle containing Wikipedia comments with toxicity labels.
