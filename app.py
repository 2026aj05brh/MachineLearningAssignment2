from pathlib import Path
import joblib
import matplotlib.pyplot as plt
import pandas as pd
import seaborn as sns
import streamlit as st

from sklearn.metrics import accuracy_score, roc_auc_score, precision_score, recall_score, f1_score, matthews_corrcoef, confusion_matrix, classification_report
st.set_page_config(page_title="Breast Cancer Classification Evaluation", page_icon = "🧬", layout="wide")

MODEL_DIRECTORY = Path("model")
TARGET_FILES = {
    "Logistic Regression":"logistic_regression.joblib",
    "Decision Tree":"decision_tree.joblib",
    "KNN": "knn.joblib",
    "Naive Bayes": "naive_bayes.joblib",
    "Support Vector Machine": "support_vector_machine.joblib"
}

@st.cache_resource
def load_feature_names():
    feature_path = MODEL_DIRECTORY / "feature_names.joblib"
    return joblib.load(feature_path)

def calculate_metrics(actual, predicted, probabibilities):
    return {
        "Accuracy": accuracy_score(actual, predicted),
        
  
