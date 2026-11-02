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

def calculate_metrics(actual, predicted_classes, predicted_probabibilities):
    return {
        "Accuracy": accuracy_score(actual, predicted_classes),
        "AUC": roc_auc_score(y_test, predicted_probabilities),
        "Precision": precision_score(y_test, predicted_classes, zero_division=0),
        "Recall": recall_score(y_test, predicted_classes, zero_division=0),
        "F1": f1_score(y_test, predicted_classes, zero_division=0),
        "MCC": matthews_corrcoef(y_test, predicted_classes)
    }

st.title("🧬 Breast Cancer Classification Model Evaluator")

st.write(
    """Uploda the supplied test CSV and select a previously trained model.
    The application evaluates the saved model without retraining it.
    """
)

st.info("Required target column: `target`."
        "Class 0 represents malignant and class 1 represents benign."
       )

selected_model_name = st.selectbox("Select a classification model", options=list(MODEL_FILES.keys()))
uploaded_file = st.file_uploader("Upload test_data.csv", type=["csv"])
if uploaded_file is None:
    st.warning("Upload the generated test_csv file to start evaluation.")
else:
    try:
        
