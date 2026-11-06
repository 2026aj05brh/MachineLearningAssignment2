from pathlib import Path
import joblib
import matplotlib.pyplot as plt
import pandas as pd
import seaborn as sns
from seaborn import matrix
import streamlit as st

from sklearn.metrics import accuracy_score, roc_auc_score, precision_score, recall_score, f1_score, matthews_corrcoef, confusion_matrix, classification_report
st.set_page_config(page_title="Breast Cancer Classification Evaluation", page_icon = "🧬", layout="wide")

MODEL_DIRECTORY = Path("model")
TARGET_COLUMN = "target"
MODEL_FILES = {
    "Logistic Regression":"logistic_regression.joblib",
    "Decision Tree":"decision_tree.joblib",
    "KNN": "knn.joblib",
    "Naive Bayes": "naive_bayes.joblib",
    "Support Vector Machine": "support_vector_machine.joblib"
}

@st.cache_resource
def load_saved_model(model_file_name):
    model_path = MODEL_DIRECTORY / model_file_name
    return joblib.load(model_path)

@st.cache_resource
def load_feature_names():
    feature_path = MODEL_DIRECTORY / "feature_names.joblib"
    return joblib.load(feature_path)

def calculate_metrics(actual, predicted_classes, predicted_probabibilities):
    return {
        "Accuracy": accuracy_score(actual, predicted_classes),
        "AUC": roc_auc_score(actual, predicted_probabibilities),
        "Precision": precision_score(actual, predicted_classes, zero_division=0),
        "Recall": recall_score(actual, predicted_classes, zero_division=0),
        "F1": f1_score(actual, predicted_classes, zero_division=0),
        "MCC": matthews_corrcoef(actual, predicted_classes)
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
        uploaded_data = pd.read_csv(uploaded_file)
        required_features = load_feature_names()

        st.subheader("Uploaded-data-preview")
        st.dataframe(uploaded_data.head(10), use_container_width=True)

        missing_features = [feature for feature in required_features if feature not in uploaded_data.columns]
        if TARGET_COLUMN not in uploaded_data.columns:
            st.error(f"The uploaded csv must contain the "
                     f"`{TARGET_COLUMN}` column.")
        elif missing_features:
            st.error("The uploaded csv is missing these features: "
                     +", ".join(missing_features))
        else:
            X_test = uploaded_data[required_features]
            y_test = uploaded_data[TARGET_COLUMN]

            selected_model = load_saved_model(MODEL_FILES[selected_model_name])
            predictions = selected_model.predict(X_test)
            probabilities = selected_model.predict_proba(X_test)[:, 1]
            metrics = calculate_metrics(y_test, predictions, probabilities)
            
            st.success(f"{selected_model_name} evaluated successfully. The model was not retrained.")
            st.subheader("Evaluation metrics")
            metric_columns = st.columns(6)

            for column, (metric_name, metric_value) in zip(metric_columns, metrics.items()):
                column.metric(metric_name, f"{metric_value:.4f}")
            left_column, right_column = st.columns(2)
            with left_column:
                st.subheader("Confusion matrix")

                matrix = confusion_matrix(y_test, predictions)

                figure, axis = plt.subplots(figsize=(5,4))

                sns.heatmap(matrix, annot = True, fmt="d", cmap="Blues", cbar=False, 
                            xticklabels=["Malignant", "Benign"], yticklabels=["Malignant", "Benign"], ax=axis)
                axis.set_xlabel("Predicted Class")
                axis.set_ylabel("Actual Class")
                axis.set_title(selected_model_name)

                st.pyplot(figure)

            with right_column:
                st.subheader("Classification report")
                report = classification_report(y_test, 
                                               predictions, 
                                               target_names=["Malignant", "Benign"], 
                                               output_dict = True, 
                                               zero_division = 0
                                               )
                accuracy = report.pop("accuracy")
                report_table = pd.DataFrame(report).transpose()
                st.metric("Overall Accuracy", f"{accuracy:.4f}")
                st.dataframe(
                    report_table.style.format({
                        "precision": "{:.4f}",
                        "recall": "{:.4f}",
                        "f1-score": "{:.4f}",
                        "support": "{:,.0f}"
                    }), 
                    use_container_width=True
                    )

            result_table = uploaded_data.copy()
            result_table["predicted_target"] = predictions
            result_table["probability_benign"] = probabilities

            st.download_button(label = "Download predictions results", data=result_table.to_csv(index=False), file_name="prediction_results.csv", mime="text/csv")
    # except ValueError as error:
    #     st.error(f"Evaluation error: {error}")

    except Exception as error:
        st.error(
            "The file could not be processed. " 
            f"Technical details: {error}")

st.markdown(
    """
    ---
    **Note:** The application does not retrain the model. It only evaluates the selected model on the uploaded test data.
    """
)
st.caption(
    """
    The application was developed by Bachcha Ram Harijan (BITS ID: 2026AJ05136) as part of the Machine Learning Assignment 2.
    """
)
        
