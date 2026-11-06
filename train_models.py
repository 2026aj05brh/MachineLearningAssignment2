# import dependencies
from pathlib import Path
import joblib
import pandas as pd

# from sklearn dependencies

from sklearn.datasets import load_breast_cancer
from sklearn.model_selection import train_test_split
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler

from sklearn.linear_model import LogisticRegression
from sklearn.tree import DecisionTreeClassifier
from sklearn.neighbors import KNeighborsClassifier
from sklearn.naive_bayes import GaussianNB
from sklearn.svm import SVC

from sklearn.metrics import accuracy_score, roc_auc_score, precision_score, recall_score, f1_score, matthews_corrcoef

# Global variables
RANDOM_STATE = 42
MODEL_DIRECTORY = Path("model")
MODEL_DIRECTORY.mkdir(exist_ok = True)

# 1. Load dataset
dataset = load_breast_cancer(as_frame = True)
X = dataset.data.copy()
y = dataset.target.copy()
target_name = "target"

# display the shape of the dataset
print("Dataset dimention: ", X.shape)
print("\nClass Distribution: ", y.value_counts())

# 2. slit train and test data (20% test data, 80% train data)
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=RANDOM_STATE, stratify=y)

# 3. define all five models
models = {
    # 1. Logistic regression
    "Logistic Regression" : Pipeline([
        ("Scalar", StandardScaler()), 
        ("classifier", LogisticRegression(max_iter = 2000, random_state=RANDOM_STATE))
    ]),
  
    # 2. decision tree
    "Decision Tree": DecisionTreeClassifier(max_depth=5, min_samples_split=5, random_state=RANDOM_STATE),
  
    # 3. K Neighbors
    "KNN": Pipeline([
        ("scalar", StandardScaler()), 
        ("classifier", KNeighborsClassifier(n_neighbors=5))
    ]),

    # 4. Naive Bayes
    "Naive Bayes": GaussianNB(),

    # 5. Support vector machine
    "Support Vector Machine": Pipeline([
        ("scalar", StandardScaler()),
        ("classifier", SVC(kernel = "rbf", probability=True, random_state=RANDOM_STATE))
    ])
}

# model file names
model_file_names = {
    "Logistic Regression":"logistic_regression.joblib",
    "Decision Tree":"decision_tree.joblib",
    "KNN": "knn.joblib",
    "Naive Bayes": "naive_bayes.joblib",
    "Support Vector Machine": "support_vector_machine.joblib"
}

# 4. train, evaluate and save for every model
evaluation_rows = []

for model_name, model in models.items():
    print(f"\nTraining {model_name}...")
    # train the model
    model.fit(X_train, y_train)

    # predict the classes
    predicted_classes = model.predict(X_test)
    predicted_probabilities = model.predict_proba(X_test)[:, 1]

    # calculate the evaluation scores
    row = {
        "ML Model Name": model_name,
        "Accuracy": accuracy_score(y_test, predicted_classes),
        "AUC": roc_auc_score(y_test, predicted_probabilities),
        "Precision": precision_score(y_test, predicted_classes, zero_division=0),
        "Recall": recall_score(y_test, predicted_classes, zero_division=0),
        "F1": f1_score(y_test, predicted_classes, zero_division=0),
        "MCC": matthews_corrcoef(y_test, predicted_classes)
    }

    evaluation_rows.append(row)
    save_path = MODEL_DIRECTORY / model_file_names[model_name]
    joblib.dump(model, save_path)

    print(f"Saved model to {save_path}") 

# 5. save feature names
joblib.dump(list(X.columns), MODEL_DIRECTORY / "feature_names.joblib")

# 6. save_test data with target column
test_data = X_test.copy()
test_data[target_name] = y_test.copy()
test_data.to_csv("test_data.csv", index=False)

# 7. save and print comparision
results = pd.DataFrame(evaluation_rows)
metric_columns = ["Accuracy", "AUC", "Precision", "Recall", "F1", "MCC"]
results[metric_columns] = results[metric_columns].round(4)

print("\nModel comparision:")
print(results.to_string(index=False))

winner_row = results.loc[results["MCC"].idxmax()]
print("\nOverall winner based on MCC: ", winner_row["ML Model Name"])
