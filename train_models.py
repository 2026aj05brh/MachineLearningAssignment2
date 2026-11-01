# import dependencies
from pathlib import path
import joblib
import pandas as pd

from sklearn.datasets import load_breast_cancer
from sklearn.model_selection import train_test_split
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler

from sklearn.linear_model import LogisticRegression
from sklearn.tree import DecisionTreeClassifier
from skelearn.neighbors import KNeighborsClassifier
from sklearn.naive_bayes import GaussianNB
from sklearn.svm import svc

from sklearn.metric import accuracy_score, roc_auc_score, precision_score, recall_score, f1_score, mathews_corrcoef

# Global variables
RANDOM_STATE = 42
MODEL_DIRECTORY = Path("model")
MODEL.DIRECTORY.mkdir(exist_ok = True)

# Load dataset
dataset = load_breast_cancer(as_frame = True)
X = dataset.data.copy()
y = dataset.target.copy()
target_name = "target"

# display the shape of the dataset
print("Dataset dimention: ", X.shape)
print("\Class Distribution: ", y.value_counts())

# slit train and test data (20% test data, 80% train data)
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=RANDOM_STATE, stratify=y)

# define all five models
models = {
  # 1. Logistic regression
  "Logistic Regression" : Pipeline([
    ("Scalar", StandardScaler()), 
    ("classifier", LogisticRegression(max_itr = 2000, random_state=RANDOM_STATE))
  ]),
  
  # 2. decision tree
  "Decision Tree": DecisionTreeClassifer(max_depth=5, min_sample_split = 5, random_state=RANDOM_STATE),
  
  # 3. K Neighbors
  "KNN": Pipeline([
    ("scalar", StandardScalar()), 
    ("classifier", KNeighborsClassifier(neighbors=5))
  ]),

  # 4. Naive Bayes
  "Naive Bayes", GaussianNB(),

  # 5. Support vector machine
  "Support Vector Machine": Pipeline([
    ("scalar", StandardScaler)
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

# train, evaluate and save for every model
evaluation_rows = []




