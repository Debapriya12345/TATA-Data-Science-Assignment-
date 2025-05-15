import numpy as np                      # For numerical operations
import pandas as pd                     # For data manipulation and analysis
import matplotlib.pyplot as plt         # For data visualization

# 🔧 Preprocessing and Pipeline Tools from Scikit-learn
from sklearn.pipeline import Pipeline                           # For building ML pipelines
from sklearn.preprocessing import OneHotEncoder, StandardScaler # For categorical encoding and feature scaling
from sklearn.compose import ColumnTransformer                   # For applying different preprocessing to columns


import matplotlib.pyplot as plt
import seaborn as sns
import plotly.express as px

# ⚖️ Handling Imbalanced Data
from imblearn.combine import SMOTEENN                           # Combines SMOTE and ENN for class balancing
from imblearn.pipeline import Pipeline as ImbPipeline           # Pipeline that supports imbalanced-learn transformers

# 📊 Model Training and Evaluation
from sklearn.model_selection import train_test_split, learning_curve, RandomizedSearchCV


# 🔍 Feature Selection
from sklearn.feature_selection import SelectFromModel           # Select features based on importance weights

# 🌲 Machine Learning Model
from sklearn.ensemble import RandomForestClassifier             # Random forest model for classification tasks

# 📈 Evaluation Metrics
from sklearn.metrics import (accuracy_score, precision_score, recall_score,
                             f1_score, roc_auc_score, classification_report,
                             roc_curve, ConfusionMatrixDisplay)
import pickle


df = pd.read_csv("WA_Fn-UseC_-Telco-Customer-Churn.csv")
df.head()
df['TotalCharges'] = pd.to_numeric(df['TotalCharges'], errors='coerce')
df = df.dropna()

y = df['Churn'].map({'No': 0, 'Yes': 1})  # Encode target to 0/1
X = df.drop(['customerID', 'Churn'], axis=1)  # Drop ID & raw target

categorical_cols = X.select_dtypes(include=['object']).columns.tolist()
numerical_cols   = X.select_dtypes(include=['int64', 'float64']).columns.tolist()

# Split resampled data into train & test sets
X_train, X_test, y_train, y_test = train_test_split(
    X,
    y,
    test_size=0.2,
    random_state=42
)

# Define transformers
numeric_transformer = Pipeline(steps=[
    ('scaler', StandardScaler())
])

categorical_transformer = Pipeline(steps=[
    ('onehot', OneHotEncoder(drop='first', sparse_output=False, handle_unknown='ignore'))
])

preprocessor = ColumnTransformer(transformers=[
    ('num', numeric_transformer, numerical_cols),
    ('cat', categorical_transformer, categorical_cols)
])

# Full pipeline: preprocessing + model
model_pipeline = Pipeline(steps=[
    ('preprocessor', preprocessor),
    ('classifier', RandomForestClassifier())
])

# Fit it
model_pipeline.fit(X_train, y_train)


print("Training set shape:", X_train.shape, y_train.shape)
print("Testing set shape:", X_test.shape, y_test.shape)


rf = model_pipeline

# Fit the model on the training data
rf.fit(X_train, y_train)

# Generate predictions and probabilities
y_pred  = rf.predict(X_test)              # class labels
y_proba = rf.predict_proba(X_test)[:, 1]  # probability of positive class (Churn)

# Print key metrics
print("Default Random Forest Performance")
print(f"F1-score : {f1_score(y_test, y_pred):.4f}")
print(f"AUC-ROC  : {roc_auc_score(y_test, y_proba):.4f}\n")
print(classification_report(y_test, y_pred))

# 📌Save Model Pipeline  and Artifacts Separately
model_path = "models/Randomforest_model.pkl"
pickle.dump(rf, open(model_path, 'wb'))
print("Saved the pickle File of Randomforest model")

print("EOF")