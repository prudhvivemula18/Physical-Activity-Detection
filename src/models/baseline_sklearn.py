# baseline_sklearn.py
import os
import pandas as pd
import joblib
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split
from src.data.load_ucihar import load_ucihar


# Import from your modular scripts
# Placeholder preprocess function for sample dataset
def preprocess_data(df, feature_cols):
    # For now, just return the dataframe as-is
    return df

from src.features.feature_extraction import extract_features
# Placeholder feature extraction function for sample dataset
def extract_features(df):
    # For sample dataset, just return all columns except the label
    feature_cols = [col for col in df.columns if col != 'Activity_Type']
    return df[feature_cols]

from src.eval.eval_metrics import evaluate_model

# -------------------------------
# Step 0: Ensure results folder exists
# -------------------------------
os.makedirs("results", exist_ok=True)

# -------------------------------
# Step 1: Load dataset
# -------------------------------
# Replace this path with your actual CSV file

# data_path = "data/processed/activity_data.csv"  
# df = pd.read_csv(data_path)
# Load real UCI HAR dataset
X_train, y_train, X_test, y_test = load_ucihar()

# -------------------------------
# Step 2: Define features & label
# -------------------------------
# Replace 'Activity_Type' with your label column name
label_col = 'Activity_Type'

# If your CSV has extra columns (e.g., timestamp), list features explicitly
# feature_cols = [col for col in df.columns if col != label_col]

# -------------------------------
# Step 3: Preprocess data
# -------------------------------
# df = preprocess_data(df, feature_cols)
# ( i already have X_train, y_train, X_test, y_test from UCI HAR, so no need to preprocess the old df.)
# -------------------------------
# Step 4: Optional feature extraction (if needed)
# -------------------------------
# features_df = extract_features(df, feature_cols)
# X = features_df

# X = df[feature_cols]
# y = df[label_col]

# -------------------------------
# Step 5: Split train/test
# -------------------------------
# X_train, X_test, y_train, y_test = train_test_split(
#     X, y, test_size=0.2, random_state=42
# )

# -------------------------------
# Step 6: Train Random Forest
# -------------------------------
clf = RandomForestClassifier(n_estimators=100, random_state=42)
clf.fit(X_train, y_train)

# -------------------------------
# Step 7: Evaluate
# -------------------------------
y_pred = clf.predict(X_test)
evaluate_model(clf, X_test, y_test)

# -------------------------------
# Step 8: Save model
# -------------------------------
model_path = "results/random_forest_model.pkl"
joblib.dump(clf, model_path)
print(f"Trained model saved to {model_path}")
