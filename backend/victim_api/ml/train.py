import os
import joblib
import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import classification_report, accuracy_score, f1_score, precision_score, recall_score, roc_auc_score, average_precision_score
from sklearn.datasets import load_breast_cancer

# Paths
MODEL_DIR = os.path.dirname(os.path.abspath(__file__))
MODEL_PATH = os.path.join(MODEL_DIR, "fraud_model.pkl")

def get_real_data():
    print("Loading Financial Fraud dataset (10,000 records, 50/50 balance)...")
    dataset_path = os.path.abspath(os.path.join(MODEL_DIR, "..", "..", "..", "dataset.csv"))
    if not os.path.exists(dataset_path):
        print("Dataset missing! Downloading and generating now...")
        import subprocess, sys
        script_path = os.path.join(os.path.dirname(dataset_path), "download_dataset.py")
        subprocess.run([sys.executable, script_path], check=True)
    
    df = pd.read_csv(dataset_path)
    X = df.drop(columns=["Class"])
    # Map classes to binary 0/1 for scikit-learn metric calculations
    y = df["Class"].map({"legitimate": 0, "fraud": 1})
    return X, y

if __name__ == "__main__":
    X, y = get_real_data()
    
    print("Splitting dataset (80/20 Stratified)...")
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42, stratify=y)
    
    print("Training Random Forest model...")
    model = RandomForestClassifier(
        n_estimators=100, max_depth=None, 
        random_state=42, n_jobs=1
    )
    model.fit(X_train, y_train)
    
    print("Evaluating model...")
    y_pred = model.predict(X_test)
    y_pred_probs = model.predict_proba(X_test)[:, 1]
    
    acc = accuracy_score(y_test, y_pred)
    f1 = f1_score(y_test, y_pred)
    prec = precision_score(y_test, y_pred)
    rec = recall_score(y_test, y_pred)
    roc_auc = roc_auc_score(y_test, y_pred_probs)
    pr_auc = average_precision_score(y_test, y_pred_probs)
    
    print("\n--- Model Metrics ---")
    print(f"Accuracy:        {acc:.4f}")
    print(f"F1 Score:        {f1:.4f}")
    print(f"Precision:       {prec:.4f}")
    print(f"Recall:          {rec:.4f}")
    print(f"AUC-ROC:         {roc_auc:.4f}")
    print(f"PR AUC:          {pr_auc:.4f}")
    print("\nFull Classification Report:")
    print(classification_report(y_test, y_pred))
    
    print(f"\nSaving model to {MODEL_PATH}...")
    joblib.dump(model, MODEL_PATH)
    print("Training complete.")


