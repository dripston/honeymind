import os
import pandas as pd
import xgboost as xgb
from sklearn.model_selection import train_test_split
from sklearn.metrics import classification_report
import joblib

DIR = os.path.dirname(__file__)
DATA_PATH = os.path.join(DIR, "training_data.csv")
MODEL_PATH = os.path.join(DIR, "xgboost_detector.pkl")

def main():
    if not os.path.exists(DATA_PATH):
        print(f"Error: {DATA_PATH} not found. Run generate_training_data.py first.")
        return
        
    print("Loading training data...")
    df = pd.read_csv(DATA_PATH)
    
    # Use all 8 geometric and statistical features
    X = df.drop(columns=["label"])
    y = df["label"]
    
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42, stratify=y)
    
    print("Training XGBoost Multiclass Classifier...")
    model = xgb.XGBClassifier(
        objective="multi:softprob",
        num_class=5,
        max_depth=4,
        learning_rate=0.1,
        n_estimators=100,
        random_state=42
    )
    
    model.fit(X_train, y_train)
    
    print("\nEvaluating Model...")
    y_pred = model.predict(X_test)
    
    target_names = ["Normal", "Knockoff", "JBDA", "Analytical", "Evolutionary"]
    print(classification_report(y_test, y_pred, target_names=target_names))
    
    print(f"Saving model to {MODEL_PATH}...")
    joblib.dump(model, MODEL_PATH)
    print("Done!")

if __name__ == "__main__":
    main()
