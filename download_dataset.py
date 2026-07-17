import os
import pandas as pd
from sklearn.datasets import fetch_openml

def download_and_balance_kaggle_dataset():
    print("Downloading the Real Kaggle Credit Card Fraud Dataset from OpenML (this may take a minute)...")
    # Download the dataset (OpenML ID 1597 is the famous Kaggle Credit Card Fraud dataset)
    data = fetch_openml(data_id=1597, parser='auto')
    
    df = data.frame
    
    print(f"Original Dataset Shape: {df.shape}")
    print(f"Original Class Balance:\n{df['Class'].value_counts()}")
    
    # The dataset has classes '0' (legitimate) and '1' (fraud)
    # Let's separate the classes
    fraud_df = df[df['Class'] == '1']
    legit_df = df[df['Class'] == '0']
    
    # Perfectly balance the dataset by undersampling the legitimate transactions
    # to match the exact number of fraud transactions (~492)
    print("\nBalancing dataset...")
    legit_downsampled = legit_df.sample(n=len(fraud_df), random_state=42)
    
    # Combine back together
    balanced_df = pd.concat([fraud_df, legit_downsampled])
    
    # Shuffle the dataset
    balanced_df = balanced_df.sample(frac=1, random_state=42).reset_index(drop=True)
    
    # The Kaggle dataset has columns: V1..V28, Time, Amount, Class (31 columns total if Time is included, OpenML has 29 features + 1 target = 30)
    # To perfectly match our API which expects V1 to V30, we will rename the columns
    # V1 to V28 remain the same. We rename V29 and V30.
    feature_cols = list(balanced_df.columns)
    feature_cols.remove('Class')
    
    # OpenML Kaggle dataset has 29 features. Our API expects 30.
    # Add a dummy feature to make it exactly 30 features.
    balanced_df['Dummy_Feature'] = 0.0
    feature_cols.append('Dummy_Feature')
    
    rename_mapping = {}
    for i, col in enumerate(feature_cols):
        rename_mapping[col] = f"V{i+1}"
        
    balanced_df = balanced_df.rename(columns=rename_mapping)
    
    # Map the class labels for clarity
    balanced_df['Class'] = balanced_df['Class'].map({'0': 'legitimate', '1': 'fraud'})
    
    # Save to CSV
    out_path = os.path.join(os.path.dirname(os.path.abspath(__file__)), "dataset.csv")
    balanced_df.to_csv(out_path, index=False)
    
    print(f"\nSuccessfully saved perfectly balanced Kaggle Dataset to {out_path}")
    print(f"Final Shape: {balanced_df.shape}")
    print(f"Final Class Balance:\n{balanced_df['Class'].value_counts()}")

if __name__ == "__main__":
    download_and_balance_kaggle_dataset()
