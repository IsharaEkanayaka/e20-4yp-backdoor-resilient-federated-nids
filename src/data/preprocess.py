import pandas as pd
import numpy as np
import torch
import os
from sklearn.preprocessing import MinMaxScaler
from sklearn.model_selection import train_test_split

# --- CONFIGURATION ---
RAW_TRAIN_PATH = "data/unsw-nb15/raw/UNSW_NB15_training-set.csv"
RAW_TEST_PATH = "data/unsw-nb15/raw/UNSW_NB15_testing-set.csv"
PROCESSED_DIR = "data/unsw-nb15/processed"

# Features that follow a Power Law (huge range) and need Log transform
LOG_COLS = ['dur', 'sbytes', 'dbytes', 'Sload', 'Dload', 'Spkts', 'Dpkts']

# Categorical features to One-Hot Encode
CAT_COLS = ['proto', 'service', 'state']

def clean_and_process():
    print("🚀 Starting Data Preprocessing Pipeline...")

    # 1. LOAD & MERGE
    print(f"   📂 Loading raw files from {os.path.dirname(RAW_TRAIN_PATH)}...")
    df1 = pd.read_csv(RAW_TRAIN_PATH)
    df2 = pd.read_csv(RAW_TEST_PATH)
    
    # Concatenate to create one big pool for consistent scaling
    df = pd.concat([df1, df2], ignore_index=True)
    print(f"   ✅ Merged Dataset Shape: {df.shape}")

    # 2. BASIC CLEANING
    # Drop ID (useless) and 'label' (binary redundancy of attack_cat)
    # We keep 'attack_cat' as our target.
    drop_cols = ['id', 'label'] 
    df = df.drop(columns=[c for c in drop_cols if c in df.columns], errors='ignore')

    # Handle NaNs (rare in UNSW, but good practice)
    df = df.fillna(0)

    # 3. NUMERICAL SCALING (Log + MinMax)
    print("   ⚖️ Scaling Numerical Features...")
    
    # A) Log Transform heavy columns
    for col in LOG_COLS:
        if col in df.columns:
            # clip(lower=0) ensures no negative values before log
            df[col] = np.log1p(df[col].clip(lower=0))

    # B) MinMax Scale ALL numerical columns (except target)
    # Select only numeric types first
    num_cols = df.select_dtypes(include=['float64', 'int64']).columns
    # Remove target column from scaling if it accidentally got selected
    if 'attack_cat' in num_cols:
        num_cols = num_cols.drop('attack_cat')
    
    scaler = MinMaxScaler(feature_range=(0, 1))
    df[num_cols] = scaler.fit_transform(df[num_cols])

    # 4. CATEGORICAL ENCODING (Top-K + One-Hot)
    print("   🏷️ Encoding Categorical Features...")
    for col in CAT_COLS:
        if col in df.columns:
            # Keep Top 10 frequent values, map rest to 'other'
            top_10 = df[col].value_counts().nlargest(10).index
            df.loc[~df[col].isin(top_10), col] = 'other'
    
    # One-Hot Encode (creates boolean columns, then casts to float)
    df = pd.get_dummies(df, columns=CAT_COLS, dtype=float)

    # 5. TARGET ENCODING
    # We need to turn strings ('Normal', 'Analysis', 'Backdoor') into Integers (0, 1, 2)
    print("   🎯 Encoding Targets...")
    # Ensure 'Normal' is always 0 for consistency
    unique_attacks = sorted(df['attack_cat'].unique())
    # Move 'Normal' to index 0 manually if present
    if 'Normal' in unique_attacks:
        unique_attacks.remove('Normal')
        unique_attacks.insert(0, 'Normal')
    
    # Create the map
    label_map = {name: i for i, name in enumerate(unique_attacks)}
    df['attack_cat'] = df['attack_cat'].map(label_map)
    
    print(f"   ℹ️ Class Mapping: {label_map}")

    # 6. SPLIT & SAVE
    print("   💾 Saving PyTorch Tensors...")
    X = df.drop('attack_cat', axis=1).values.astype('float32')
    y = df['attack_cat'].values.astype('int64')

    # Stratified Split (80% Train Pool, 20% Global Test)
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.20, random_state=42, stratify=y
    )

    # Save dictionary containing Tensors + Metadata (feature count, class map)
    os.makedirs(PROCESSED_DIR, exist_ok=True)
    
    train_payload = {
        'X': torch.tensor(X_train),
        'y': torch.tensor(y_train),
        'label_map': label_map
    }
    
    test_payload = {
        'X': torch.tensor(X_test),
        'y': torch.tensor(y_test),
        'label_map': label_map
    }

    torch.save(train_payload, f"{PROCESSED_DIR}/train_pool.pt")
    torch.save(test_payload, f"{PROCESSED_DIR}/global_test.pt")

    print(f"   ✅ SUCCESS! Processed data saved to {PROCESSED_DIR}")
    print(f"   📊 Final Feature Count: {X.shape[1]}")

if __name__ == "__main__":
    clean_and_process()