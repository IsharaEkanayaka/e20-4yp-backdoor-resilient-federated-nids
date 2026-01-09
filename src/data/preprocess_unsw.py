import pandas as pd
import numpy as np
import torch
import os
from sklearn.preprocessing import LabelEncoder, StandardScaler
from sklearn.model_selection import train_test_split 

RAW_DIR = "data/unsw-nb15/raw/"
PROCESSED_DIR = "data/unsw-nb15/processed"

UNSW_COLS = [
    'srcip', 'sport', 'dstip', 'dsport', 'proto', 'state', 'dur', 'sbytes', 'dbytes', 'sttl', 
    'dttl', 'sloss', 'dloss', 'service', 'Sload', 'Dload', 'Spkts', 'Dpkts', 'swin', 'dwin', 
    'stcpb', 'dtcpb', 'smeansz', 'dmeansz', 'trans_depth', 'res_bdy_len', 'Sjit', 'Djit', 
    'Stime', 'Ltime', 'Sintpkt', 'Dintpkt', 'tcprtt', 'synack', 'ackdat', 'is_sm_ips_ports', 
    'ct_state_ttl', 'ct_flw_http_mthd', 'is_ftp_login', 'ct_ftp_cmd', 'ct_srv_src', 'ct_srv_dst', 
    'ct_dst_ltm', 'ct_src_ ltm', 'ct_src_dport_ltm', 'ct_dst_sport_ltm', 'ct_dst_src_ltm', 
    'attack_cat', 'label'
]

def load_data():
    RAW_DIR = "data/unsw-nb15/raw/"
    PROCESSED_DIR = "data/unsw-nb15/processed"

    target_files = [
            "UNSW-NB15_1.csv",
            "UNSW-NB15_2.csv",
            "UNSW-NB15_3.csv",
            "UNSW-NB15_4.csv"
        ]
    file_paths = [os.path.join(RAW_DIR, f) for f in target_files if os.path.exists(os.path.join(RAW_DIR, f))]

    if len(file_paths) != 4:
        raise FileNotFoundError(f"❌ Expected 4 UNSW-NB15 files, found {len(file_paths)}")

    print(f"📂 Found {len(file_paths)} files. Merging... (This may take RAM)")

    df_list = []
    for filename in file_paths:
        print(f"   Reading {os.path.basename(filename)}...")
        df_temp = pd.read_csv(filename, header=None, names=UNSW_COLS, low_memory=False)
        df_temp.columns = df_temp.columns.str.strip()
        df_list.append(df_temp)

    df_data = pd.concat(df_list, ignore_index=True)
    print(f"✅ Merged Dataset Shape: {df_data.shape}")
    return df_data

def clean_and_process(df_data):
    # ============================================
    # STEP 1: HANDLE MISSING VALUES
    # ============================================
    df_data['attack_cat'] = df_data['attack_cat'].fillna('Normal')
    df_data['is_ftp_login'] = df_data['is_ftp_login'].fillna(0)
    df_data['ct_flw_http_mthd'] = df_data['ct_flw_http_mthd'].fillna(0)

    # Clean whitespace in attack_cat
    df_data['attack_cat'] = df_data['attack_cat'].str.strip()

    # ============================================
    # STEP 2: DROP HIGH-CARDINALITY FEATURES
    # ============================================
    columns_to_drop = ['sport', 'dsport', 'srcip', 'dstip']
    df_data = df_data.drop(columns=columns_to_drop)

    # ============================================
    # STEP 3: ENCODE CATEGORICAL FEATURES
    # ============================================
    # Label encode 'proto'
    le_proto = LabelEncoder()
    df_data['proto'] = le_proto.fit_transform(df_data['proto'])

    # Encode target variable
    le_target = LabelEncoder()
    y = le_target.fit_transform(df_data['attack_cat'])

    # ============================================
    # STEP 4: SEPARATE FEATURES AND TARGET
    # ============================================
    X = df_data.drop(['attack_cat', 'label'], axis=1)

    # One-hot encode remaining categorical features
    categorical_cols = ['state', 'service', 'ct_ftp_cmd']
    X_encoded = pd.get_dummies(X, columns=categorical_cols, drop_first=False)

    # ============================================
    # STEP 5: TRAIN/TEST SPLIT (DO THIS BEFORE SCALING!)
    # ============================================
    X_train, X_test, y_train, y_test = train_test_split(
        X_encoded, y, test_size=0.2, random_state=42, stratify=y
    )

    # ============================================
    # STEP 6: FEATURE SCALING (FIT ON TRAIN ONLY!)
    # ============================================
    # CRITICAL: Fit scaler ONLY on training data to prevent data leakage
    scaler = StandardScaler()
    X_train_scaled = scaler.fit_transform(X_train)  # Fit and transform train
    X_test_scaled = scaler.transform(X_test)        # Only transform test (no fit!)

    print(f"✅ Preprocessing complete!")
    print(f"   Training samples: {X_train_scaled.shape[0]}")
    print(f"   Testing samples: {X_test_scaled.shape[0]}")
    print(f"   Feature dimension: {X_train_scaled.shape[1]}")
    print(f"   Number of classes: {len(np.unique(y))}")

    # ============================================
    # STEP 7: CONVERT TO PYTORCH TENSORS
    # ============================================
    X_train_tensor = torch.FloatTensor(X_train_scaled)
    y_train_tensor = torch.LongTensor(y_train)
    X_test_tensor = torch.FloatTensor(X_test_scaled)
    y_test_tensor = torch.LongTensor(y_test)

    # ============================================
    # STEP 8: SAVE TENSORS TO DISK
    # ============================================
    # Save training data with class mapping
    train_payload = {
        'X': X_train_tensor,
        'y': y_train_tensor,
        'label_map': le_target.classes_  # Attack category names
    }

    # Save test data with class mapping
    test_payload = {
        'X': X_test_tensor,
        'y': y_test_tensor,
        'label_map': le_target.classes_  # Attack category names
    }
    print("   💾 Saving PyTorch Tensors...")
    os.makedirs(PROCESSED_DIR, exist_ok=True)
    torch.save(train_payload, f"{PROCESSED_DIR}/train_pool.pt")
    torch.save(test_payload, f"{PROCESSED_DIR}/global_test.pt")

    print(f"   ✅ Done! ")
    print(f"\n📋 Class mapping (index -> attack type):")
    for idx, class_name in enumerate(le_target.classes_):
        print(f"   {idx}: {class_name}")
    print(f"\n🎯 Initialize your model with:")
    print(f"   model = Net(input_dim={X_train_scaled.shape[1]}, num_classes={len(np.unique(y))})")
if __name__ == '__main__':
    df_data = load_data()
    clean_and_process(df_data)