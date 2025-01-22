import sys
PROJECT_PATH = "/home/mrcong/Code/mylab-nilm-files/nilm-physical-features"
sys.path.append(f"{PROJECT_PATH}/src/common")

import numpy as np
import polars as pl
import pandas as pd
import os
from joblib import dump, load
from tqdm import tqdm
from sklearn.metrics import classification_report, accuracy_score, f1_score, precision_score, recall_score
import logging
import onnxruntime as ort
from model_eval import ModelEvaluation
from utils import extract_features
from nilm_dao import get_vndale2_data, get_label_encoder

# Đường dẫn đến folder của mô hình
mode                = 0
window_size         = 200
model_path          = f"{PROJECT_PATH}/results/models/VNDALE2/window_{window_size}/5_comb"
model_file          = f"rf_['Irms', 'P', 'MeanPF', 'S', 'Q'].onnx"
model_name          = model_file.replace(".onnx", "")
features            = extract_features(model_name)
label_encoder_path  = f"/opt/nilm-shared-data/nilm_device_detection/ICTA2024_dataset/label_encoder/classes.npy"
batch_size          = 2**12
data_type           = "val"

# Set up logging
log_path = f"{model_path}/model_evaluation"
os.makedirs(log_path, exist_ok=True)
logging.basicConfig(filename=f"{log_path}/{model_name}_model_evaluation.log", level=logging.INFO, 
                    format='%(asctime)s - %(levelname)s - %(message)s')

# Select model for evaluation
label_encoder = get_label_encoder()
val_df  = get_vndale2_data(data_type, window_size)
print(f"[+] Validation dataframe {len(val_df['Label'].unique())} with labels: {len(val_df['Label'].unique())}")
print(val_df.head())

# Load ONNX model
print("[+] Model importing!")
onnx_model_path = f"{model_path}/{model_file}"
ort_session = ort.InferenceSession(onnx_model_path)
print("[+] Model imported!")

# Import data
X_val = val_df.select(features).to_numpy()
y_val = val_df['Label'].to_numpy()
print(f"[+] Data imported: {X_val.shape} - {y_val.shape}")

# Function to predict in batches
def predict_in_batches(ort_session, X, batch_size):
    y_pred = []
    for i in range(0, len(X), batch_size):
        X_batch = X[i:i + batch_size]
        ort_inputs = {ort_session.get_inputs()[0].name: X_batch.astype(np.float32)}
        ort_outs = ort_session.run(None, ort_inputs)
        y_pred_batch = np.argmax(ort_outs[0], axis=1)
        y_pred.extend(y_pred_batch)
    return np.array(y_pred)

# Evaluation on validation set
all_y_pred = predict_in_batches(ort_session, X_val, batch_size)
all_y_val = y_val

acc = accuracy_score(y_pred=all_y_pred, y_true=all_y_val)
f1 = f1_score(y_pred=all_y_pred, y_true=all_y_val, average='weighted')

print(f"Accuracy: {acc}")
print(f"F1 Score: {f1}")

logging.info(f"Accuracy: {acc}")
logging.info(f"F1 Score: {f1}")