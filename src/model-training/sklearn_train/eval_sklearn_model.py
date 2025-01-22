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
from model_eval import ModelEvaluation
from utils import extract_features
from nilm_dao import get_vndale2_data, get_label_encoder, get_vndale1_data
from sklearn.model_selection import train_test_split

# Đường dẫn đến folder của mô hình
mode                = 0
window_size         = 1800
model_path          = f"{PROJECT_PATH}/results/models/VNDALE1/window_{window_size}/5_comb"
model_file          = f"rf_['Irms', 'P', 'MeanPF', 'S', 'Q'].joblib"
model_name          = model_file.replace(".joblib", "")
features            = extract_features(model_name)
label_encoder_path  = f"/opt/nilm-shared-data/nilm_device_detection/ICTA2024_dataset/label_encoder/classes.npy"
batch_size          = 2**12
data_type           = "train"

# Set up logging
log_path = f"{model_path}/model_evaluation"
os.makedirs(log_path, exist_ok=True)
logging.basicConfig(filename=f"{log_path}/{model_name}_model_evaluation.log", level=logging.INFO, 
                    format='%(asctime)s - %(levelname)s - %(message)s')

#Model valiation
print("[+] Model importing!")
model = load(f"{model_path}/{model_file}")
logging.info(f"[+] Model imported: {model_name}")
logging.info(f"Model: {model}")
print("[+] Model imported!")

#Select model for evaluation
label_encoder = get_label_encoder("vndale1")
val_df  = get_vndale1_data(data_type, window_size)
print(f"[+] Validation dataframe {len(val_df['Label'].unique())} with labels: {len(val_df['Label'].unique())}")
print(val_df.head())

# Import data
X_val = val_df.select(features).to_numpy()
y_val = val_df['Label'].to_numpy()
if data_type == "train":
    X_val, _, y_val, _ = train_test_split(X_val, y_val, test_size=0.9, random_state=42, stratify=y_val)
print(f"[+] Data imported: {X_val.shape} - {y_val.shape}")
val_df = None
# Evaluation on validation set
model_eval = ModelEvaluation()
all_y_pred, all_y_val = model_eval.split_test_set_into_batches(X_val, y_val, batch_size, model=model)
acc = accuracy_score(y_pred=all_y_pred, y_true=all_y_val)
f1 = f1_score(y_pred=all_y_pred, y_true=all_y_val, average='weighted')
precision = precision_score(y_pred=all_y_pred, y_true=all_y_val, average='weighted')
recall = recall_score(y_pred=all_y_pred, y_true=all_y_val, average='weighted')

logging.info(f"[+] Evaluation results with {data_type}: {model_name}, data size: {X_val.shape[0]}")
logging.info(f"Validation accuracy: {acc}")
logging.info(f"Validation F1-Score: {f1}")
logging.info(f"Precision score: {precision}")
logging.info(f"Recall score: {recall}")
logging.info(f"Classification report: {classification_report(y_true=all_y_val, y_pred=all_y_pred)}")

# report_df = model_eval.report_most_error_each_class(y_true=all_y_val, y_pred=all_y_pred, label_encoder=label_encoder)
# report_df.to_csv(f"{log_path}/{model_name.replace(".joblib", "")}_{data_type}_error_report.csv", index=False)

# Print evaluation results
# print(f"[+] Evaluation results with {data_type}: {model_name}")
# print(f"Validation accuracy: {acc}")
# print(f"Validation F1-Score: {f1}")
# print(f"Precision score: {precision}")
# print(f"Recall score: {recall}")