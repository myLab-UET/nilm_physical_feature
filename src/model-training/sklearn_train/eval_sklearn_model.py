import sys
PROJECT_PATH = "<your_path>"
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
from nilm_dao import *
from sklearn.model_selection import train_test_split

import argparse
# Đường dẫn đến folder của mô hình
parser = argparse.ArgumentParser(description="Script parameters for model evaluation")
parser.add_argument("--dataset", type=str, default="vndale1", help="Name of the dataset")
parser.add_argument("--data_type", type=str, default="test", help="Data type")
parser.add_argument("--model_file", type=str, default="rf_['Irms', 'P', 'MeanPF', 'S', 'Q'].joblib", help="Model file name")
parser.add_argument("--window_size", type=int, default=1800, help="Window size value")
parser.add_argument("--batch_size", type=int, default=4096, help="Batch size")
parser.add_argument("--is_norm", type=lambda x: (str(x).lower() in ['true', '1']), default=False, help="Normalization flag")
parser.add_argument("--size", type=float, default=1, help="Data size")

args = parser.parse_args()
dataset_name = args.dataset
window_size  = args.window_size
model_file   = args.model_file
batch_size   = args.batch_size
data_type    = args.data_type
is_norm      = args.is_norm
size         = args.size

# Set up model
print("[+] Arguments: ", args)
model_name          = model_file.replace(".joblib", "")
features            = extract_features(model_file)
if dataset_name == "vndale1":
    model_path = f"{PROJECT_PATH}/results/models/VNDALE1/window_{window_size}/{len(features)}_comb"
elif dataset_name == "rae":
    model_path = f"{PROJECT_PATH}/results/models/rae/{len(features)}_comb"
elif dataset_name == "iawe":
    model_path = f"{PROJECT_PATH}/results/models/iawe/{len(features)}_comb"
else:
    raise ValueError("Invalid dataset name!")

# Set up logging
log_path = f"{model_path}/model_evaluation"
if not os.path.exists(log_path):
    os.makedirs(log_path)
logging.basicConfig(filename=f"{log_path}/{model_name}_model_evaluation.log", level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')

#Model valiation
print("[+] Model importing!")
model = load(f"{model_path}/{model_file}")
logging.info(f"[+] Model imported: {model_name}")
logging.info(f"Model: {model}")
print("[+] Model imported!")

#Select model for evaluation
if dataset_name == "vndale1":
    val_df  = get_vndale1_data(data_type, window_size, is_norm)
elif dataset_name == "iawe":
    val_df = get_iawe_data(data_type, is_norm)
elif dataset_name == "rae":
    val_df = get_rae_data(data_type, is_norm)
else:
    raise ValueError("Invalid dataset name!")
print(f"[+] Validation dataframe {len(val_df['Label'].unique())} with labels: {len(val_df['Label'].unique())}")
print(val_df.head())

# Import data
X_val = val_df.select(features).to_numpy()
y_val = val_df['Label'].to_numpy()
if args.size < 1:
    X_val, _, y_val, _ = train_test_split(X_val, y_val, train_size=size, random_state=42)
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

# Classification report
label_encoder = get_label_encoder(dataset_name)
classification_rep = classification_report(y_true=all_y_val, y_pred=all_y_pred, target_names=label_encoder.classes_)
logging.info(f"Classification report: {classification_rep}")
report_df = model_eval.report_most_error_each_class(y_true=all_y_val, y_pred=all_y_pred, label_encoder=label_encoder)
report_df.to_csv(f"{log_path}/{model_name.replace('.joblib', '')}_{data_type}_error_report.csv", index=False)