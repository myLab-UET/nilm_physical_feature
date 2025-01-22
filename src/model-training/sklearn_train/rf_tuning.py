import sys
PROJECT_PATH = "/home/mrcong/Code/mylab-nilm-files/nilm-physical-features"
sys.path.append(f"{PROJECT_PATH}/src/common")

from sklearn.ensemble import RandomForestClassifier
from joblib import dump
import numpy as np
import pandas as pd
import polars as pl
import datetime
import os
import logging
import argparse
from sklearn.metrics import accuracy_score, f1_score
from nilm_dao import get_vndale2_data, get_vndale1_data
from model_eval import ModelEvaluation
from sklearn.model_selection import train_test_split

# Load data
parser = argparse.ArgumentParser()
parser.add_argument("--rms_window_size", type=int, default=1800, help="Size of the RMS window")
args = parser.parse_args()

if args.rms_window_size <= 0:
    raise ValueError("rms_window_size must be a positive integer")

window_size = args.rms_window_size
train_df = get_vndale1_data("train", window_size)
val_df = get_vndale1_data("val", window_size)

# Set up logging
logger = logging.getLogger()
model_eval = ModelEvaluation()
date_time = datetime.datetime.now().strftime("%Y-%m-%d")
MODEL_DIR = f"{PROJECT_PATH}/results/models/VNDALE1/window_{window_size}/5_comb/rf_tuning"
if not os.path.exists(MODEL_DIR):
    os.makedirs(MODEL_DIR)
log_file_path = f"{MODEL_DIR}/{date_time}_rf_tuning.log"
logging.basicConfig(level=logging.INFO, 
                    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
                    handlers=[
                        logging.FileHandler(log_file_path),
                        logging.StreamHandler(sys.stdout)
                    ])

features = ['Irms', 'P', 'MeanPF', 'S', 'Q']
X_train = train_df.select(features).to_numpy()
y_train = train_df["Label"].to_numpy()
X_val = val_df.select(features).to_numpy()
y_val = val_df["Label"].to_numpy()

# Reduce the size of the training set
X_train, _, y_train, _ = train_test_split(X_train, y_train, test_size=0.9, random_state=42, stratify=y_train)
logger.info(f"[+] Train data imported: {X_train.shape} - {y_train.shape}")
logger.info(f"[+] Validation data imported: {X_val.shape} - {y_val.shape}")
train_df, val_df = None, None

# Define the parameter grid
param_grid = {
    'n_estimators': range(5, 21, 1),
    'max_depth': range(15, 31, 1)
}

best_f1_score = 0
best_params = None
best_model = None

tuning_progress = {
    'n_estimators': [],
    'max_depth': [],
    'val_f1': [],
    'train_f1': [],
    'val_acc': [],
    'train_acc': []
}

# Iterate over all parameter combinations
for n_estimators in param_grid['n_estimators']:
    for max_depth in param_grid['max_depth']:
        model_path = f"{MODEL_DIR}/rf_tree_{n_estimators}_depth_{max_depth}.joblib"
        if os.path.exists(model_path):
            logger.info(f"[+] Model with n_estimators={n_estimators}, max_depth={max_depth} already trained. Skipping...")
            continue
        start_time = datetime.datetime.now()
        logger.info(f"[+] Training Random Forest with n_estimators={n_estimators}, max_depth={max_depth}")
        rf = RandomForestClassifier(n_estimators=n_estimators, max_depth=max_depth, random_state=42, n_jobs=6)
        rf.fit(X_train, y_train)
        logger.info(f"[+] Training time: {datetime.datetime.now() - start_time}")
        
        # Validation
        all_y_pred, all_y_val = model_eval.split_test_set_into_batches(X_val, y_val, 2**12, model=rf)
        val_acc = accuracy_score(all_y_val, all_y_pred)
        val_f1 = f1_score(all_y_val, all_y_pred, average='weighted')
        logger.info(f"[+] Validation accuracy: {val_acc}, F1-Score: {val_f1}")
        
        # Training
        all_y_pred, all_y_val = model_eval.split_test_set_into_batches(X_train, y_train, 2**12, model=rf)
        train_acc = accuracy_score(all_y_val, all_y_pred)
        train_f1 = f1_score(all_y_val, all_y_pred, average='weighted')
        logger.info(f"[+] Training accuracy: {train_acc}, F1-Score: {train_f1}")
        
        # Save the progress
        tuning_progress['n_estimators'].append(n_estimators)
        tuning_progress['max_depth'].append(max_depth)
        tuning_progress['val_f1'].append(val_f1)
        tuning_progress['train_f1'].append(train_f1)
        tuning_progress['val_acc'].append(val_acc)
        tuning_progress['train_acc'].append(train_acc)
        
        # Check overfitting and save the model
        if val_f1 > best_f1_score:
            best_f1_score = val_f1
            best_params = {"n_estimators": n_estimators, "max_depth": max_depth}
            best_model = rf
            model_save_path = f"{MODEL_DIR}/rf_best_model.joblib"
            logger.info(f"[+] Best model with hyperparameters: {best_params}")
        dump(rf, f"{model_path}", compress=3)
logger.info(f"[+] Best parameters found: {best_params}")
logger.info(f"[+] Best validation F1-Score: {best_f1_score}")

# Save the best model
model_save_path = f"{MODEL_DIR}/rf_best_model.joblib"
dump(best_model, model_save_path, compress=3)
logger.info(f"[+] Best Random Forest model saved at: {model_save_path}")

# Save the tuning progress
tuning_progress_df = pd.DataFrame(tuning_progress)
tuning_progress_df.to_csv(f"{MODEL_DIR}/tuning_progress.csv", index=False)