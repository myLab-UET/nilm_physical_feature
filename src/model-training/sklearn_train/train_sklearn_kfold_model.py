import sys
PROJECT_PATH = "/home/mrcong/Code/mylab-nilm-files/nilm-physical-features"
sys.path.append("/home/mrcong/Code/mylab-nilm-files/nilm-physical-features/src/common")

from sklearn.ensemble import RandomForestClassifier
from xgboost import XGBClassifier
from sklearn.model_selection import KFold
from sklearn.metrics import accuracy_score, f1_score
from joblib import dump
import numpy as np
import pandas as pd
import polars as pl
import datetime
import os
import logging
import argparse
from nilm_dao import *

# Load data
parser = argparse.ArgumentParser()
parser.add_argument("--data", type=str, required=True, default="vndale1", help="Dataset to train")
parser.add_argument("--is_norm", required=True, type=lambda x: (str(x).lower() in ['true', '1']), default=False, help="Normalization flag")
parser.add_argument("--window", type=int, default=1800, help="Size of the RMS window, only for VNDALE1 and VNDALE2")
parser.add_argument("--n_folds", type=int, default=5, help="Number of folds for cross-validation")
args = parser.parse_args()

print(f"Arguments: {args}")
if args.window < 1:
    raise ValueError("Window size must be greater than 1")
if args.n_folds < 2:
    raise ValueError("Number of folds must be at least 2")

# Load dataset
if args.data == "vndale1":
    data_df = get_vndale1_data("train", args.window, args.is_norm)
    no_class = 128
elif args.data == "rae":
    data_df = get_rae_data("train", args.is_norm)
    no_class = len(data_df["Label"].unique())
elif args.data == "iawe":
    data_df = get_iawe_data("train", args.is_norm)
    no_class = len(data_df["Label"].unique())
else:
    raise ValueError("Invalid dataset name")

# Feature sets and hyperparameters
feature_combs = [
    # ['Irms'],
    # ['P'],
    # ['Irms', 'P'],
    # ['Irms', 'P', 'MeanPF'],
    # ['Irms', 'P', 'MeanPF', 'S'],
    ['Irms', 'P', 'MeanPF', 'S', 'Q'],
]

hyperparameters_set = {
    "rf": {
        "n_estimators": 15, "max_depth": 25,
        "random_state": 42, "n_jobs": 2, "verbose": 0
    },
    "xgb": {
        "n_estimators": 15, "max_depth": 20,
        "objective": "multi:softmax", "tree_method": "hist",
        "num_class": no_class, "gamma": 0.1, "subsample": 0.8, "verbosity": 0,
        "min_child_weight": 3
    }
}

# Set up logging
date_time = datetime.datetime.now().strftime("%Y-%m-%d")
if args.data == "vndale1":
    MODEL_DIR = f"{PROJECT_PATH}/results/models/VNDALE1/window_{args.window}/kfold"
elif args.data == "rae":
    MODEL_DIR = f"{PROJECT_PATH}/results/models/rae/kfold"
elif args.data == "iawe":
    MODEL_DIR = f"{PROJECT_PATH}/results/models/iawe/kfold"
else:
    raise ValueError("Invalid dataset name")

# Create directory if not exists
if not os.path.exists(MODEL_DIR):
    os.makedirs(MODEL_DIR)
if not os.path.exists(f"{MODEL_DIR}/logs"):
    os.makedirs(f"{MODEL_DIR}/logs")

# Set up logging
log_file_path = f"{MODEL_DIR}/logs/{date_time}_sklearn_kfold_train.log"
logging.basicConfig(level=logging.INFO,
                    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
                    handlers=[
                        logging.FileHandler(log_file_path),
                        logging.StreamHandler(sys.stdout)
                    ])
logger = logging.getLogger()
logger.info(f"[+] Configuration: {args}")
logger.info(f"[+] Data head: {data_df.head()}")

# Prepare data
for feature_set in feature_combs:
    print(f"[+] Feature set: {feature_set}")
    X = data_df.select(feature_set).to_numpy()
    y = data_df["Label"].to_numpy()

    # K-Fold Cross-Validation
    kf = KFold(n_splits=args.n_folds, shuffle=True, random_state=42)
    fold_metrics = []

    for fold_idx, (train_idx, val_idx) in enumerate(kf.split(X)):
        logger.info(f"[+] Starting fold {fold_idx + 1}/{args.n_folds}")
        
        # Split data
        X_train, X_val = X[train_idx], X[val_idx]
        y_train, y_val = y[train_idx], y[val_idx]
        logger.info(f"[+] Fold {fold_idx + 1} - Train shape: {X_train.shape}, {y_train.shape}")
        logger.info(f"[+] Fold {fold_idx + 1} - Validation shape: {X_val.shape}, {y_val.shape}")
        
        # Train Random Forest
        rf_model = RandomForestClassifier(**hyperparameters_set["rf"])
        rf_model.fit(X_train, y_train)
        rf_preds = rf_model.predict(X_val)
        rf_accuracy = accuracy_score(y_val, rf_preds)
        rf_f1 = f1_score(y_val, rf_preds, average='weighted')
        logger.info(f"[+] Fold {fold_idx + 1} - Random Forest Accuracy: {rf_accuracy:.4f}, F1 Score: {rf_f1:.4f}")
        # rf_save_path = f"{MODEL_DIR}/rf_{feature_set}_fold_{fold_idx + 1}.joblib"
        # dump(rf_model, rf_save_path)
        # logger.info(f"[+] Random Forest model saved at: {rf_save_path}")
        rf_model = None  # Clear model to free memory
        
        # Train XGBoost
        xgb_model = XGBClassifier(**hyperparameters_set["xgb"])
        xgb_model.fit(X_train, y_train)
        xgb_preds = xgb_model.predict(X_val)
        xgb_accuracy = accuracy_score(y_val, xgb_preds)
        xgb_f1 = f1_score(y_val, xgb_preds, average='weighted')
        logger.info(f"[+] Fold {fold_idx + 1} - XGBoost Accuracy: {xgb_accuracy:.4f}, F1 Score: {xgb_f1:.4f}")
        # xgb_save_path = f"{MODEL_DIR}/xgb_{feature_set}_fold_{fold_idx + 1}.joblib"
        # dump(xgb_model, xgb_save_path)
        # logger.info(f"[+] XGBoost model saved at: {xgb_save_path}")
        xgb_model = None
        
        # Store metrics
        fold_metrics.append({
            "fold": fold_idx + 1,
            "feature_set": feature_set,
            "rf_accuracy": rf_accuracy,
            "rf_f1": rf_f1,
            "xgb_accuracy": xgb_accuracy,
            "xgb_f1": xgb_f1
        })
        logger.info(f"[+] Fold {fold_idx + 1} metrics: {fold_metrics[-1]}")

# Save metrics
metrics_df = pd.DataFrame(fold_metrics)
metrics_save_path = f"{MODEL_DIR}/kfold_metrics.csv"
metrics_df.to_csv(metrics_save_path, index=False)
logger.info(f"[+] K-Fold metrics saved at: {metrics_save_path}")