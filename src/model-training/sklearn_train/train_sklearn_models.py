import sys
PROJECT_PATH = "<your_path>"
sys.path.append("<your_path>/src/common")

from sklearn.ensemble import RandomForestClassifier
from xgboost import XGBClassifier
from joblib import dump, load
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder
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
parser.add_argument("--train_size", type=float, default=1, help="Train size")
args = parser.parse_args()
print(f"Arguments: {args}")
if args.window < 1:
    raise ValueError("Window size must be greater than 1")
train_size = args.train_size
no_class = 128
if train_size > 1 or train_size < 0:
    raise ValueError("Data proportion must be in range [0, 1]")
if args.data == "vndale1":
    train_df = get_vndale1_data("train", args.window, args.is_norm)
    no_class = 128
elif args.data == "rae":
    train_df = get_rae_data("train", args.is_norm)
elif args.data == "iawe":
    train_df = get_iawe_data("train", args.is_norm)
    no_class = 10
else:
    raise ValueError("Invalid dataset name")

# Change this to the features and hyperparameters you want to train
feature_set = [
    ['Irms'],
    ['P'],
    ['Irms', 'P'],
    ['Irms', 'P', 'MeanPF'],
    ['Irms', 'P', 'MeanPF', 'S'],
    ['Irms', 'P', 'MeanPF', 'S', 'Q'],
]

hyperparameters_set = [
    {
        "rf":{
            "n_estimators": 15, "max_depth": 30,
            "random_state": 42, "n_jobs": 3, "verbose": 0
        },
        "xgb": {
            "n_estimators": 15, "max_depth": 30,
            "objective": "multi:softmax", "tree_method": "hist", 
            "num_class": no_class, "gamma": 0.1, "subsample": 0.8, "verbosity": 0,
            "min_child_weight": 3
        }
    },
    {
        "rf":{
            "n_estimators": 15, "max_depth": 30,
            "random_state": 42, "n_jobs": 3, "verbose": 0
        },
        "xgb": {
            "n_estimators": 15, "max_depth": 30, 
            "objective": "multi:softmax", "tree_method": "hist", 
            "num_class": no_class, "gamma": 0.1, "subsample": 0.8, "verbosity": 0,
            "min_child_weight": 3
        }
    },
    {
        "rf":{
            "n_estimators": 15, "max_depth": 30,
            "random_state": 42, "n_jobs": 3, "verbose": 0
        },
        "xgb": {
            "n_estimators": 15, "max_depth": 30, 
            "objective": "multi:softmax", "tree_method": "hist", 
            "num_class": no_class, "gamma": 0.1, "subsample": 0.8, "verbosity": 0
        }
    },
    {
        "rf":{
            "n_estimators": 15, "max_depth": 30,
            "random_state": 42, "n_jobs": 3, "verbose": 0
        },
        "xgb": {
            "n_estimators": 15, "max_depth": 30,
            "objective": "multi:softmax", "tree_method": "hist", 
            "num_class": no_class, "gamma": 0.1, "subsample": 0.8, "verbosity": 0
        }
    },
    {
        "rf":{
            "n_estimators": 15, "max_depth": 30,
            "random_state": 42, "n_jobs": 3, "verbose": 0
        },
        "xgb": {
            "n_estimators": 15, "max_depth": 30, 
            "objective": "multi:softmax", "tree_method": "hist", 
            "num_class": no_class, "gamma": 0.1, "subsample": 0.8, "verbosity": 0
        }
    },
    {
        "rf":{
            "n_estimators": 15, "max_depth": 30,
            "random_state": 42, "n_jobs": 3, "verbose": 0
        },
        "xgb": {
            "n_estimators": 15, "max_depth": 30,
            "objective": "multi:softmax", "tree_method": "hist", 
            "num_class": no_class, "gamma": 0.1, "subsample": 0.8, "verbosity": 0
        }
    }
]

# Set up logging
date_time = datetime.datetime.now().strftime("%Y-%m-%d")
if args.data == "vndale1":
    MODEL_DIR = f"{PROJECT_PATH}/results/models/VNDALE1/window_{args.window}"
elif args.data == "rae":
    MODEL_DIR = f"{PROJECT_PATH}/results/models/rae"
elif args.data == "iawe":
    MODEL_DIR = f"{PROJECT_PATH}/results/models/iawe"
else:
    raise ValueError("Invalid dataset name")

# Create directory if not exists
if not os.path.exists(MODEL_DIR):
    os.makedirs(MODEL_DIR)
if not os.path.exists(f"{MODEL_DIR}/logs"):
    os.makedirs(f"{MODEL_DIR}/logs")

# Set up logging
log_file_path = f"{MODEL_DIR}/logs/{date_time}_sklearn_train.log"
logging.basicConfig(level=logging.INFO, 
                    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
                    handlers=[
                        logging.FileHandler(log_file_path),
                        logging.StreamHandler(sys.stdout)
                    ])
logger = logging.getLogger()
logger.info(f"[+] Configuration: {args}")
logger.info(f"[+] Train data head: {train_df.head()}")
for features, hyperparameters in zip(feature_set, hyperparameters_set):
    logging.info(f"[+] Training model with features: {features}")
    logging.info(f"[+] Hyperparameters: {hyperparameters}")
    # Load label encoders
    model_save_path = f"{MODEL_DIR}/{len(features)}_comb"
    if not os.path.exists(model_save_path):
        os.makedirs(model_save_path)
    
    # Load data
    X_train = train_df.select(features).to_numpy()
    y_train = train_df["Label"].to_numpy()
    
    # Reduce data
    if train_size < 1:
        X_train, _, y_train, _ = train_test_split(X_train, y_train, train_size=train_size, random_state=42, stratify=y_train)    
    logger.info(f"Data shape: {X_train.shape}, {y_train.shape}")
    
    # Clear memory
    if len(feature_set) == 1:
        train_df = None
    
    # Training Random Forest model
    if os.path.exists(f"{model_save_path}/rf_{features}.joblib"):
        logger.info(f"[+] Model {features} already exists, skipping...")
    else:
        rf_start_time = datetime.datetime.now()
        rf_params = hyperparameters["rf"]
        save_path = f"{model_save_path}/rf_{features}.joblib"
        logger.info(f"[+] Training Random Forest with {features} with hyperparameters: {rf_params} and {X_train.shape}!")
        model = RandomForestClassifier(**rf_params)
        model.fit(X_train, y_train)
        logger.info(f"[+] Random Forest training time: {datetime.datetime.now() - rf_start_time}")
        dump(model, save_path, compress=3)
        print(f"Model saved to {save_path}")
        logger.info(f"[+] Random Forest model saved at: {save_path}")

    # Training XGBoost model
    if os.path.exists(f"{model_save_path}/xgb_{features}.joblib"):
        logger.info(f"[+] Model {features} already exists, skipping...")
        model = None
        continue
    xgb_start_time = datetime.datetime.now()
    xgb_params = hyperparameters["xgb"]
    xgb_model_name = f"xgb_{features}"
    save_path = f"{model_save_path}/{xgb_model_name}.joblib"
    logger.info(f"[+] Training XGBoost model with {features}!")
    model = XGBClassifier(**xgb_params)
    model.fit(X_train, y_train)
    logger.info(f"[+] XGBoost training time: {datetime.datetime.now() - xgb_start_time}")
    dump(model, save_path, compress=3)
    logger.info(f"[+] XGBoost model saved at: {save_path}")
    
    # Clear memory
    model = None