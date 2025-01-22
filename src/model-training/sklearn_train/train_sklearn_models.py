import sys
PROJECT_PATH = "/home/mrcong/Code/mylab-nilm-files/nilm-physical-features"
sys.path.append("/home/mrcong/Code/mylab-nilm-files/nilm-physical-features/src/common")

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
from nilm_dao import get_vndale2_data, get_vndale1_data

# Load data
parser = argparse.ArgumentParser()
parser.add_argument("--rms_window_size", type=int, default=1800, help="Size of the RMS window")
args = parser.parse_args()

if args.rms_window_size <= 0:
    raise ValueError("rms_window_size must be a positive integer")

window_size = args.rms_window_size
train_df = get_vndale1_data("train", window_size)
# Change this to the features and hyperparameters you want to train
feature_set = [
    ['Irms', 'P', 'MeanPF', 'S', 'Q'],
]
hyperparameters_set = [
    {
        "rf":{
            "n_estimators": 15, "max_depth": 30,
            "random_state": 42, "n_jobs": 1, "verbose": 0
        },
        "xgb": {
            "n_estimators": 30, "max_depth": 30,
            "objective": "multi:softmax", "tree_method": "hist", 
            "num_class": 128, "gamma": 0.1, "subsample": 0.8, "verbosity": 0
        }
    } 
]

# Set up logging
logger = logging.getLogger()
date_time = datetime.datetime.now().strftime("%Y-%m-%d")
MODEL_DIR = f"{PROJECT_PATH}/results/models/VNDALE1/window_{window_size}"
if not os.path.exists(MODEL_DIR):
    os.makedirs(MODEL_DIR)
log_file_path = f"{MODEL_DIR}/{date_time}_sklearn_train.log"
logging.basicConfig(level=logging.INFO, 
                    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
                    handlers=[
                        logging.FileHandler(log_file_path),
                        logging.StreamHandler(sys.stdout)
                    ])

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
    logger.info(f"Data shape: {X_train.shape}, {y_train.shape}")
    _X_train = X_train
    _y_train = y_train
    
    # Reduce data
    _X_train, _, _y_train, _ = train_test_split(X_train, y_train, test_size=0.9, random_state=42, stratify=y_train)
    # X_train, y_train = _X_train, _y_train
    logger.info(f"After reducing: {_X_train.shape}, {_y_train.shape}")
    
    # Clear memory
    if len(features) == 1:
        train_df = None
    # Training Random Forest model
    rf_start_time = datetime.datetime.now()
    rf_params = hyperparameters["rf"]
    save_path = f"{model_save_path}/rf_{features}.joblib"
    if os.path.exists(save_path):
        logger.info(f"[+] Random Forest model already exists at: {save_path}")
    else:
        logger.info(f"[+] Training Random Forest with {features} with hyperparameters: {rf_params} and {_X_train.shape}!")
        model = RandomForestClassifier(**rf_params)
        model.fit(_X_train, _y_train)
        logger.info(f"[+] Random Forest training time: {datetime.datetime.now() - rf_start_time}")
        dump(model, save_path, compress=3)
        print(f"Model saved to {save_path}")
        logger.info(f"[+] Random Forest model saved at: {save_path}")

    # Training XGBoost model
    # xgb_params = hyperparameters["xgb"]
    # xgb_model_name = f"xgb_{features}.joblib"
    # save_path = f"{model_save_path}/{xgb_model_name}.joblib"
    # if os.path.exists(save_path):
    #     logger.info(f"[+] XGBoost model already exists at: {model_save_path}/xgb_{features}.joblib")
    # else:
    #     logger.info(f"[+] Training XGBoost model with {features}!")
    #     model = XGBClassifier(**xgb_params)
    #     model.fit(X_train, y_train)
    #     dump(model, save_path, compress=3)
    #     model = None
    #     logger.info(f"[+] XGBoost model saved at: {save_path}")