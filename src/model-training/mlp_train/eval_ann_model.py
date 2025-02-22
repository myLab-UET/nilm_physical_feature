import sys
PROJECT_PATH = "/home/mrcong/Code/mylab-nilm-files/nilm-physical-features"
sys.path.append(f"{PROJECT_PATH}/src/common")

import numpy as np
import polars as pl
# from sklearn.preprocessing import LabelEncoder
from sklearn.metrics import classification_report
import os
from tqdm import tqdm

# for DL modeling
import torch
# import torch.nn as nn
# import torch.nn.functional as F
from torch.utils.data import DataLoader,TensorDataset
from sklearn.preprocessing import LabelEncoder
from ann_models import AnnRMSModel, MLPModel2
import logging
from model_eval import ModelEvaluation
from utils import extract_features
from nilm_dao import *
from sklearn.metrics import classification_report, accuracy_score, f1_score, precision_score, recall_score
import argparse
from sklearn.model_selection import train_test_split

# Configs
parser = argparse.ArgumentParser(description="Evaluate ANN model")
parser.add_argument("--dataset", type=str, required=True, help="Name of the dataset")
parser.add_argument("--data_type", type=str, required=True, default="test", help="Type of data to evaluate on")
parser.add_argument("--model_name", type=str, required=True, help="Name of the model file")
parser.add_argument("--window_size", type=int, default=1800, help="Window size for data processing")
parser.add_argument("--is_norm", type=lambda x: (str(x).lower() in ['true', '1']), default=True, help="Normalization flag")
parser.add_argument("--eval_batch_size", type=int, default=512, help="Batch size for evaluation")
parser.add_argument("--dropout_rate", type=float, default=0.0, help="Dropout rate for the model")
parser.add_argument("--is_bn", type=lambda x: (str(x).lower() in ['true', '1']), default=False, help="Batch normalization flag")
parser.add_argument("--size", type=float, default=1, help="Size of the evaluation model")

args = parser.parse_args()
eval_batch_size = args.eval_batch_size
window_size = args.window_size
model_name = args.model_name
is_norm = args.is_norm
is_bn = args.is_bn
dropout_rate = args.dropout_rate
data_type = args.data_type
dataset_name = args.dataset
size = args.size
print("[+] Evaluation arguments: ", args)

# Setup other things
model_eval          = ModelEvaluation()
features            = extract_features(model_name)

# Model path
if dataset_name == "vndale1":
    model_path = f"{PROJECT_PATH}/results/models/VNDALE1/window_{window_size}/{len(features)}_comb"
elif dataset_name == "iawe":
    model_path = f"{PROJECT_PATH}/results/models/iawe/{len(features)}_comb"
elif dataset_name == "rae":
    model_path = f"{PROJECT_PATH}/results/models/rae/{len(features)}_comb"
else:
    raise ValueError("Invalid dataset name!")

state_dict_path     = f"{model_path}/{model_name}"
log_path            = f"{model_path}/model_evaluation"
log_file            = os.path.join(log_path, f"{model_name}_evaluation.log")
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s - %(levelname)s - %(message)s",
    handlers=[
        logging.FileHandler(log_file),
        logging.StreamHandler(sys.stdout)
    ]
)

def evaluate_test_dataset(device, model:AnnRMSModel, test_loader:DataLoader):
    # Initialize lists to store evaluation metrics
    y_true_list = []
    y_pred_list = []
    
    # Set the model to evaluation mode
    model.eval()
    with torch.no_grad():
        for X, y in tqdm(iterable=test_loader, desc="Predicting"):
            # transfer data to GPU
            X = X.to(device)
            y = y.to(device)

            # Forward pass
            yHat = model(X)
            y_pred = torch.argmax(yHat, axis=1)

            # Calculate evaluation metrics
            y_true_list.extend(y.cpu().numpy())
            y_pred_list.extend(y_pred.cpu().numpy())
    y_true_list = np.array(y_true_list)
    y_pred_list = np.array(y_pred_list)
    return y_true_list, y_pred_list
    
#Starting Main():
device = torch.device("cuda:0")
print(f"[+] Evaluation on {device} with model path: {state_dict_path}")

#Loading the data
print("[+] Loading data:")
if dataset_name == "vndale1":
    validation_df = get_vndale1_data(data_type, window_size, is_norm)
elif dataset_name == "iawe":
    validation_df = get_iawe_data(data_type, is_norm)
elif dataset_name == "rae":
    validation_df = get_rae_data(data_type, is_norm)
else:
    raise ValueError("Invalid dataset name!")
print(validation_df.head())
label_encoder = get_label_encoder(dataset_name)

#Transform X_train and X_val to numpy
print("[+] Transform into numpy:")
X_train = validation_df.select(features).to_numpy()
y_train = validation_df["Label"].to_numpy()
if size < 1.0:
    X_train, _, y_train, _ = train_test_split(X_train, y_train, test_size=size, random_state=42, stratify=y_train)
# no_classes = len(np.unique(y_train))
no_classes = len(label_encoder.classes_)
#Change X, y to tensors, and setup
print("[+] Dataset information:")
X_val = torch.tensor(X_train).float()
y_val = torch.tensor(y_train)
print(f"[+] Validation set: {X_val.shape}, y_val: {y_val.shape}")

#Load the model
ann_rms = AnnRMSModel(input_dim=len(features), output_dim=no_classes, is_bn=is_bn, dropout=dropout_rate)
ann_rms.to(device)

# Load the previous model
if(os.path.isfile(state_dict_path)):
    logging.info(f"[+] Model load in : {state_dict_path}")
    ann_rms.load_state_dict(torch.load(state_dict_path))
    logging.info(f"[+] Model loaded!")
else:
    logging.error("[+] Model path not found!")
    exit()

#Train and validation dataloaders
val_dataDataset = TensorDataset(X_val, y_val)
val_loader = DataLoader(val_dataDataset, batch_size=eval_batch_size, shuffle=True, drop_last=False, num_workers=12, pin_memory=True, persistent_workers=True)

#Evaluate model
all_y_true, all_y_pred = evaluate_test_dataset(device=device, model=ann_rms, test_loader=val_loader)
val_accuracy = accuracy_score(y_true=all_y_true, y_pred=all_y_pred)
val_f1_macro = f1_score(y_true=all_y_true, y_pred=all_y_pred, average='weighted')
val_precision = precision_score(y_true=all_y_true, y_pred=all_y_pred, average='weighted')
val_recall = recall_score(y_true=all_y_true, y_pred=all_y_pred, average='weighted')
logging.info(f"[+] ANN results: {model_name} with {data_type} data")
logging.info(f"[+] {data_type} accuracy: {val_accuracy}, F1-Score: {val_f1_macro}, Precision: {val_precision}, Recall: {val_recall}")

report = classification_report(all_y_true, all_y_pred, target_names=label_encoder.classes_)
logging.info(f"{report}")

# Report most error classes
report_df = model_eval.report_most_error_each_class(y_true=all_y_true, y_pred=all_y_pred, label_encoder=label_encoder)
report_df.to_csv(f"{log_path}/{model_name}_error_report_{data_type}.csv", index=False)
