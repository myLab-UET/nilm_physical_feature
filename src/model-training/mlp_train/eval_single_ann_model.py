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
from ann_models import AnnRMSModel
import logging
from model_eval import ModelEvaluation
from utils import extract_features
from nilm_dao import get_vndale1_data
# Configs
model_eval          = ModelEvaluation()
eval_batch_size     = 2**9
window_size         = 1800
model_dir           = f"{PROJECT_PATH}/results/models/VNDALE1/window_1800/mlp_model"
model_name          = "mlp_['Irms', 'P', 'MeanPF', 'S', 'Q'].pt"
features            = extract_features(model_name)
state_dict_path     = f"{model_dir}/{model_name}"
data_type           = "val"

# Setup logging
log_path = model_dir
log_file = os.path.join(log_path, "mlp_evaluation.log")
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
  
def evaluate_and_print_results(y_pred, y_true):
    print("[+] Evaluating results")
    accuracy = model_eval.cal_accuracy(y_true, y_pred)
    f1_macro = model_eval.cal_custom_f1_score_macro(y_true, y_pred)
    precision, recall = model_eval.cal_precision_recall_macro(y_true, y_pred)
    return accuracy, f1_macro, precision, recall
    
#Starting Main():
device = torch.device("cuda:0")
print(f"[+] Evaluation on {device} with model path: {state_dict_path}")

#Load the model
ann_rms = AnnRMSModel(features=features)
ann_rms.to(device)
# Load the previous model
if(os.path.isfile(state_dict_path)):
    logging.info(f"[+] Model load in : {state_dict_path}")
    ann_rms.load_state_dict(torch.load(state_dict_path))
    logging.info(f"[+] Model loaded!")
else:
    logging.error("[+] Model path not found!")
    exit()

#Loading the data
print("[+] Loading data:")
validation_df = get_vndale1_data(data_type, window_size)
print(validation_df.head())

#Transform X_train and X_val to numpy
print("[+] Transform into numpy:")

#Change X, y to tensors, and setup
print("[+] Dataset information:")
X_val = torch.tensor(validation_df.select(features).to_numpy()).float()
y_val = torch.tensor(validation_df["Label"].to_numpy())
print(f"[+] Validation set: {X_val.shape}, y_val: {y_val.shape}")

#Train and validation dataloaders
val_dataDataset = TensorDataset(X_val, y_val)
val_loader = DataLoader(val_dataDataset, batch_size=eval_batch_size, shuffle=True, drop_last=False)

#Evaluate model
all_y_true, all_y_pred = evaluate_test_dataset(device=device, model=ann_rms, test_loader=val_loader)
val_accuracy, val_f1_macro, val_precision, val_recall = evaluate_and_print_results(y_true=all_y_true, y_pred=all_y_pred)
report = classification_report(all_y_true, all_y_pred)
logging.info(f"[+] ANN results: {model_name} with {data_type} data")
logging.info(f"[+] Validation accuracy: {val_accuracy}, F1-Score: {val_f1_macro}, Precision: {val_precision}, Recall: {val_recall}")
logging.info(f"[+] Classification report: {report}")
# report_df = model_eval.report_most_error_each_class(y_true=all_y_true, y_pred=all_y_pred, label_encoder=label_encoder)
# report_df.to_csv(f"{model_dir}/{model_name}_error_report_{data_type}.csv", index=False)

# print(f"[+] ANN results: {model_name} with {data_type} data")
# print(f"[+] Validation accuracy: {val_accuracy}, F1-Score: {val_f1_macro}, Precision: {val_precision}, Recall: {val_recall}")
# print(f"[+] Classification report: {report}")