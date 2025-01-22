import os
import sys
PROJECT_PATH = "/home/mrcong/Code/mylab-nilm-files/nilm-physical-features"
sys.path.append(f"{PROJECT_PATH}/src/common")

import torch
import numpy as np
import polars as pl
import pandas as pd
from torch.utils.data import Dataset, DataLoader
from sklearn.metrics import classification_report, accuracy_score, f1_score
from sklearn.preprocessing import LabelEncoder
from tqdm import tqdm

from sequence_model import GRUModel, LSTMModel, RNNModel
from ts_utils import data_base_dir, create_timeseries_dataset
from utils import setup_logger, extract_features, extract_model_name
import argparse

# Parsing arguments
parser = argparse.ArgumentParser(description="Sequence model training.")
parser.add_argument("--file_name", type=str, required=True, help="Model type to train.")
parser.add_argument("--batch_size", type=int, default=512, help="Batch size for validation.")
parser.add_argument("--gpu", type=int, default=0, help="GPU device to use.")
parser.add_argument("--window_size", type=int, default=30, help="Window size for time series data.")
parser.add_argument("--hidden_size", type=int, default=30, help="Hidden size for RNN models.")
parser.add_argument("--num_layers", type=int, default=6, help="Number of layers for RNN models.")
parser.add_argument("--normalize", type=bool, default=False, help="Normalize the data.")
parser.add_argument("--data_type", type=str, default="val", help="Data type to evaluate.")
args = parser.parse_args()

# Set up arguments
file_name = args.file_name
batch_size = int(args.batch_size)
device = torch.device(f"cuda:{args.gpu}" if torch.cuda.is_available() else "cpu")
window_size = int(args.window_size)
hidden_size = int(args.hidden_size)
num_layers = int(args.num_layers)
features = extract_features(file_name)
model_name = extract_model_name(file_name)
normalize = bool(args.normalize)
data_type = args.data_type
if data_type not in ["train", "val", "test"]:
    raise ValueError(f"Invalid data type: {data_type}")

# Main function
def main():
    # Path to the project directory
    model_path = f"{PROJECT_PATH}/results/models/VNDALE/window_1800/ts_model/{file_name}"
    logger = setup_logger(f"{PROJECT_PATH}/results/models/VNDALE/window_1800/ts_model/eval_logs", f"{file_name}_eval")
    logger.info(f"Evaluating model: {model_path}")
    logger.info(f"Data type: {data_type}")

    # Initialize and load model
    model = None
    if "LSTM" in model_name:
        model = LSTMModel(
            input_size=len(features),
            hidden_size=hidden_size,
            num_layers=num_layers,
            output_size=128
        ).to(device)
    elif "RNN" in model_name:
        model = RNNModel(
            input_size=len(features),
            hidden_size=hidden_size,
            num_layers=num_layers,
            output_size=128
        ).to(device)
    elif "GRU" in model_name:
        model = GRUModel(
            input_size=len(features),
            hidden_size=hidden_size,
            num_layers=num_layers,
            output_size=128
        ).to(device)
    else:
        raise ValueError(f"Invalid model name: {model_name}")
    model.load_state_dict(torch.load(model_path, weights_only=True))
    model.eval()
    
    # Example: load a test CSV file. Adjust path as needed.
    test_loader = create_timeseries_dataset(window_size, batch_size, features, data_type)
    # Perform inference
    all_targets, all_preds = [], []
    with torch.no_grad():
        for inputs, targets in tqdm(test_loader, desc="Testing"):
            inputs = inputs.to(device)
            outputs = model(inputs)
            preds = torch.argmax(outputs, dim=1).cpu().numpy()
            all_preds.extend(preds)
            all_targets.extend(targets.numpy())

    # Print results
    logger.info("Evaluation Results:")
    logger.info("Classification Report:")
    logger.info(f"Accuracy: {accuracy_score(all_targets, all_preds)}")
    logger.info(f"F1-Score: {f1_score(all_targets, all_preds, average='macro')}")
    classes = np.load(f"{data_base_dir}/label_encoder/classes.npy", allow_pickle=True)
    logger.info(classification_report(all_targets, all_preds, target_names=classes))

if __name__ == "__main__":
    main()