import sys
PROJECT_PATH = "/home/mrcong/Code/mylab-nilm-files/nilm-physical-features"
sys.path.append(f"{PROJECT_PATH}/src/common")

import os
import numpy as np
import pandas as pd
import polars as pl
import torch
from torch.utils.data import DataLoader, Dataset
from tqdm import tqdm
from torchinfo import summary
from sequence_model import LSTMModel, RNNModel, GRUModel
import torch.nn as nn
from utils import process_combination_str
from sklearn.preprocessing import LabelEncoder, StandardScaler
from joblib import load, dump
# Directory where the dataset is stored
data_base_dir = "/opt/nilm-shared-data/nilm_device_detection/ICTA2024_dataset"

# Load label encoder
label_encoder = LabelEncoder()
label_encoder.classes_ = np.load(f"{data_base_dir}/label_encoder/classes.npy", allow_pickle=True)
standard_scaler = load(f"{PROJECT_PATH}/results/scaler/standard_scaler.joblib")

# Dataset information
data_info = f"{data_base_dir}/data_information/data_information.xlsx"
data_info_df = pd.read_excel(data_info)

class TimeSeriesDataset(Dataset):
    def __init__(self, series, window_size, label):
        self.series = series
        self.window_size = window_size
        self.label = label

    def __len__(self):
        return len(self.series) - self.window_size

    def __getitem__(self, idx):
        x = self.series[idx:idx+self.window_size]
        y = self.label
        return torch.tensor(x, dtype=torch.float32), torch.tensor(y, dtype=torch.float32)

def get_RMS_data(select_row, data_type, normalized) -> pl.DataFrame:
    data_paths = [data_type + "_" + data_info_df.iloc[select_row][f"data{i}"].replace(".xlsx", ".csv") for i in range(1, 5)]
    dfs = []
    combination = process_combination_str(data_info_df.iloc[select_row]["combination"])
    label = label_encoder.transform([combination])[0]
    for path in data_paths:
        file_path = f"{data_base_dir}/time_series_data/window_1800/{path}"
        if os.path.isfile(file_path):
            df = pl.read_csv(file_path)
            if normalized:
                labels = df.select("Label")
                times = df.select("Time")
                df = df.select(["In", "Un", "PF_n", "Irms", "Urms", "MeanPF", "P", "Q", "S"])
                df = standard_scaler.transform(df)
                df = pl.DataFrame(df, schema=["In", "Un", "PF_n", "Irms", "Urms", "MeanPF", "P", "Q", "S"])
                df = df.hstack(labels)
                df = df.hstack(times)
                df = df.select(["Time", "In", "Un", "PF_n", "Irms", "Urms", "MeanPF", "P", "Q", "S", "Label"])
        else:
            df = None    
        dfs.append(df)
    return dfs, label

def create_timeseries_dataset(window_size, batch_size, features, data_type, num_workers=12, normalized=False):
    # Start loading data
    combined_dataset = None
    for select_row in tqdm(range(len(data_info_df)), desc=f"Loading {data_type} data"):
        dfs, label = get_RMS_data(select_row, data_type, normalized=normalized)
        for df in dfs:
            if df is None:
                continue
            series = df.select(features).to_numpy()
            dataset = TimeSeriesDataset(series, window_size, label)
            combined_dataset = dataset if combined_dataset is None else combined_dataset + dataset
    dataloader = DataLoader(combined_dataset, batch_size=batch_size, num_workers=num_workers, 
                        pin_memory=True, shuffle=True, drop_last=True, persistent_workers=True)
    return dataloader
            