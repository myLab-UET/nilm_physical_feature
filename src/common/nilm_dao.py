import pandas as pd
import polars as pl
import numpy as np
from sklearn.preprocessing import LabelEncoder
from sklearn.preprocessing import StandardScaler
import os
from tqdm import tqdm

VNDALE2_PATH = "/opt/nilm-shared-data/nilm_device_detection/VNDALE_v2"
VNDALE1_PATH = "/opt/nilm-shared-data/nilm_device_detection/VNDALE_v1"

def get_label_encoder(dataset)->LabelEncoder:
    if dataset == "vndale1":
        label_encoder_path = f"{VNDALE1_PATH}/label_encoder/labels.npy"
    elif dataset == "vndale2":
        label_encoder_path = f"{VNDALE2_PATH}/label_encoder/labels.npy"
    label_encoder = LabelEncoder()
    label_encoder.classes_ = np.load(label_encoder_path, allow_pickle=True)
    return label_encoder

def get_vndale2_data(data_type, rms_window_size) -> pl.DataFrame:
    data_dir = f"{VNDALE2_PATH}/RMS_window_rand/window_{rms_window_size}/{data_type}"
    all_data_df = None
    for csv_file in tqdm(os.listdir(data_dir), desc=f"Getting {data_type} data - window size: {rms_window_size}"):
        if csv_file.endswith(".csv"):
            data_df = pl.read_csv(f"{data_dir}/{csv_file}")
            if all_data_df is None:
                all_data_df = data_df
            else:
                all_data_df = pl.concat([all_data_df, data_df], how="vertical")
    return all_data_df

def get_vndale1_data(data_type, rms_window_size, is_norm=False) -> pl.DataFrame:
    # Get the data
    path = f"/opt/nilm-shared-data/nilm_device_detection/VNDALE_v1/RMS_data/window_{rms_window_size}/{data_type}"
    all_data_df = None
    for csv_file in tqdm(os.listdir(path), desc=f"Getting {data_type} data - window size: {rms_window_size}"):
        if csv_file.endswith(".csv"):
            data_df = pl.read_csv(f"{path}/{csv_file}")
            if all_data_df is None:
                all_data_df = data_df
            else:
                all_data_df = pl.concat([all_data_df, data_df], how="vertical")
    if is_norm:
        features = all_data_df.select(['In', 'Un', 'PF_n', "Irms", "Urms", "MeanPF", 'P', 'Q', 'S']).to_numpy()
        scaler = StandardScaler()
        scaled_features = scaler.fit_transform(features)
        scaled_data = pl.DataFrame(scaled_features)
        scaled_data.columns = ['In', 'Un', 'PF_n', "Irms", "Urms", "MeanPF", 'P', 'Q', 'S']
        scaled_data = pl.concat([scaled_data, all_data_df.select(["Label"])], how="horizontal")
        scaled_data = pl.concat([scaled_data, all_data_df.select(["Time"])], how="horizontal")
        return scaled_data
    return all_data_df
