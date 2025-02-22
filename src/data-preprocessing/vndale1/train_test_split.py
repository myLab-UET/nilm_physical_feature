import pandas as pd
import numpy as np
import os
from tqdm import tqdm
from sklearn.model_selection import train_test_split
import logging
from sklearn.preprocessing import LabelEncoder
import argparse

# Configure logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s', handlers=[
    logging.FileHandler("/home/mrcong/Code/mylab-nilm-files/nilm-physical-features/results/data_preprocessing/vndale1_train_test_split.log"),
    logging.StreamHandler()
])

# Argument parser
parser = argparse.ArgumentParser(description="Train Test Split for VNDALE v1")
parser.add_argument("--window_size", type=int, default=1800, help="Window size for RMS calculation")
args = parser.parse_args()
window_size = args.window_size

def cal_rms(inputs):
    return np.sqrt(np.mean(np.power(inputs, 2)))

def cal_all_rms(instant_current, instant_voltage, pfs):
    Irms = cal_rms(instant_current)
    Urms = cal_rms(instant_voltage)
    avg_pf = np.mean(pfs) / 100
    P = Urms * Irms * avg_pf
    S = Urms * Irms
    Q = Urms * Irms * (1 - avg_pf ** 2)
    return Irms, Urms, avg_pf, P, Q, S

def get_features(df):
    instant_current = df["currentWaveform"].values
    instant_voltage = df["voltageWaveform"].values
    pfs = df["powerFactor"].values
    return cal_all_rms(instant_current, instant_voltage, pfs)

def create_directory_if_exist(directory_path):
    if not os.path.exists(directory_path):
        os.makedirs(directory_path)

def cal_RMS_features(window_df, window_size, label):
    process_datas = []
    for i in range(window_size, len(window_df)):
        window = window_df.iloc[i - window_size:i]
        time = window["Time"].values[0]
        In = window["currentWaveform"].values[0]
        Un = window["voltageWaveform"].values[0]
        pfN = window["powerFactor"].values[0]/100
        Irms, Urms, avg_pf, P, Q, S = get_features(window)
        process_datas.append([time, In, Un, pfN, Irms, Urms, avg_pf, P, Q, S, label])
    return pd.DataFrame(process_datas, columns=["Time", "In", "Un", "PF_n", "Irms", "Urms", "MeanPF", "P", "Q", "S", "Label"])

def process_combination_str(comb_str):
    combination = comb_str.replace("]", "").replace("[", "").split(", ")
    combination = ",".join([str(int(item)) for item in combination])
    return combination

def get_data(data_info_df, select_row, data_index):
    # Get data
    base_path = "/opt/nilm-shared-data/nilm_device_detection/VNDALE_v1/fix_raw_data"
    combination = data_info_df.iloc[select_row]["combination"].replace("]", "").replace("[", "").split(", ")
    combination = ",".join([str(int(item)) for item in combination])
    data_path = data_info_df.iloc[select_row][f"data{data_index}"].replace(".xlsx", ".csv")
    file_path = f"{base_path}/{data_path}"
    
    # Check if file exist
    if os.path.isfile(file_path):
        df = pd.read_csv(file_path)
        if df.shape[0] == 0:
            return None, None, None, None
        # Split data into windows of size 2 * window_size
        windows = []
        for start in range(0, len(df) - 2 * window_size + 1, window_size):
            window = df.iloc[start:start + 2 * window_size]
            windows.append(window)
        # Split windows into train, val, test
        train_windows = []
        test_windows = []
        val_windows = []
        counts = 0
        for _, window in enumerate(windows):
            if counts<=7:
                train_windows.append(window)
            if counts==8:
                val_windows.append(window)
            if counts==9:
                test_windows.append(window)
            counts+=1
            if counts == 10:
                counts = 0
        return train_windows, val_windows, test_windows, combination
    # Return None if file not found
    return None, None, None, None

def main():
    # Load data information
    base_path = "/opt/nilm-shared-data/nilm_device_detection/VNDALE_v1"
    save_dir = f"{base_path}/RMS_data/window_{window_size}"
    create_directory_if_exist(f"{save_dir}/train")
    create_directory_if_exist(f"{save_dir}/val")
    create_directory_if_exist(f"{save_dir}/test")
    
    # Label encoder
    label_encoder = LabelEncoder()
    label_encoder.classes_ = np.load(f"{base_path}/label_encoder/labels.npy", allow_pickle=True)
    
    # Process data
    data_info = pd.read_excel(f"{base_path}/data_information/data_information.xlsx", sheet_name="data_info")
    for select_row in tqdm(range(len(data_info)), desc="Processing data"):
        comb_train_data, comb_val_data, comb_test_data = [], [], []
        for data_index in range(1, 5):
            train_windows, val_windows, test_windows, combination = get_data(data_info, select_row, data_index)
            if train_windows is None:
                logging.info(f"[-] Combination - file {data_index}: {combination} - Not found")
                continue
            # Transform data
            label = label_encoder.transform([combination])[0]
            logging.info(f"[+] Combination {combination} - {label} - file {data_index} windows:, {len(train_windows)} - {len(val_windows)} - {len(test_windows)}")
            for train_window in train_windows:
                train_transform_data = cal_RMS_features(train_window, window_size, label)
                comb_train_data.append(train_transform_data)
            for val_window in val_windows:
                val_transform_data = cal_RMS_features(val_window, window_size, label)
                comb_val_data.append(val_transform_data)
            for test_window in test_windows:
                test_transform_data = cal_RMS_features(test_window, window_size, label)
                comb_test_data.append(test_transform_data)
        # Save data
        comb_train_data = pd.concat(comb_train_data)
        comb_val_data = pd.concat(comb_val_data)
        comb_test_data = pd.concat(comb_test_data)
        # Logging
        comb_train_data.to_csv(f"{save_dir}/train/{label}_train.csv", index=False)
        comb_val_data.to_csv(f"{save_dir}/val/{label}_val.csv", index=False)
        comb_test_data.to_csv(f"{save_dir}/test/{label}_test.csv", index=False)
        logging.info("[+] Save data to csv")
    logging.info("[+] Done!")
    
if __name__ == "__main__":
    main()
