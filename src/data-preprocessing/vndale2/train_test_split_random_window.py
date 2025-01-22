import pandas as pd
import numpy as np
import warnings
warnings.filterwarnings('ignore')
import os
from tqdm import tqdm
from sklearn.model_selection import train_test_split
import logging
from sklearn.preprocessing import LabelEncoder
import argparse

# Configure logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s', handlers=[
    logging.FileHandler("/home/mrcong/Code/mylab-nilm-files/mylab-nilm-device-detection/results/logs/train_test_split_vndale_v2.log"),
    logging.StreamHandler()
])
# Argument parser
parser = argparse.ArgumentParser(description="Train Test Split for VNDALE v2")
parser.add_argument("--window_size", type=int, default=100, help="Window size for RMS calculation")
args = parser.parse_args()
window_size = args.window_size
BASE_PATH = "/opt/nilm-shared-data/nilm_device_detection/VNDALE_v2"
RAW_DATA_PATH = f"{BASE_PATH}/fix_raw_data"

def cal_rms(inputs):
    return np.sqrt(np.mean(np.power(inputs, 2)))

def cal_all_rms(instant_current, instant_voltage, pfs):
    Irms = cal_rms(instant_current)
    Urms = cal_rms(instant_voltage)
    avg_pf = np.mean(pfs)
    P = Urms * Irms * avg_pf
    S = Urms * Irms
    Q = Urms * Irms * (1 - avg_pf ** 2)
    return Irms, Urms, avg_pf, P, Q, S

def get_rms_features(df):
    instant_current = df["In"].values
    instant_voltage = df["Un"].values
    pfs = df["Power Factor"].values/100
    return cal_all_rms(instant_current, instant_voltage, pfs)

def create_directory_if_exist(directory_path):
    if not os.path.exists(directory_path):
        os.makedirs(directory_path)
        
def cal_RMS_features(window_df, window_size, label):
    process_datas = []
    for i in range(window_size, len(window_df)):
        window = window_df.iloc[i - window_size:i]
        In = window["In"].values[0]
        Un = window["Un"].values[0]
        pfN = window["Power Factor"].values[0]/100
        Irms, Urms, avg_pf, P, Q, S = get_rms_features(window)
        process_datas.append([In, Un, pfN, Irms, Urms, avg_pf, P, Q, S, label])
    return pd.DataFrame(process_datas, columns=["In", "Un", "PF_n", "Irms", "Urms", "MeanPF", "P", "Q", "S", "Label"])

def process_combination_str(comb_str):
    combination = comb_str.replace("]", "").replace("[", "").split(", ")
    combination = ",".join([str(int(item)) for item in combination])
    return combination

def get_data(csv_file, label_encoder: LabelEncoder):
    device_comb_name = csv_file.replace(".csv", "")
    device_comb_id = label_encoder.transform([device_comb_name])[0]
    file_path = f"{BASE_PATH}/fix_raw_data/{csv_file}"
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
        return train_windows, val_windows, test_windows, device_comb_id

def main():
    # Create directory
    save_dir = f"{BASE_PATH}/RMS_window_rand/window_{window_size}"
    create_directory_if_exist(f"{save_dir}/train")
    create_directory_if_exist(f"{save_dir}/val")
    create_directory_if_exist(f"{save_dir}/test")
    
    # Label encoder
    label_encoder = LabelEncoder()
    label_encoder.classes_ = np.load(f"{BASE_PATH}/label_encoder/labels.npy", allow_pickle=True)
    
    # Process data
    csv_files = [f for f in os.listdir(RAW_DATA_PATH) if f.endswith(".csv")]
    for file_name in tqdm(csv_files, desc="Processing data"):
        train_windows, val_windows, test_windows, device_comb_id = get_data(file_name, label_encoder)
        print(f"[+] Combination: {device_comb_id}, {len(train_windows)} - {len(val_windows)} - {len(test_windows)}")
        all_train_data, all_val_data, all_test_data = [], [], []
        for train_window in train_windows:
            train_transform_data = cal_RMS_features(train_window, window_size, device_comb_id)
            all_train_data.append(train_transform_data)
        for val_window in val_windows:
            val_transform_data = cal_RMS_features(val_window, window_size, device_comb_id)
            all_val_data.append(val_transform_data)
        for test_window in test_windows:
            test_transform_data = cal_RMS_features(test_window, window_size, device_comb_id)
            all_test_data.append(test_transform_data)
        all_train_data = pd.concat(all_train_data)
        all_val_data = pd.concat(all_val_data)
        all_test_data = pd.concat(all_test_data)
        total_data = all_train_data.shape[0] + all_val_data.shape[0] + all_test_data.shape[0]
        logging.info(f"Data percentage train: {all_train_data.shape[0]/total_data}, Val: {all_val_data.shape[0]/total_data}, Test: {all_test_data.shape[0]/total_data}")
        all_train_data.to_csv(f"{save_dir}/train/{file_name}", index=False)
        all_val_data.to_csv(f"{save_dir}/val/{file_name}", index=False)
        all_test_data.to_csv(f"{save_dir}/test/{file_name}", index=False)
        logging.info(f"Data saved successfully")

if __name__ == "__main__":
    main()