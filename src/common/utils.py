import numpy as np
import pandas as pd
import ast
from tqdm import tqdm
import os
import logging
from datetime import datetime

RESULT_DIR="<please specify the result directory>"
DATA_DIR="<please specify the data directory>"
SRC_DIR="<please specify the source directory>"
SEED=42

def count_labels(labels, isPercentage=False):
    unique_labels, counts = np.unique(labels, return_counts=True)
    if isPercentage:
        label_percentages = (counts/len(labels)) * 100
        label_percentages = dict(zip(unique_labels, label_percentages))
        return label_percentages
    label_counts = dict(zip(unique_labels, counts))
    return label_counts

def sort_filenames(filenames, index):
    """
    Sorts a list of filenames in ascending order based on the numeric part of the filename.

    Args:
        filenames (list): A list of filenames (e.g., ["fridge_1568.xlsx", "fridge_2773.xlsx", ...])

    Returns:
        list: Sorted list of filenames
    """
    def extract_numeric_part(filename):
        # Extract the numeric part from the filename (e.g., "fridge_1568.xlsx" -> 1568)
        return int(filename.split("_")[index].split(".")[0])
    # Sort the filenames based on the numeric part
    sorted_filenames = sorted(filenames, key=extract_numeric_part)
    return sorted_filenames
    
def extract_features(filename):
    # Assuming the filename is in the format: prefix_['feature1', 'feature2', ...].extension,
    # we extract the substring between the first underscore and the first period.
    start_idx = filename.find("_")
    end_idx = filename.find(".", start_idx)
    if start_idx == -1 or end_idx == -1:
        raise ValueError("Filename format is incorrect")
    features_str = filename[start_idx+1:end_idx]
    # Convert the string representation of the list into an actual list
    if features_str == "[Irms]":
        return ["Irms"]
    elif features_str == "[P]":
        return ["P"]
    features = ast.literal_eval(features_str)
    return features

def extract_model_name(filename):
    return filename.split("/")[-1].split(".")[0]

def create_n_gram_data(X, y, n_grams):
    """
    Creates n-gram data from the input data.

    Args:
        X (numpy.ndarray): Input data
        y (numpy.ndarray): Target labels
        window_size (int): Window size for creating n-gram data

    Returns:
        tuple: Tuple containing the n-gram data and labels
    """
    X_out, y_out = [], []
    for i in tqdm(range(n_grams, len(X)), desc="Creating n-gram data"):
        window_X = X[i-n_grams:i].flatten()
        window_y = y[i-n_grams:i]
        if np.all(window_y == window_y[0]):
            X_out.append(window_X)
            y_out.append(window_y[0])
    return np.array(X_out), np.array(y_out)

def setup_logger(save_path, model_name):
    # Create logger
    logger = logging.getLogger(model_name)
    logger.setLevel(logging.INFO)
    
    # Create formatters
    console_formatter = logging.Formatter('%(levelname)s - %(message)s')
    file_formatter = logging.Formatter('%(asctime)s - %(name)s - %(levelname)s - %(message)s')
    
    # Create console handler
    console_handler = logging.StreamHandler()
    console_handler.setLevel(logging.INFO)
    console_handler.setFormatter(console_formatter)
    
    # Create file handler
    os.makedirs(save_path, exist_ok=True)
    log_file = os.path.join(save_path, f'{datetime.now().strftime("%Y%m%d")}_training_{model_name}.log')
    file_handler = logging.FileHandler(log_file)
    file_handler.setLevel(logging.INFO)
    file_handler.setFormatter(file_formatter)
    
    # Add handlers to logger
    logger.addHandler(console_handler)
    logger.addHandler(file_handler)
    
    return logger

def process_combination_str(comb_str):
    combination = comb_str.replace("]", "").replace("[", "").split(", ")
    combination = ",".join([str(int(item)) for item in combination])
    return combination