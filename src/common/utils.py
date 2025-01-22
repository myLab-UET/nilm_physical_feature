import numpy as np
import pandas as pd
import ast
from tqdm import tqdm
import os
import logging
from datetime import datetime

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
    # Extract the features part from the filename (e.g., "rf_[Irms, P, ...]_" -> [Irms, P, ...])
    return ast.literal_eval(filename.split("_")[1].split(".")[0])

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