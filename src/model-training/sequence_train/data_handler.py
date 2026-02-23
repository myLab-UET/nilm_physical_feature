import os
import numpy as np
import torch
from torch.utils.data import Dataset, DataLoader
from sklearn.preprocessing import OneHotEncoder
from sklearn.model_selection import train_test_split
from common.utils import SEED, DATA_DIR

def select_stratified_by_sample(x_data, y_data, no_samples, random_state=SEED):
    """
    Select a stratified random subset from the data based on max samples.
    """
    if no_samples >= len(x_data):
        return x_data, y_data
    
    subset_ratio = no_samples / len(x_data)
    y_indices = np.argmax(y_data, axis=1)
    
    x_subset, _, y_subset, _ = train_test_split(
        x_data, y_data,
        train_size=subset_ratio,
        stratify=y_indices,
        random_state=random_state
    )
    
    return x_subset, y_subset

def select_stratified_by_ratio(x_data, y_data, data_ratio, random_state=SEED):
    """
    Select a stratified random subset from the data based on a ratio.
    """
    if not (0 < data_ratio <= 1):
        raise ValueError("data_ratio must be in the range (0, 1]")
    
    no_samples = int(len(x_data) * data_ratio)
    return select_stratified_by_sample(x_data, y_data, no_samples, random_state)

def _load_split_arrays(root_dir_dataset):
    """
    Load train/test arrays from a dataset directory.
    Expected naming: X_train/y_train/X_test/y_test.
    """
    required_files = {
        'x_train': 'X_train.npy',
        'y_train': 'y_train.npy',
        'x_test': 'X_test.npy',
        'y_test': 'y_test.npy',
    }

    if not all(os.path.exists(os.path.join(root_dir_dataset, file_name)) for file_name in required_files.values()):
        raise FileNotFoundError(
            f"Missing dataset files in {root_dir_dataset}. Expected either "
            "[X_train.npy, y_train.npy, X_test.npy, y_test.npy]."
        )

    x_train = np.load(os.path.join(root_dir_dataset, required_files['x_train']))
    y_train = np.load(os.path.join(root_dir_dataset, required_files['y_train']))
    x_test = np.load(os.path.join(root_dir_dataset, required_files['x_test']))
    y_test = np.load(os.path.join(root_dir_dataset, required_files['y_test']))
    return x_train, y_train, x_test, y_test


def get_tsc_train_dataset(dataset_name, data_ratio=1.0):
    """
    Get the training dataset for time series classification.
    """
    dataset_name_normalized = dataset_name.lower()
    if dataset_name_normalized in ["iawe", "vndale1", "rae"]:
        root_dir_dataset = os.path.join(DATA_DIR, dataset_name_normalized)
        x_train, y_train, x_test, y_test = _load_split_arrays(root_dir_dataset)

        if dataset_name_normalized == 'iawe':
            # iAWE dataset has 5 channels, we only use 4 channels (0, 1, 3, 4)
            x_train = x_train[:, :, [0, 1, 3, 4]]
            x_test = x_test[:, :, [0, 1, 3, 4]]

        # Apply stratified data reduction if ratio < 1.0
        if data_ratio < 1.0:
            # Stratify training set
            unique_classes = np.unique(y_train)
            train_indices = []
            
            for cls in unique_classes:
                cls_indices = np.where(y_train == cls)[0]
                cls_samples = int(len(cls_indices) * data_ratio)
                if cls_samples > 0:
                    selected_indices = np.random.choice(cls_indices, cls_samples, replace=False)
                    train_indices.extend(selected_indices)
            
            train_indices = np.array(train_indices)
            x_train = x_train[train_indices]
            y_train = y_train[train_indices]
            
            # Stratify test set
            test_indices = []
            unique_test_classes = np.unique(y_test)
            
            for cls in unique_test_classes:
                cls_indices = np.where(y_test == cls)[0]
                cls_samples = int(len(cls_indices) * data_ratio)
                if cls_samples > 0:
                    selected_indices = np.random.choice(cls_indices, cls_samples, replace=False)
                    test_indices.extend(selected_indices)
            
            test_indices = np.array(test_indices)
            x_test = x_test[test_indices]
            y_test = y_test[test_indices]
            
            print(f"Dataset {dataset_name} reduced to {data_ratio*100:.1f}% with stratified sampling")

        # Cast types
        x_train = x_train.astype(np.float32)
        y_train = y_train.astype(np.float32)
        x_test = x_test.astype(np.float32)
        y_test = y_test.astype(np.float32)

    else:
        raise ValueError("dataset_name must be one of: 'iawe', 'vndale1', 'rae'")

    return x_train, y_train, x_test, y_test

def preprocess_data(x_train, y_train, x_test, y_test):
    """
    One-hot encode the labels and reshape the data if univariate.
    """
    # Transform the labels from integers to one hot vectors
    enc = OneHotEncoder(categories='auto')
    print("y_train shape before encoding:", y_train.shape)
    print("y_test shape before encoding:", y_test.shape)
    enc.fit(np.concatenate((y_train, y_test), axis=0).reshape(-1, 1))
    y_train = enc.transform(y_train.reshape(-1, 1)).toarray()
    y_test = enc.transform(y_test.reshape(-1, 1)).toarray()

    if len(x_train.shape) == 2:  # if univariate
        # add a dimension to make it multivariate with one dimension 
        x_train = x_train.reshape((x_train.shape[0], x_train.shape[1], 1))
        x_test = x_test.reshape((x_test.shape[0], x_test.shape[1], 1))

    # Cast types to float32
    x_train = x_train.astype(np.float32)
    x_test = x_test.astype(np.float32)
    y_train = y_train.astype(np.float32)
    y_test = y_test.astype(np.float32)

    return x_train, y_train, x_test, y_test, enc

class TSCDataset(Dataset):
    def __init__(self, x, y):
        self.x = torch.from_numpy(x)
        self.y = torch.from_numpy(y)

    def __len__(self):
        return len(self.x)

    def __getitem__(self, idx):  # ty:ignore[invalid-method-override]
        return self.x[idx], self.y[idx]

def create_data_loader(x_data, y_data, batch_size=64, shuffle=True, num_workers=0, pin_memory=False):
    dataset = TSCDataset(x_data, y_data)
    return DataLoader(dataset, batch_size=batch_size, shuffle=shuffle, num_workers=num_workers, pin_memory=pin_memory)

# Alias for compatibility
create_dataset = create_data_loader
