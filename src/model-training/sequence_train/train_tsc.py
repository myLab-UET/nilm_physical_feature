import os
import argparse
import numpy as np
import yaml
import torch

from common.utils import RESULT_DIR, SEED
from data_handler import get_tsc_train_dataset, preprocess_data
from clf_wrapper import ClassifierWrapper

# Set seeds
np.random.seed(SEED)
torch.manual_seed(SEED)
if torch.cuda.is_available():
    torch.cuda.manual_seed(SEED)

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description='Time Series Backdoor Attack (PyTorch)')
    parser.add_argument('--gpu', type=str, default='0', help='GPU device ID to use (e.g., 0, 1, or 0,1)')
    parser.add_argument("--dataset_name", type=str, default='iawe', help="Dataset name")
    parser.add_argument("--clf_name", type=str, help="Classifier name", choices=['fcn', 'resnet', 'transformer', 'hf_transformer', 'lstm'])
    parser.add_argument('--data_ratio', type=float, default=1.0, help='Ratio of data to use for training')
    parser.add_argument('--num_workers', type=int, default=4, help='Number of data loading workers')

    args = parser.parse_args()
    
    # Load config
    with open(f"training_tsc.yaml", 'r') as f:
        training_configs = yaml.safe_load(f)
        
    # Set visible GPU device
    os.environ['CUDA_VISIBLE_DEVICES'] = args.gpu
    print(f"Using GPU: {args.gpu}")
    print(f"PyTorch device: {torch.cuda.get_device_name(0) if torch.cuda.is_available() else 'CPU'}")

    # Other arguments
    dataset_name = args.dataset_name
    clf_name = args.clf_name
    data_ratio = args.data_ratio
    training_config = training_configs[clf_name][dataset_name]
    
    print(f"[+] Parameters: dataset_name={dataset_name}, classifier={clf_name}, data_ratio={data_ratio}")
    print(f"[+] Training configuration: {training_config}")
        
    # Import the dataset
    print(f"Loading dataset: {dataset_name}")
    x_train, y_train, x_test, y_test = get_tsc_train_dataset(
        dataset_name=dataset_name,
        data_ratio=data_ratio
    )
    print(f"Data shapes: {x_train.shape}, {y_train.shape}, {x_test.shape}, {y_test.shape}")
    
    # Prepare the data
    x_train, y_train, x_test, y_test, enc = preprocess_data(
        x_train=x_train,
        y_train=y_train,
        x_test=x_test,
        y_test=y_test
    )
    
    print(f"Preprocessed Data shapes (N, C, L): {x_train.shape}, {y_train.shape}, {x_test.shape}, {y_test.shape}")

    # Create the classifier
    output_directory = os.path.join(RESULT_DIR, dataset_name, clf_name)
    if not os.path.exists(output_directory):
        os.makedirs(output_directory)
    print(f"[+] Output directory: {output_directory}")
    
    nb_classes = enc.categories_[0].shape[0]
    print(f"[+] Number of classes: {nb_classes}")
    
    # Training the classifier
    classifier = ClassifierWrapper(
        training_config=training_config,
        clf_name=clf_name,
        input_shape=x_train.shape[1:], # (C, L)
        nb_classes=nb_classes,
        output_directory=output_directory,
        verbose=True,
        num_workers=args.num_workers,
        pin_memory=False
    )
    
    classifier.train(x_train, y_train, x_test, y_test)
    print(f"[+] Classifier {clf_name} trained successfully.")