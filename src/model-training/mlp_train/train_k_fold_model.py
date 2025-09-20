import numpy as np
import pandas as pd
import polars as pl
from sklearn.model_selection import KFold
from sklearn.metrics import f1_score, accuracy_score
import time
import sys
PROJECT_PATH = "<your_path>"
sys.path.append(f"{PROJECT_PATH}/src/common")
from nilm_dao import *

# for DL modeling
from torch.utils.tensorboard import SummaryWriter
import torch
import torch.nn as nn
from torch.utils.data import DataLoader, TensorDataset
import torch.optim.lr_scheduler as lr_scheduler
from ann_models import AnnRMSModel
import datetime
import logging
import os
import argparse
from torchinfo import summary

# Configs
parser = argparse.ArgumentParser(description="Train MLP model with K-fold cross validation")
parser.add_argument("--data", required=True, type=str, default="vndale1", help="Dataset to train")
parser.add_argument("--is_norm", required=True, type=lambda x: (str(x).lower() in ['true', '1']), default=True, help="Normalization flag")
parser.add_argument("--numepochs", type=int, default=80, help="Number of epochs")
parser.add_argument("--scheduler_end_factor", type=float, default=0.1, help="Scheduler end factor")
parser.add_argument("--learning_rate", type=float, default=1.2e-3, help="Learning rate")
parser.add_argument("--weight_decay", type=float, default=1e-5, help="Weight decay")
parser.add_argument("--dropout_rate", type=float, default=0, help="Dropout rate")
parser.add_argument("--batch_size", type=int, default=512, help="Batch size")
parser.add_argument("--window_size", type=int, default=1800, help="Window size (Only for VNDALE1 dataset)")
parser.add_argument("--is_bn", type=lambda x: (str(x).lower() in ['true', '1']), default=False, help="Batch normalization flag")
parser.add_argument("--lr_sloth_factor", type=float, default=1, help="Learning rate sloth factor")
parser.add_argument("--gpu", type=int, default=0, help="GPU device")
parser.add_argument("--n_folds", type=int, default=4, help="Number of folds for cross-validation")

# Parse arguments
args = parser.parse_args()
dataset                  = args.data
numepochs                = args.numepochs
scheduler_end_factor     = args.scheduler_end_factor
learning_rate            = args.learning_rate
weight_decay             = args.weight_decay
dropout_rate             = args.dropout_rate
batch_size               = args.batch_size
window_size              = args.window_size
is_norm                  = args.is_norm or False
is_bn                    = args.is_bn or False
lr_sloth_factor          = args.lr_sloth_factor
n_folds                  = args.n_folds
# Learning rate scheduler
scheduler_iter           = int(lr_sloth_factor*numepochs)
device                   = torch.device(f"cuda:{args.gpu}" if torch.cuda.is_available() else "cpu")
# Model saving directory
name = "norm" if is_norm else "nonorm"
name = f"{name}_bn" if is_bn else name
name = f"{name}_kfold_{n_folds}"
name = f"{name}_epochs_{numepochs}"
name = f"{name}_lr_{learning_rate}"
date_time = datetime.datetime.now().strftime("%Y-%m-%d")

# Function to train the model
def trainTheModel(ann_rms, writer: SummaryWriter, train_loader:DataLoader, val_loader:DataLoader, 
                 learning_rate, numepochs:int, fold_idx:int, features:list, model_saving_dir:str):
    # loss function and optimizer
    lossfun = nn.CrossEntropyLoss()
    optimizer = torch.optim.Adam(ann_rms.parameters(), lr=learning_rate, weight_decay=weight_decay)
    scheduler = lr_scheduler.LinearLR(optimizer, start_factor=1.0, end_factor=scheduler_end_factor, total_iters=scheduler_iter)
    min_val_loss = np.inf
    best_accuracy = 0
    best_f1 = 0
    
    # loop over epochs
    for epochi in range(0, numepochs):
        # switch on training mode
        ann_rms.train()
        start_time = time.time()
        
        # loop over training data batches
        batchLoss = []
        for X,y in train_loader:
            # transfer data to GPU
            X = X.to(device)
            y = y.to(device)
            # forward pass and loss
            yHat = ann_rms(X)
            loss = lossfun(yHat,y)
            
            # backprop
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

            # loss from this batch
            batchLoss.append(loss.item())
        # end of batch loop and make scheduler reduce
        scheduler.step()
        writer.add_scalar("Learning rate", optimizer.param_groups[0]["lr"], epochi)
            
        # Validation metrics
        ann_rms.eval()
        valBatchLoss = []
        all_preds = []
        all_true = []
        
        with torch.no_grad():
            for X_val, y_val in val_loader:
                # transfer data to GPU
                X_val = X_val.to(device)
                y_val = y_val.to(device)          
                
                # Compute predictions and loss
                yHat = ann_rms(X_val)
                val_loss = lossfun(yHat, y_val).item()
                valBatchLoss.append(val_loss)
                
                # Get predictions for metrics
                _, predicted = torch.max(yHat.data, 1)
                
                # Collect predictions and true labels for metrics calculation
                all_preds.extend(predicted.cpu().numpy())
                all_true.extend(y_val.cpu().numpy())
            # end of validation loop...
        
        # Calculate metrics
        accuracy = accuracy_score(all_true, all_preds)
        f1 = f1_score(all_true, all_preds, average='weighted')
                
        # Write data to tensorboard    
        batch_train_loss = np.mean(batchLoss)
        writer.add_scalar("Loss/Training", batch_train_loss, epochi)
        batch_val_loss = np.mean(valBatchLoss)
        writer.add_scalar("Loss/Validation", batch_val_loss, epochi)
        writer.add_scalar("Metrics/Accuracy", accuracy, epochi)
        writer.add_scalar("Metrics/F1_Score", f1, epochi)
        
        logger.info(f"[+] Fold {fold_idx} - Epoch {epochi}/{numepochs} - "
                   f"Training loss: {batch_train_loss:.6f} - "
                   f"Validation loss: {batch_val_loss:.6f} - "
                   f"Accuracy: {accuracy:.4f} - "
                   f"F1 Score: {f1:.4f} - "
                   f"lr: {optimizer.param_groups[0]['lr']:.6f} - "
                   f"Training time: {time.time() - start_time:.2f}s")
        
        # Saving model's state if validation loss is less than previous
        if batch_val_loss < min_val_loss:
            min_val_loss = batch_val_loss
            best_accuracy = accuracy
            best_f1 = f1
            model_output_name = f"mlp_{features}_fold_{fold_idx}.pt"
            output_state_dict_path = f"{model_saving_dir}/{model_output_name}"
            logging.info(f"[+] Saving model's state for fold {fold_idx}!")
            try:
                torch.save(ann_rms.state_dict(), f"{output_state_dict_path}")
                logging.info(f"[+] Model saved to {output_state_dict_path}")      
            except Exception as e:
                logging.error(e)
                torch.save(ann_rms.state_dict(), f"{model_output_name}")
    
    # function output
    return ann_rms, min_val_loss, best_accuracy, best_f1

# Loading data
print(f"[+] Configs: {args}")
print("[+] Loading data")
if dataset == "vndale1":
    train_df = get_vndale1_data("train", window_size, is_norm=is_norm)
    val_df = get_vndale1_data("val", window_size, is_norm=is_norm)
    test_df = get_vndale1_data("test", window_size, is_norm=is_norm)
    # Combine all datasets for K-fold validation
    print("[+] Combining datasets for K-fold cross-validation")
    combined_df = pl.concat([train_df, val_df, test_df])
elif dataset == "iawe":
    train_df = get_iawe_data("train", is_norm=is_norm)
    test_df = get_iawe_data("test", is_norm=is_norm)
    # Combine training and test datasets for K-fold validation
    print("[+] Combining datasets for K-fold cross-validation")
    combined_df = pl.concat([train_df, test_df])
elif dataset == "rae":
    train_df = get_rae_data("train", is_norm=is_norm)
    test_df = get_rae_data("test", is_norm=is_norm)
    # Combine training and test datasets for K-fold validation
    print("[+] Combining datasets for K-fold cross-validation")
    combined_df = pl.concat([train_df, test_df])
else:
    raise ValueError("Invalid dataset name")

print(f"[+] Combined df shape: {combined_df.shape}")
  
# Features combs
feature_combs = [
    ['Irms', 'P', 'MeanPF', 'S', 'Q'],
    ['Irms', 'P', 'MeanPF', 'S'],
    ['Irms', 'P', 'MeanPF'],
    ['Irms', 'P'],
    ['P'],
    ['Irms'],
]

# Train model with each feature comb
for features in feature_combs:
    # Model saving directory
    no_features = len(features)
    if dataset == "vndale1":
        model_saving_dir = f"{PROJECT_PATH}/results/models/VNDALE1/window_{window_size}/{no_features}_comb/kfold_{n_folds}"
        log_file_path = f"{PROJECT_PATH}/results/models/VNDALE1/window_{window_size}/{no_features}_comb/logs/{date_time}_mlp_{name}_kfold.log"
        tb_log_dir = f"{PROJECT_PATH}/results/tensorboard/VNDALE1/MLP_{window_size}_kfold_{n_folds}"
    elif dataset == "iawe":
        model_saving_dir = f"{PROJECT_PATH}/results/models/iawe/{no_features}_comb/kfold_{n_folds}"
        log_file_path = f"{PROJECT_PATH}/results/models/iawe/{no_features}_comb/logs/{date_time}_mlp_{name}_kfold.log"
        tb_log_dir = f"{PROJECT_PATH}/results/tensorboard/iawe/MLP_kfold_{n_folds}"
    elif dataset == "rae":
        model_saving_dir = f"{PROJECT_PATH}/results/models/rae/{no_features}_comb/kfold_{n_folds}"
        log_file_path = f"{PROJECT_PATH}/results/models/rae/{no_features}_comb/logs/{date_time}_mlp_{name}_kfold.log"
        tb_log_dir = f"{PROJECT_PATH}/results/tensorboard/rae/MLP_kfold_{n_folds}"
    else:
        raise ValueError("[+] Invalid dataset name")

    # Create directory if not exists
    if not os.path.exists(model_saving_dir):
        os.makedirs(model_saving_dir)
    
    # Create logs directory path
    log_dir = os.path.dirname(log_file_path)
    if not os.path.exists(log_dir):
        os.makedirs(log_dir)
    
    if not os.path.exists(f"{model_saving_dir}/kfold_{n_folds}"):
        os.makedirs(f"{model_saving_dir}/kfold_{n_folds}")
    if not os.path.exists(tb_log_dir):
        os.makedirs(tb_log_dir)
    
    # Setup logger
    logger = logging.getLogger()
    date_time = datetime.datetime.now().strftime("%Y-%m-%d")
    logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(name)s - %(levelname)s - %(message)s', 
                        handlers=[logging.FileHandler(log_file_path), logging.StreamHandler(sys.stdout)])
    logger.info(f"[+] Training model with features: {features}")
    X = combined_df.select(features).to_numpy()
    y = combined_df["Label"].to_numpy()
    no_classes = len(np.unique(y))
    logger.info(f"[+] Combined data shape: {X.shape}, {y.shape}")
    logger.info(f"[+] Number of classes: {no_classes}")
    
    # K-fold cross validation
    kf = KFold(n_splits=n_folds, shuffle=True, random_state=42)
    fold_metrics = []
    
    for fold_idx, (train_idx, val_idx) in enumerate(kf.split(X)):
        logger.info(f"[+] Starting fold {fold_idx + 1}/{n_folds}")
        
        # Split data for this fold
        X_train, X_val = X[train_idx], X[val_idx]
        y_train, y_val = y[train_idx], y[val_idx]
        
        logger.info(f"[+] Fold {fold_idx + 1} - Train shape: {X_train.shape}, {y_train.shape}")
        logger.info(f"[+] Fold {fold_idx + 1} - Val shape: {X_val.shape}, {y_val.shape}")
        
        # Convert to tensors
        X_train_tensor = torch.tensor(X_train).float()
        y_train_tensor = torch.tensor(y_train)
        X_val_tensor = torch.tensor(X_val).float()
        y_val_tensor = torch.tensor(y_val)
        
        # Create DataLoader
        train_dataset = TensorDataset(X_train_tensor, y_train_tensor)
        val_dataset = TensorDataset(X_val_tensor, y_val_tensor)
        
        train_loader = DataLoader(train_dataset, 
                                batch_size=batch_size, 
                                pin_memory=True, num_workers=12, 
                                persistent_workers=True, 
                                shuffle=True, drop_last=True)
        val_loader = DataLoader(val_dataset, 
                                batch_size=batch_size, 
                                pin_memory=True, num_workers=12, 
                                persistent_workers=True, 
                                shuffle=True, drop_last=True)
        
        # Create model and transfer to GPU
        model = AnnRMSModel(input_dim=len(features), output_dim=no_classes, is_bn=is_bn, dropout=dropout_rate)
        summary(model, input_size=(batch_size, len(features)))
        model.to(device)
        
        # Weight initialization
        logger.info("[+] Weight initialization for new model!")
        for p in model.named_parameters():
            if 'weight' in p[0] and p[1].data.dim() >= 2:
                p[1].data = nn.init.kaiming_normal_(p[1].data, nonlinearity="relu")
        
        # Tensorboard writer for this fold
        writer = SummaryWriter(log_dir=f"{tb_log_dir}/MLP_{features}_fold_{fold_idx + 1}")
        
        # Train the model
        logger.info(f"[+] Start training model for fold {fold_idx + 1}!")
        trained_model, min_val_loss, best_accuracy, best_f1 = trainTheModel(
            ann_rms=model,
            writer=writer,
            train_loader=train_loader,
            val_loader=val_loader,
            learning_rate=learning_rate,
            numepochs=numepochs,
            fold_idx=fold_idx + 1,
            features=features,
            model_saving_dir=model_saving_dir
        )
        
        # Store fold metrics
        fold_metrics.append({
            "fold": fold_idx + 1,
            "min_val_loss": min_val_loss,
            "accuracy": best_accuracy,
            "f1_score": best_f1
        })
        
        # Close tensorboard writer
        writer.close()
        
        # Free memory
        del X_train_tensor, y_train_tensor, X_val_tensor, y_val_tensor
        del train_dataset, val_dataset, train_loader, val_loader
        del model, trained_model
        torch.cuda.empty_cache()
    
    # Calculate average metrics across all folds
    avg_val_loss = np.mean([m['min_val_loss'] for m in fold_metrics])
    avg_accuracy = np.mean([m['accuracy'] for m in fold_metrics]) 
    avg_f1 = np.mean([m['f1_score'] for m in fold_metrics])

    logger.info(f"[+] Feature combination: {features} - Average metrics across {n_folds} folds:")
    logger.info(f"    - Validation loss: {avg_val_loss:.6f}")
    logger.info(f"    - Accuracy: {avg_accuracy:.4f}")
    logger.info(f"    - F1 Score: {avg_f1:.4f}")
    
    # Save fold metrics to file
    metrics_df = pd.DataFrame(fold_metrics)
    metrics_df.to_csv(f"{model_saving_dir}/fold_metrics_{features}.csv", index=False)
    logger.info(f"[+] Fold metrics saved to {model_saving_dir}/fold_metrics_{features}.csv")
    
logger.info("[+] Finished training all feature combinations with K-fold cross-validation!")