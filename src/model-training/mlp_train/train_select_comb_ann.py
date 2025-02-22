import numpy as np
import pandas as pd
import polars as pl
from sklearn.preprocessing import LabelEncoder
import time
import sys
PROJECT_PATH = "/home/mrcong/Code/mylab-nilm-files/nilm-physical-features"
sys.path.append(f"{PROJECT_PATH}/src/common")
from nilm_dao import *

# for DL modeling
from torch.utils.tensorboard import SummaryWriter
import torch
import torch.nn as nn
from torch.utils.data import DataLoader, TensorDataset
import torch.optim.lr_scheduler as lr_scheduler
from ann_models import AnnRMSModel, MLPModel2
import datetime
import logging
import os
from sklearn.model_selection import train_test_split
import argparse
from torchinfo import summary

# Configs
parser = argparse.ArgumentParser(description="Train MLP model with selected features")
parser.add_argument("--data", required=True, type=str, default="vndale1", help="Dataset to train")
parser.add_argument("--is_norm", required=True, type=lambda x: (str(x).lower() in ['true', '1']), default=True, help="Normalization flag")
parser.add_argument("--train_size", type=float, default=1, help="Training data proportion")
parser.add_argument("--numepochs", type=int, default=80, help="Number of epochs")
parser.add_argument("--scheduler_end_factor", type=float, default=0.1, help="Scheduler end factor")
parser.add_argument("--learning_rate", type=float, default=1.2e-3, help="Learning rate")
parser.add_argument("--weight_decay", type=float, default=1e-5, help="Weight decay")
parser.add_argument("--dropout_rate", type=float, default=0, help="Dropout rate")
parser.add_argument("--batch_size", type=int, default=512, help="Batch size")
parser.add_argument("--window_size", type=int, default=1800, help="Window size")
parser.add_argument("--is_bn", type=lambda x: (str(x).lower() in ['true', '1']), default=False, help="Batch normalization flag")
parser.add_argument("--lr_sloth_factor", type=float, default=1, help="Learning rate sloth factor")
parser.add_argument("--gpu", type=int, default=0, help="GPU device")

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
train_size               = args.train_size
if train_size > 1 or train_size < 0:
  raise ValueError("Data proportion must be in range [0, 1]")
# Learning rate scheduler
scheduler_iter           = int(lr_sloth_factor*numepochs)
device                   = torch.device(f"cuda:{args.gpu}" if torch.cuda.is_available() else "cpu")
# Model saving directory
name = "norm" if is_norm else "nonorm"
name = f"{name}_bn" if is_bn else name
name = f"{name}_train_size_{train_size}"
name = f"{name}_epochs_{numepochs}"
name = f"{name}_lr_{learning_rate}"
date_time = datetime.datetime.now().strftime("%Y-%m-%d")

# Function to train the model
def trainTheModel(ann_rms, writer: SummaryWriter, train_loader:DataLoader, val_loader:DataLoader, learning_rate, numepochs:int):
  # loss function and optimizer
  lossfun = nn.CrossEntropyLoss()
  optimizer = torch.optim.Adam(ann_rms.parameters(), lr=learning_rate, weight_decay=weight_decay)
  scheduler = lr_scheduler.LinearLR(optimizer, start_factor=1.0, end_factor=scheduler_end_factor, total_iters=scheduler_iter)
  min_val_loss = np.inf
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
      
    # Validation loss
    ann_rms.eval()
    valBatchLoss = []
    with torch.no_grad():
      for X_val, y_val in val_loader:
        # extract X_val, y_val from test dataloader, then transfer to GPU
        X_val = X_val.to(device)
        y_val = y_val.to(device)          
        # Compute loss
        yHat = ann_rms(X_val)
        val_loss = lossfun(yHat, y_val).item()
        valBatchLoss.append(val_loss)
      # end of validation loop...
        
    # Write data to tensorboard    
    batch_train_loss = np.mean(batchLoss)
    writer.add_scalar("Loss/Training", batch_train_loss, epochi)
    batch_val_loss = np.mean(valBatchLoss)
    writer.add_scalar("Loss/Validation", batch_val_loss, epochi)
    logger.info(f"[+] Epoch {epochi}/{numepochs} - Training loss: {batch_train_loss} - Validation loss: {batch_val_loss} - lr: {optimizer.param_groups[0]['lr']} - Training time: {time.time() - start_time}")
    
    #Saving model's state if validation loss is less than previous
    if batch_val_loss < min_val_loss:
      min_val_loss = batch_val_loss
      model_output_name = f"mlp_{features}.pt"
      output_state_dict_path = f"{model_saving_dir}/{model_output_name}"
      logging.info("[+] Saving model's state!")
      try:
        torch.save(ann_rms.state_dict(), f"{output_state_dict_path}")
        logging.info(f"[+] Model saved to {output_state_dict_path}")      
      except Exception as e:
        logging.error(e)
        torch.save(ann_rms.state_dict(), f"{model_output_name}")
  # function output
  return ann_rms

# Loading data
print(f"[+] Configs: {args}")
print("[+] Loading data")
if dataset == "vndale1":
  train_df = get_vndale1_data("train", window_size, is_norm=is_norm)
  validation_df = get_vndale1_data("val", window_size, is_norm=is_norm)
elif dataset == "iawe":
  train_df = get_iawe_data("train", is_norm=is_norm)
  validation_df = get_iawe_data("test", is_norm=is_norm)
elif dataset == "rae":
  train_df = get_rae_data("train", is_norm=is_norm)
  validation_df = get_rae_data("test", is_norm=is_norm)
else:
  raise ValueError("Invalid dataset name")

# Print data
print("[+] Training df")
print(train_df.head())
print("[+] Validation df")
print(validation_df.head())
  
# Features combs
feature_combs = [
  ['Irms', 'P', 'MeanPF', 'S', 'Q'],
  ['P'],
  ['Irms', 'P', 'MeanPF', 'S'],
  ['Irms', 'P', 'MeanPF'],
  ['Irms', 'P'],
  ['Irms'],
]

# Train model with each feature comb
for features in feature_combs:
  # Model saving directory
  no_features = len(features)
  if dataset == "vndale1":
    model_saving_dir         = f"{PROJECT_PATH}/results/models/VNDALE1/window_{window_size}/{no_features}_comb"
    log_file_path            = f"{PROJECT_PATH}/results/models/VNDALE1/window_{window_size}/{no_features}_comb/logs/{date_time}_mlp_{name}.log"
  elif dataset == "iawe":
    model_saving_dir         = f"{PROJECT_PATH}/results/models/iawe/{no_features}_comb"
    log_file_path            = f"{PROJECT_PATH}/results/models/iawe/{no_features}_comb/logs/{date_time}_mlp_{name}.log"
  elif dataset == "rae":
    model_saving_dir         = f"{PROJECT_PATH}/results/models/rae/{no_features}_comb"
    log_file_path            = f"{PROJECT_PATH}/results/models/rae/{no_features}_comb/logs/{date_time}_mlp_{name}.log"
  else:
    raise ValueError("[+] Invalid dataset name")

  # Create directory if not exists
  if not os.path.exists(model_saving_dir):
    os.makedirs(model_saving_dir)
  if not os.path.exists(f"{model_saving_dir}/logs"):
    os.makedirs(f"{model_saving_dir}/logs")
  
  # Setup logger
  logger = logging.getLogger()
  date_time = datetime.datetime.now().strftime("%Y-%m-%d")
  logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(name)s - %(levelname)s - %(message)s', 
                      handlers=[logging.FileHandler(log_file_path), logging.StreamHandler(sys.stdout)])
  
  # Change X, y to tensors, and setup
  logger.info(f"[+] Training model with features: {features}")
  X_val = validation_df.select(features).to_numpy()
  X_train = train_df.select(features).to_numpy()
  y_train = train_df["Label"].to_numpy()
  y_val = validation_df["Label"].to_numpy()
  no_classes = len(np.unique(y_train))
  # Select only a few data
  if train_size < 1:
    X_train, _, y_train, _ = train_test_split(X_train, y_train, train_size=train_size, random_state=42, stratify=y_train)
  logger.info(f"[+] Train shape: {X_train.shape}, {y_train.shape}")
  logger.info(f"[+] Val shape: {X_val.shape}, {y_val.shape}")
  
  # Convert to tensors
  X_val     = torch.tensor(X_val).float()
  X_train   = torch.tensor(X_train).float()
  y_train   = torch.tensor(y_train)
  y_val     = torch.tensor(y_val)
  
  logger.info(f"Training set: {X_train.shape}, y_train: {y_train.shape}")
  logger.info(f"Validation set: {X_val.shape}, y_val: {y_val.shape}")
  
  # Create DataLoader & Free memory
  train_dataDataset = TensorDataset(X_train, y_train)
  val_dataDataset   = TensorDataset(X_val, y_val)
  
  # Train and validation dataloaders
  train_loader = DataLoader(train_dataDataset, 
                            batch_size=batch_size, 
                            pin_memory=True, num_workers=12, 
                            persistent_workers=True, 
                            shuffle=True, drop_last=True)
  val_loader   = DataLoader(val_dataDataset, batch_size=batch_size, 
                            pin_memory=True, num_workers=12, 
                            persistent_workers=True, 
                            shuffle=True, drop_last=True)
  
  # Free memory
  X_train, y_train, X_val, y_val = None, None, None, None
  
  # Transfer model to GPU
  model = AnnRMSModel(input_dim=len(features), output_dim=no_classes, is_bn=is_bn, dropout=dropout_rate)
  summary(model, input_size=(batch_size, len(features)))
  model.to(device)
  
  # Model training
  logger.info("[+] Weight initialization for new model!")
  for p in model.named_parameters():
    if 'weight' in p[0] and p[1].data.dim() >= 2:
      p[1].data = nn.init.kaiming_normal_(p[1].data, nonlinearity="relu")
  logger.info("[+] Start training model!")
  
  # Tensorboard writer
  if dataset == "vndale1":
    tb_log_dir = f"{PROJECT_PATH}/results/tensorboard/VNDALE1/MLP_{window_size}"
  elif dataset == "iawe":
    tb_log_dir = f"{PROJECT_PATH}/results/tensorboard/iawe/MLP"
  elif dataset == "rae":
    tb_log_dir = f"{PROJECT_PATH}/results/tensorboard/rae/MLP"
  else:
    raise ValueError("[+] Invalid dataset name")
  
  writer = SummaryWriter(log_dir=f"{tb_log_dir}/MLP_{features}")
  model = trainTheModel(ann_rms=model, writer=writer, train_loader=train_loader, val_loader=val_loader, learning_rate=learning_rate, numepochs=numepochs)
  # Finsihe training
  logger.info("[+] Finished training!")
  writer.close()
  
# Running script
# python train_select_comb_ann.py --numepochs 60 --scheduler_end_factor 0.05 --learning_rate 1e-3 --weight_decay 1e-4 --dropout_rate 0.1 --batch_size 512 --window_size 1800 --is_norm True --is_bn False --lr_sloth_factor 0.5 --data-proportion 0.5