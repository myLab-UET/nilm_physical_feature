import numpy as np
import pandas as pd
import polars as pl
from sklearn.preprocessing import LabelEncoder
import time
import sys
PROJECT_PATH = "/home/mrcong/Code/mylab-nilm-files/nilm-physical-features"
sys.path.append(f"{PROJECT_PATH}/src/common")
from nilm_dao import get_vndale1_data, get_vndale2_data

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
from sklearn.model_selection import train_test_split
import argparse

logger = logging.getLogger()
date_time = datetime.datetime.now().strftime("%Y-%m-%d")

# Configs
# Argument parser
parser = argparse.ArgumentParser(description="Train MLP model with selected features")
parser.add_argument("--numepochs", type=int, default=60, help="Number of epochs")
parser.add_argument("--scheduler_end_factor", type=float, default=0.1, help="Scheduler end factor")
parser.add_argument("--learning_rate", type=float, default=1.2e-3, help="Learning rate")
parser.add_argument("--weight_decay", type=float, default=1e-4, help="Weight decay")
parser.add_argument("--dropout_rate", type=float, default=0, help="Dropout rate")
parser.add_argument("--batch_size", type=int, default=512, help="Batch size")
parser.add_argument("--window_size", type=int, default=1800, help="Window size")
parser.add_argument("--is_norm", type=bool, default=True, help="Normalization flag")
parser.add_argument("--is_bn", type=bool, default=False, help="Batch normalization flag")
parser.add_argument("--lr_sloth_factor", type=float, default=0.5, help="Learning rate sloth factor")
parser.add_argument("--data-proportion", type=float, default=0.5, help="Data proportion")

args = parser.parse_args()

numepochs                = args.numepochs
scheduler_end_factor     = args.scheduler_end_factor
learning_rate            = args.learning_rate
weight_decay             = args.weight_decay
dropout_rate             = args.dropout_rate
batch_size               = args.batch_size
window_size              = args.window_size
is_norm                  = args.is_norm
is_bn                    = args.is_bn
lr_sloth_factor          = args.lr_sloth_factor
data_proportion          = args.data_proportion
if data_proportion > 1 or data_proportion < 0:
  raise ValueError("Data proportion must be in range [0, 1]")
# Learning rate scheduler
scheduler_iter           = int(lr_sloth_factor*numepochs)
device                   = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')

# Model saving directory
model_saving_dir         = f"{PROJECT_PATH}/results/models/VNDALE1/window_{window_size}/mlp_model"
log_file_path            = f"{model_saving_dir}/train_ann_diff_data.log"
if not os.path.exists(model_saving_dir):
  os.makedirs(model_saving_dir)
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(name)s - %(levelname)s - %(message)s', 
                    handlers=[logging.FileHandler(log_file_path), logging.StreamHandler(sys.stdout)])

# Function to train the model
def trainTheModel(ann_rms: AnnRMSModel, writer: SummaryWriter, train_loader:DataLoader, val_loader:DataLoader, learning_rate, numepochs:int):
  # loss function and optimizer
  lossfun = nn.CrossEntropyLoss()
  optimizer = torch.optim.Adam(ann_rms.parameters(), lr=learning_rate, weight_decay=weight_decay)
  scheduler = lr_scheduler.LinearLR(optimizer, start_factor=1.0, end_factor=scheduler_end_factor, total_iters=scheduler_iter)
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
    writer.add_scalar("Loss/Training", np.mean(batchLoss), epochi)
    writer.add_scalar("Loss/Validation", np.mean(valBatchLoss), epochi)
    logger.info(f"[+] Epoch {epochi}/{numepochs} - Training loss: {np.mean(batchLoss)} - Validation loss: {val_loss} - lr: {optimizer.param_groups[0]['lr']} - Training time: {time.time() - start_time}")
  # function output
  return ann_rms

# Loading data
logging.info(f"Configs: {args}")
logging.info("[+] Loading data")
train_df = get_vndale1_data("train", window_size, is_norm=is_norm)
validation_df = get_vndale1_data("val", window_size, is_norm=is_norm)

logging.info("[+] Training df")
logging.info(train_df.head())

logging.info("[+] Validation df")
logging.info(validation_df.head())

# Features combs
feature_combs = [
  ['Irms', 'P', 'MeanPF', 'S', 'Q'],
]

for features in feature_combs:
  # Change X, y to tensors, and setup
  logging.info(f"[+] Training model with features: {features}")
  X_val = validation_df.select(features).to_numpy()
  X_train = train_df.select(features).to_numpy()
  y_train = train_df["Label"].to_numpy()
  y_val = validation_df["Label"].to_numpy()
  
  # Select only a few data
  _X_train, _, _y_train, _ = train_test_split(X_train, y_train, test_size=data_proportion, random_state=42, stratify=y_train)
  X_train, y_train = _X_train, _y_train
  
  logging.info(f"[+] Train shape: {X_train.shape}, {y_train.shape}")
  logging.info(f"[+] Val shape: {X_val.shape}, {y_val.shape}")
  logging.info(f"[+] First 5 rows of X_train: {X_train[:5]}")
  
  # Convert to tensors
  X_val   = torch.tensor(X_val).float()
  X_train = torch.tensor(X_train).float()
  y_train = torch.tensor(y_train)
  y_val   = torch.tensor(y_val)
  
  logging.info(f"Training set: {X_train.shape}, y_train: {y_train.shape}")
  logging.info(f"Validation set: {X_val.shape}, y_val: {y_val.shape}")
  
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
  if len(features) == 1:
    train_df, validation_df = None, None
  
  # Transfer model to GPU
  model = AnnRMSModel(features, is_bn=is_bn, dropout=dropout_rate)
  test_example = next(iter(train_loader))
  test_output = model(test_example[0])
  logging.info(f"[+] Model output shape: {test_output.shape}")
  
  # Model training
  logging.info("[+] Weight initialization for new model!")
  for p in model.named_parameters():
    if 'weight' in p[0] and p[1].data.dim() >= 2:
      p[1].data = nn.init.kaiming_normal_(p[1].data, nonlinearity="relu")
  logging.info("[+] Start training model!")
  model.to(device)
  tb_log_dir = f"{PROJECT_PATH}/results/tensorboard/VNDALE1/MLP_{window_size}"
  writer = SummaryWriter(log_dir=f"{tb_log_dir}/MLP_{features}")
  model = trainTheModel(ann_rms=model, writer=writer, train_loader=train_loader, val_loader=val_loader, learning_rate=learning_rate, numepochs=numepochs)
  logging.info("[+] Finished training!")

  #Saving model's state
  model_output_name = f"mlp_{features}.pt"
  output_state_dict_path = f"{model_saving_dir}/{model_output_name}"
  logging.info("[+] Saving model's state!")
  try:
    torch.save(model.state_dict(), f"{output_state_dict_path}")
    logging.info(f"[+] Model saved to {output_state_dict_path}")      
  except Exception as e:
    logging.error(e)
    torch.save(model.state_dict(), f"{model_output_name}")
  writer.close()
  
# Running script
# python train_select_comb_ann.py --numepochs 60 --scheduler_end_factor 0.05 --learning_rate 1e-3 --weight_decay 1e-4 --dropout_rate 0.1 --batch_size 512 --window_size 1800 --is_norm True --is_bn False --lr_sloth_factor 0.5 --data-proportion 0.5