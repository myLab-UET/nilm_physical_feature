import sys
PROJECT_PATH = "/home/mrcong/Code/mylab-nilm-files/nilm-physical-features"
sys.path.append(f"{PROJECT_PATH}/src/common")

import numpy as np
import pandas as pd
import polars as pl
import torch
from tqdm import tqdm
from torchinfo import summary
from sequence_model import LSTMModel, RNNModel, GRUModel
import torch.nn as nn
from utils import setup_logger
import torch.optim.lr_scheduler as lr_scheduler
from torch.utils.tensorboard import SummaryWriter
from ts_utils import create_timeseries_dataset, data_base_dir
import argparse

# Parsing arguments
parser = argparse.ArgumentParser(description="Sequence model training.")
parser.add_argument("--epochs", type=int, default=60, help="Number of training epochs.")
parser.add_argument("--batch_size", type=int, default=512, help="Batch size for training.")
parser.add_argument("--learning_rate", type=float, default=1e-3, help="Initial learning rate.")
parser.add_argument("--model", type=str, default="GRU", help="Model type to train.")
parser.add_argument("--features", type=str, default="Irms,MeanPF,P,Q,S", help="Features to use for training, including Irms,MeanPF,P,Q,S,In,Un,PF_n.")
parser.add_argument("--gpu", type=int, default=0, help="GPU device to use.")
parser.add_argument("--window_size", type=int, default=15, help="Window size for time series data.")
parser.add_argument("--hidden_size", type=int, default=15, help="Hidden size for RNN models.")
parser.add_argument("--num_layers", type=int, default=2, help="Number of layers for RNN models.")
parser.add_argument("--normalized", type=bool, default=False, help="Normalize the data.")
parser.add_argument("--workers", type=int, default=12, help="Number of workers for data loading.")
args = parser.parse_args()

num_epochs = args.epochs
batch_size = args.batch_size
lr_init = args.learning_rate
features = args.features.split(",")
model_name_arg = args.model
device = torch.device(f"cuda:{args.gpu}" if torch.cuda.is_available() else "cpu")
hiden_size = args.hidden_size
num_layers = args.num_layers
window_size = args.window_size
num_workers = args.workers

# Set logging level
data_info   = f"{data_base_dir}/data_information/data_information.xlsx"
save_path   = f"{PROJECT_PATH}/results/models/VNDALE/window_1800/ts_model"
model_name  = f"{model_name_arg}_{features}_w_{window_size}_h_{hiden_size}_n_{num_layers}"
logger      = setup_logger(save_path, model_name)
tb_log_dir  = f"{PROJECT_PATH}/results/tensorboard/VNDALE/TS_1800"
writer      = SummaryWriter(log_dir=f"{tb_log_dir}/{model_name}")

def main():
    logger.info(f"[+] Training {model_name} on {device}")
    logger.info("Loading data...")
    trainDataloader = create_timeseries_dataset(window_size, batch_size, features, "train", num_workers, normalized=args.normalized)
    valDataloader = create_timeseries_dataset(window_size, batch_size, features, "val", num_workers, normalized=args.normalized)
    logger.info("Data loaded successfully!")
    logger.info(f"Train data len: {len(trainDataloader.dataset)}, Val data len: {len(valDataloader.dataset)}")
    logger.info(f"First sample data: {trainDataloader.dataset[20000][0]}")
    
    # Start training
    model = None
    if model_name_arg == "LSTM":
        model = LSTMModel(input_size=len(features), hidden_size=hiden_size, num_layers=num_layers, output_size=128)
    elif model_name_arg == "RNN":
        model = RNNModel(input_size=len(features), hidden_size=hiden_size, num_layers=num_layers, output_size=128)
    elif model_name_arg == "GRU":
        model = GRUModel(input_size=len(features), hidden_size=hiden_size, num_layers=num_layers, output_size=128)
    else:
        raise ValueError("Model type not supported.")
    
    criterion = nn.CrossEntropyLoss()
    optimizer = torch.optim.Adam(model.parameters(), lr=lr_init)
    scheduler = lr_scheduler.LinearLR(optimizer, start_factor=1.0, end_factor=0.05, total_iters=int(num_epochs//2))
    logger.info(summary(model, input_size=(batch_size, window_size, len(features)))) 
    model = model.to(device)
    min_val_loss = 1e4
    for epoch in range(num_epochs):
        print(f"Epoch {epoch+1}/{num_epochs}")
        model.train()
        train_batch_loss = []
        val_batch_loss = []
        
        for inputs, labels in tqdm(trainDataloader, desc=f"Training"):
            inputs = inputs.to(device, non_blocking=True)
            labels = labels.to(device, non_blocking=True)
            outputs = model(inputs)
            loss = criterion(outputs, labels.long())
            optimizer.zero_grad(set_to_none=True)
            loss.backward()
            optimizer.step()
            train_batch_loss.append(loss.item())

        # Model evaluation
        model.eval()
        with torch.no_grad():
            for inputs, labels in tqdm(valDataloader, desc=f"Validation"):
                inputs = inputs.to(device)
                labels = labels.to(device)
                outputs = model(inputs)
                loss = criterion(outputs, labels.long())
                val_batch_loss.append(loss.item())

        # Log results
        avg_train_loss = sum(train_batch_loss) / len(train_batch_loss)
        avg_val_loss = sum(val_batch_loss) / len(val_batch_loss)
        logger.info(f'Epoch [{epoch+1}/{num_epochs}], Train loss: {avg_train_loss:.4f}, Val Loss: {avg_val_loss:.4f}')
        writer.add_scalar("Loss/Train", avg_train_loss, epoch + 1)
        writer.add_scalar("Loss/Val", avg_val_loss, epoch + 1)
        writer.add_scalar("Learning rate", optimizer.param_groups[0]["lr"], epoch + 1)
        scheduler.step()
        
        # Save best model
        if avg_val_loss < min_val_loss:
            min_val_loss = avg_val_loss
            logger.info(f"Saving best model at epoch {epoch+1}")
            torch.save(model.state_dict(), f"{save_path}/{model_name}.pt")
    # Save model
    writer.close()
    
if __name__ == "__main__":
    main()
