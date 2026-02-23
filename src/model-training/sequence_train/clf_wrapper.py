import os
import torch
import torch.nn as nn
import torch.optim as optim
import numpy as np
from clf_models import get_model_by_name
from data_handler import create_data_loader
import time

class ClassifierWrapper:
    def __init__(self, 
                 output_directory, 
                 input_shape, 
                 nb_classes, 
                 training_config,
                 clf_name='fcn', 
                 verbose=False, 
                 build=True, 
                 custom_loss=None,
                 num_workers=4,
                 pin_memory=False):
        self.output_directory = output_directory
        self.input_shape = input_shape
        self.nb_classes = nb_classes
        self.training_config = training_config
        self.custom_loss = custom_loss  
        self.clf_name = clf_name
        self.verbose = verbose
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.num_workers = num_workers
        self.pin_memory = pin_memory and torch.cuda.is_available()
        
        if build:
            self.model = self.build_model()
            self.model.to(self.device)
            # Compile model if available (PyTorch 2.0+)
            if hasattr(torch, 'compile'):
                print("Compiling model via torch.compile...")
                try:
                    self.model = torch.compile(self.model)
                except (ImportError, RuntimeError) as e:
                    print(f"Failed to compile model: {e}. Proceeding without compilation.")

    def build_model(self):
        return get_model_by_name(self.clf_name, self.input_shape, self.nb_classes)

    def train(self, x_train, y_train, x_val, y_val, batch_size=None, epochs=None):
        if batch_size is None:
            batch_size = self.training_config['batch_size']
        if epochs is None:
            epochs = self.training_config['epochs']
            
        if not os.path.exists(self.output_directory):
            os.makedirs(self.output_directory)
            
        # Create DataLoaders
        train_loader = create_data_loader(x_train, y_train, batch_size=batch_size, shuffle=True, num_workers=self.num_workers, pin_memory=self.pin_memory)
        val_loader = create_data_loader(x_val, y_val, batch_size=batch_size, shuffle=False, num_workers=self.num_workers, pin_memory=self.pin_memory)
        
        # Optimizer and Loss
        optimizer = optim.Adam(self.model.parameters(), lr=self.training_config['learning_rate'])
        criterion = self.custom_loss if self.custom_loss else nn.CrossEntropyLoss()
        
        # Scheduler
        scheduler = optim.lr_scheduler.ReduceLROnPlateau(optimizer, mode='max', factor=0.5, patience=epochs//10, min_lr=0.0001)
        
        best_val_acc = 0.0
        
        # Log file
        log_file_path = os.path.join(self.output_directory, 'training.log')
        with open(log_file_path, 'w') as f:
            f.write(f"Training model with batch size: {batch_size} and epochs: {epochs}\n")
        
        print(f"Training model with batch size: {batch_size} and epochs: {epochs}")
        
        for epoch in range(epochs):
            self.model.train()
            train_loss = 0.0
            correct = 0
            total = 0
            epoch_start_time = time.time()
            
            for inputs, targets in train_loader:
                inputs, targets = inputs.to(self.device), targets.to(self.device)
                
                # Convert one-hot targets to class indices for CrossEntropyLoss
                if targets.dim() > 1:
                    targets = torch.argmax(targets, dim=1)
                
                optimizer.zero_grad()
                outputs = self.model(inputs)
                loss = criterion(outputs, targets)
                loss.backward()
                optimizer.step()
                
                train_loss += loss.item()
                _, predicted = outputs.max(1)
                total += targets.size(0)
                correct += predicted.eq(targets).sum().item()
            
            train_acc = 100. * correct / total
            
            # Validation
            val_loss, val_acc = self.evaluate(val_loader, criterion)
            
            # Scheduler step
            scheduler.step(val_acc)
            
            if self.verbose and epoch % 5 == 0:
                epoch_duration = time.time() - epoch_start_time
                log_msg = (f"Epoch [{epoch+1}/{epochs}] "
                           f"Train Loss: {train_loss/len(train_loader):.4f} | Train Acc: {train_acc:.2f}% | "
                           f"Val Loss: {val_loss:.4f} | Val Acc: {val_acc:.2f}% | "
                           f"Training time: {epoch_duration:.2f} seconds")
                print(log_msg)
                with open(log_file_path, 'a') as f:
                    f.write(log_msg + "\n")
            
            # Save best model
            if val_acc > best_val_acc:
                best_val_acc = val_acc
                torch.save(self.model.state_dict(), os.path.join(self.output_directory, 'best_model.pth'))
                if self.verbose:
                    print("Saved best model.")
                    with open(log_file_path, 'a') as f:
                        f.write("Saved best model.\n")
        
        print(f"Training finished. Best Validation Accuracy: {best_val_acc:.2f}%")
        with open(log_file_path, 'a') as f:
            f.write(f"Training finished. Best Validation Accuracy: {best_val_acc:.2f}%\n")

    def evaluate(self, dataloader, criterion=None):
        # Change to evaluation mode
        self.model.eval()
        val_loss = 0.0
        correct = 0
        total = 0
        
        if criterion is None:
            criterion = nn.CrossEntropyLoss()
            
        with torch.no_grad():
            for inputs, targets in dataloader:
                inputs, targets = inputs.to(self.device), targets.to(self.device)
                
                if targets.dim() > 1:
                    targets = torch.argmax(targets, dim=1)
                    
                outputs = self.model(inputs)
                loss = criterion(outputs, targets)
                val_loss += loss.item()
                _, predicted = outputs.max(1)
                total += targets.size(0)
                correct += predicted.eq(targets).sum().item()
        
        return val_loss / len(dataloader), 100. * correct / total

    def predict(self, x_data, batch_size=512):
        self.model.eval()
        # Create a simple DataLoader or iterate manually
        # Since x_data is numpy, convert to tensor
        dataset = torch.utils.data.TensorDataset(torch.from_numpy(x_data))
        loader = torch.utils.data.DataLoader(dataset, batch_size=batch_size, shuffle=False)
        
        predictions = []
        with torch.no_grad():
            for inputs in loader:
                inputs = inputs[0].to(self.device)
                outputs = self.model(inputs)
                predictions.append(outputs.cpu().numpy())
        
        return np.concatenate(predictions, axis=0)
