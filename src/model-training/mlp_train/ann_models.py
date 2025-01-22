# for DL modeling
import torch
import torch.nn as nn
import torch.nn.functional as F

class AnnRMSModel(nn.Module):
  def __init__(self, features, is_bn=False, dropout=0.1):
    super().__init__()
    self.features = features
    self.is_bn = is_bn
    self.dropout = dropout
    ### input layer
    self.input = nn.Linear(len(self.features), 16)
    self.bn_input = nn.BatchNorm1d(16)
    ### hidden layers
    self.fc1    = nn.Linear(16, 32)
    self.bn1    = nn.BatchNorm1d(32)
    self.fc2    = nn.Linear(32, 64)
    self.bn2    = nn.BatchNorm1d(64)
    self.fc3    = nn.Linear(64, 80)
    self.bn3    = nn.BatchNorm1d(80)
    self.fc4    = nn.Linear(80, 96)
    self.bn4    = nn.BatchNorm1d(96)
    self.fc5    = nn.Linear(96, 160)
    self.bn5    = nn.BatchNorm1d(160)
    self.fc6    = nn.Linear(160, 160)
    self.bn6    = nn.BatchNorm1d(160)
    ### output layer
    self.output = nn.Linear(160, 128)
    self.dropout_layer = nn.Dropout(self.dropout)
      
  #forward pass
  def forward(self, x):
    if not self.is_bn:
      x = F.relu(self.input(x))
      x = self.dropout_layer(x)
      x = F.relu(self.fc1(x))
      x = self.dropout_layer(x)
      x = F.relu(self.fc2(x))
      x = self.dropout_layer(x)
      x = F.relu(self.fc3(x))
      x = self.dropout_layer(x)
      x = F.relu(self.fc4(x))
      x = self.dropout_layer(x)
      x = F.relu(self.fc5(x))
      x = self.dropout_layer(x)
      x = F.relu(self.fc6(x))
      x = self.dropout_layer(x)
    else:
      x = self.bn_input(F.relu(self.input(x)))
      x = self.dropout_layer(x)
      x = self.bn1(F.relu(self.fc1(x)))
      x = self.dropout_layer(x)
      x = self.bn2(F.relu(self.fc2(x)))
      x = self.dropout_layer(x)
      x = self.bn3(F.relu(self.fc3(x)))
      x = self.dropout_layer(x)
      x = self.bn4(F.relu(self.fc4(x)))
      x = self.dropout_layer(x)
      x = self.bn5(F.relu(self.fc5(x)))
      x = self.dropout_layer(x)
      x = self.bn6(F.relu(self.fc6(x)))
      x = self.dropout_layer(x)
    #return to the output layer
    return self.output(x)