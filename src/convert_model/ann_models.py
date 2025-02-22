# for DL modeling
import torch
import torch.nn as nn
import torch.nn.functional as F

class AnnRMSModel(nn.Module):
  def __init__(self, input_dim, output_dim, is_bn=False, dropout=0):
    super().__init__()
    self.is_bn = is_bn
    self.dropout = dropout
    ### input layer
    self.input = nn.Linear(input_dim, 16)
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
    self.output = nn.Linear(160, output_dim)
    ### dropout layers
    self.do1 = nn.Dropout(dropout)
    self.do2 = nn.Dropout(dropout)
    self.do3 = nn.Dropout(dropout)
    self.do4 = nn.Dropout(dropout)
    self.do5 = nn.Dropout(dropout)
    self.do6 = nn.Dropout(dropout)
      
  #forward pass
  def forward(self, x):
    if not self.is_bn:
      x = F.relu(self.input(x))
      x = self.do1(x)
      x = F.relu(self.fc1(x))
      x = self.do2(x)
      x = F.relu(self.fc2(x))
      x = self.do3(x)
      x = F.relu(self.fc3(x))
      x = self.do4(x)
      x = F.relu(self.fc4(x))
      x = self.do5(x)
      x = F.relu(self.fc5(x))
      x = self.do6(x)
      x = F.relu(self.fc6(x))
      x = self.do6(x)
    else:
      x = self.bn_input(F.relu(self.input(x)))
      x = self.do1(x)
      x = self.bn1(F.relu(self.fc1(x)))
      x = self.do2(x)
      x = self.bn2(F.relu(self.fc2(x)))
      x = self.do3(x)
      x = self.bn3(F.relu(self.fc3(x)))
      x = self.do4(x)
      x = self.bn4(F.relu(self.fc4(x)))
      x = self.do5(x)
      x = self.bn5(F.relu(self.fc5(x)))
      x = self.do6(x)
      x = self.bn6(F.relu(self.fc6(x)))
      x = self.do6(x)
    #return to the output layer
    return self.output(x)
  
class MLPModel2(nn.Module):
  def __init__(self, input_dim, output_dim, is_bn=False, dropout=0):
    super().__init__()
    self.is_bn = is_bn
    self.dropout = dropout
    ### input layer
    self.input = nn.Linear(input_dim, 16)
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
    self.fc7    = nn.Linear(160, 180)
    self.bn7    = nn.BatchNorm1d(180)
    self.fc8    = nn.Linear(180, 200)
    self.bn8    = nn.BatchNorm1d(200)
    ### output layer
    self.output = nn.Linear(200, output_dim)
    ### dropout layers
    self.do1 = nn.Dropout(dropout)
    self.do2 = nn.Dropout(dropout)
    self.do3 = nn.Dropout(dropout)
    self.do4 = nn.Dropout(dropout)
    self.do5 = nn.Dropout(dropout)
    self.do6 = nn.Dropout(dropout)
    self.do7 = nn.Dropout(dropout)
    self.do8 = nn.Dropout(dropout)
      
  #forward pass
  def forward(self, x):
    if not self.is_bn:
      x = F.relu(self.input(x))
      x = self.do1(x)
      x = F.relu(self.fc1(x))
      x = self.do2(x)
      x = F.relu(self.fc2(x))
      x = self.do3(x)
      x = F.relu(self.fc3(x))
      x = self.do4(x)
      x = F.relu(self.fc4(x))
      x = self.do5(x)
      x = F.relu(self.fc5(x))
      x = self.do6(x)
      x = F.relu(self.fc6(x))
      x = self.do7(x)
      x = F.relu(self.fc7(x))
      x = self.do8(x)
      x = F.relu(self.fc8(x))
      x = self.do8(x)
    else:
      x = self.bn_input(F.relu(self.input(x)))
      x = self.do1(x)
      x = self.bn1(F.relu(self.fc1(x)))
      x = self.do2(x)
      x = self.bn2(F.relu(self.fc2(x)))
      x = self.do3(x)
      x = self.bn3(F.relu(self.fc3(x)))
      x = self.do4(x)
      x = self.bn4(F.relu(self.fc4(x)))
      x = self.do5(x)
      x = self.bn5(F.relu(self.fc5(x)))
      x = self.do6(x)
      x = self.bn6(F.relu(self.fc6(x)))
      x = self.do7(x)
      x = self.bn7(F.relu(self.fc7(x)))
      x = self.do8(x)
      x = self.bn8(F.relu(self.fc8(x)))
      x = self.do8(x)
    #return to the output layer
    return self.output(x)