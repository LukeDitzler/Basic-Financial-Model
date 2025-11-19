import torch
import torch.nn as nn

class FFResidualV2(nn.Module):
    def __init__(self, input_dim):
        super(FFResidualV2, self).__init__()

        self.input_layer = nn.Linear(input_dim, 256)
        self.bn1 = nn.BatchNorm1d(256)
        
        self.hidden1 = nn.Linear(256, 128)
        self.bn2 = nn.BatchNorm1d(128)
        self.dropout1 = nn.Dropout(0.3)
        
        self.hidden2 = nn.Linear(128, 128)
        self.bn3 = nn.BatchNorm1d(128)
        self.dropout2 = nn.Dropout(0.3)
        
        self.hidden3 = nn.Linear(128, 64)
        self.bn4 = nn.BatchNorm1d(64)
        self.dropout3 = nn.Dropout(0.2)
        
        self.output = nn.Linear(64, 1)
        
    def forward(self, x):
        x = torch.relu(self.bn1(self.input_layer(x)))
        
        x = torch.relu(self.bn2(self.hidden1(x)))
        x = self.dropout1(x)
        
        # Residual connection
        identity = x
        x = torch.relu(self.bn3(self.hidden2(x)))
        x = self.dropout2(x + identity)
        
        x = torch.relu(self.bn4(self.hidden3(x)))
        x = self.dropout3(x)
        
        return self.output(x)