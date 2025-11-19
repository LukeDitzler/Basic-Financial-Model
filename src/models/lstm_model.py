import torch
import torch.nn as nn

class FinancialLSTM(nn.Module):
    def __init__(self, input_dim, hidden_dim=64, num_layers=2):
        super().__init__()
        self.lstm = nn.LSTM(input_dim, hidden_dim, num_layers, batch_first=True)
        self.fc = nn.Linear(hidden_dim, 1)

    def forward(self, x):
        output, _ = self.lstm(x)     # output: (batch, seq, hidden)
        final = output[:, -1, :]     # last timestep
        return self.fc(final)
