import torch
import torch.nn as nn

class RNNModel(nn.Module):
    def __init__(self, input_size=465, hidden_size=64, num_layers=1, num_classes=2):
        super(RNNModel, self).__init__()

        self.hidden_size = hidden_size
        self.num_layers = num_layers

        self.lstm1 = nn.LSTM(input_size, hidden_size, num_layers=1, batch_first=True)
        self.lstm2 = nn.LSTM(hidden_size, hidden_size, num_layers=1, batch_first=True)
        self.lstm3 = nn.LSTM(hidden_size, hidden_size, num_layers=1, batch_first=True)
        self.lstm4 = nn.LSTM(hidden_size, hidden_size, num_layers=1, batch_first=True)
        self.lstm5 = nn.LSTM(hidden_size, hidden_size, num_layers=1, batch_first=True)
        self.lstm6 = nn.LSTM(hidden_size, hidden_size, num_layers=1, batch_first=True)
        self.fc = nn.Linear(hidden_size, num_classes)

    def forward(self, x):
        # Reshape input to (batch_size, sequence_length, input_size)
        x = x.permute(0, 2, 3, 1)
        x = x.reshape(x.size(0), x.size(1), x.size(2)*x.size(3))
        
        # Set initial hidden and cell states
        h0 = torch.zeros(self.num_layers, x.size(0), self.hidden_size).to(x.device)
        c0 = torch.zeros(self.num_layers, x.size(0), self.hidden_size).to(x.device)

        # Forward propagate LSTMs
        out, _ = self.lstm1(x, (h0, c0))
        out, _ = self.lstm2(out, (h0, c0))
        out, _ = self.lstm3(out, (h0, c0))
        out, _ = self.lstm4(out, (h0, c0))
        out, _ = self.lstm5(out, (h0, c0))
        out, _ = self.lstm6(out, (h0, c0))

        # Decode the hidden state of the last time step
        out = self.fc(out[:, -1, :])

        return out