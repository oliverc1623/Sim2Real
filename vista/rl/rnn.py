import torch
import torch.nn as nn


import torch
import torch.nn as nn

class MyRNN(nn.Module):
    def __init__(self, input_size, hidden_size, num_layers):
        super(MyRNN, self).__init__()
        self.hidden_size = hidden_size
        self.num_layers = num_layers
        self.rnn = nn.RNN(input_size, hidden_size, num_layers, batch_first=True)
        self.fc_mean = nn.Linear(hidden_size, 1)
        self.fc_std = nn.Linear(hidden_size, 1)

    def forward(self, x):
        # Set initial hidden state
        h0 = torch.zeros(self.num_layers, x.size(0), self.hidden_size).to(x.device)

        # Forward propagate RNN
        out, _ = self.rnn(x, h0)

        # Decode hidden state of last time step
        mean = self.fc_mean(out[:, -1, :])
        std = self.fc_std(out[:, -1, :])

        return mean, std