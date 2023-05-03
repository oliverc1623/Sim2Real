import torch
import torch.nn as nn

import torch.nn as nn

class CNN(nn.Module):
    def __init__(self):
        super(CNN, self).__init__()
        self.conv1 = nn.Conv2d(3, 24, kernel_size=5, stride=2, padding=1)
        self.norm1 = nn.GroupNorm(6, 24)
        self.relu1 = nn.ReLU(inplace=True)

        self.conv2 = nn.Conv2d(24, 36, kernel_size=5, stride=2, padding=1)
        self.norm2 = nn.GroupNorm(9, 36)
        self.relu2 = nn.ReLU(inplace=True)

        self.conv3 = nn.Conv2d(36, 48, kernel_size=3, stride=1, padding=1)
        self.norm3 = nn.GroupNorm(12, 48)
        self.relu3 = nn.ReLU(inplace=True)

        self.conv4 = nn.Conv2d(48, 64, kernel_size=3, stride=1, padding=1)
        self.norm4 = nn.GroupNorm(16, 64)
        self.relu4 = nn.ReLU(inplace=True)

        self.conv5 = nn.Conv2d(64, 128, kernel_size=3, stride=1, padding=1)
        self.norm5 = nn.GroupNorm(32, 128)
        self.relu5 = nn.ReLU(inplace=True)

        self.flatten = nn.Flatten()

        self.fc1 = nn.Linear(64*19*49, 250)
        self.relu5 = nn.ReLU(inplace=True)
        self.fc2 = nn.Linear(250, 24)
        self.relu6 = nn.ReLU(inplace=True)
        self.mean = nn.Linear(24, 1)
        self.std = nn.Linear(24,1)
        
    def forward(self, x):
        x = self.conv1(x)
        x = self.relu1(x)
        x = self.norm1(x)
        
        x = self.conv2(x)        
        x = self.relu2(x)
        x = self.norm2(x)
        
        x = self.conv3(x)
        x = self.relu3(x)
        x = self.norm3(x)
        
        x = self.conv4(x)        
        x = self.relu4(x)
        x = self.norm4(x)

        x = self.flatten(x)

        x = self.fc1(x)
        x = self.relu5(x)
        x = self.fc2(x)
        
        mean = self.mean(x)
        std = self.std(x)

        return mean, std


class LSTM(nn.Module):
    def __init__(self):
        super(LSTM, self).__init__()
        self.conv1 = nn.Conv2d(3, 32, kernel_size=3, stride=1, padding=1)
        self.norm1 = nn.GroupNorm(8, 32)
        self.relu1 = nn.ReLU(inplace=True)
        
        self.conv2 = nn.Conv2d(32, 64, kernel_size=3, stride=1, padding=1)
        self.norm2 = nn.GroupNorm(16, 64)
        self.relu2 = nn.ReLU(inplace=True)
        
        self.conv3 = nn.Conv2d(64, 128, kernel_size=3, stride=1, padding=1)
        self.norm3 = nn.GroupNorm(32, 128)
        self.relu3 = nn.ReLU(inplace=True)
        
        self.conv4 = nn.Conv2d(128, 256, kernel_size=3, stride=1, padding=1)
        self.norm4 = nn.GroupNorm(64, 256)
        self.relu4 = nn.ReLU(inplace=True)
        
        self.conv5 = nn.Conv2d(256, 2, kernel_size=3, stride=1, padding=1)
        self.norm5 = nn.GroupNorm(1, 2)
        self.relu5 = nn.ReLU(inplace=True)

        self.lstm = nn.LSTM(input_size=2*30*32, hidden_size=128, num_layers=2, batch_first=True)
        self.fc = nn.Linear(128, 2)
        
    def forward(self, x):
        x = self.conv1(x)
        x = self.norm1(x)
        x = self.relu1(x)
        
        x = self.conv2(x)
        x = self.norm2(x)
        x = self.relu2(x)
        
        x = self.conv3(x)
        x = self.norm3(x)
        x = self.relu3(x)
        
        x = self.conv4(x)
        x = self.norm4(x)
        x = self.relu4(x)
        
        x = self.conv5(x)
        x = self.norm5(x)
        x = self.relu5(x)
        
        x = x.reshape(x.size(0), -1)
        x = x.unsqueeze(1)  # add a new dimension for the time steps
        
        # pass the output of the convolutions through the LSTM
        x, _ = self.lstm(x)
        x = x[:, -1, :]  # use only the output of the last time step
        
        x = self.fc(x)
        return x
