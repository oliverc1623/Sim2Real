import torchvision.models as models
from torch import nn

class ResNet34(nn.Module):
    def __init__(self):
        super(ResNet34, self).__init__()

        # Load the pre-trained ResNet50 model
        self.resnet34 = models.resnet34(weights='ResNet34_Weights.DEFAULT')

        # Replace the last fully connected layer to output two numerical values
        num_ftrs = self.resnet34.fc.in_features
        self.resnet34.fc = nn.Linear(num_ftrs, 2)

    def forward(self, x):
        x = self.resnet34(x)
        return x

class ResNet18(nn.Module):
    def __init__(self):
        super(ResNet18, self).__init__()

        # Load the pre-trained ResNet50 model
        self.resnet18 = models.resnet18(weights='ResNet18_Weights.DEFAULT')

        # Replace the last fully connected layer to output two numerical values
        num_ftrs = self.resnet18.fc.in_features
        self.resnet18.fc = nn.Linear(num_ftrs, 2)

    def forward(self, x):
        x = self.resnet18(x)
        return x