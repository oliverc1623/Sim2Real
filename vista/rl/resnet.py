import torchvision.models as models
from torch import nn

class ResNet50(nn.Module):
    def __init__(self):
        super(ResNet50, self).__init__()

        # Load the pre-trained ResNet50 model
        self.resnet50 = models.resnet50(weights='ResNet50_Weights.DEFAULT')

        # Replace the last fully connected layer to output two numerical values
        num_ftrs = self.resnet50.fc.in_features
        self.resnet50.fc = nn.Linear(num_ftrs, 2)

    def forward(self, x):
        x = self.resnet50(x)
        return x
