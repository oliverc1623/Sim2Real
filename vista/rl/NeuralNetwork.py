### Define the self-driving agent ###
import torch
from torch import nn
from torch.utils.data import DataLoader
from torchvision import datasets
from torchvision.transforms import ToTensor
from torch.nn import ReLU
import torch.nn.functional as F  # for the activation function

# hyperparameters
max_curvature = 1/8. 
max_std = 0.1 

device = "cuda" if torch.cuda.is_available() else "mps" if torch.backends.mps.is_available() else "cpu"
print(f"Using {device} device")

# Define model
class NeuralNetwork(nn.Module):
    def __init__(self):
        super().__init__()
        self.conv1 = nn.Conv2d(3, 32, 5)
        self.conv2 = nn.Conv2d(32, 48, 5)
        self.conv3 = nn.Conv2d(48, 64, 3)
        self.flatten = nn.Flatten()
        self.fc1 = nn.Linear(64 * 35 * 145, 128)
        self.fc2 = nn.Linear(128, 2)

    def forward(self, x):
        x = F.relu(self.conv1(x))
        x = F.relu(self.conv2(x))
        x = F.relu(self.conv3(x))
        x = self.flatten(x)
        x = F.relu(self.fc1(x))
        x = self.fc2(x)
        return x    

def run_driving_model(image, driving_model):
    # Arguments:
    #   image: an input image
    # Returns:
    #   pred_dist: predicted distribution of control actions 
    single_image_input = len(image.shape) == 3  # missing 4th batch dimension
    if single_image_input:
        image = torch.unsqueeze(image, dim=0)

    image = image.permute(0,3,1,2)
    distribution = driving_model(image) 

    mu, logsigma = torch.split(distribution, 1, dim=1)
    mu = max_curvature * torch.tanh(mu) # conversion
    sigma = max_std * torch.sigmoid(logsigma) + 0.005 # conversion
    
    pred_dist = torch.distributions.normal.Normal(loc=mu, scale=sigma)

    return pred_dist


def compute_driving_loss(dist, actions, rewards):
    # Arguments:
    #   logits: network's predictions for actions to take
    #   actions: the actions the agent took in an episode
    #   rewards: the rewards the agent received in an episode
    # Returns:
    #   loss
    with torch.enable_grad():
        # print(f"type: {type(actions)}")
        neg_logprob = -1 * dist.log_prob(actions)
        neg_logprob = neg_logprob.detach().numpy() 
        total_loss = torch.tensor(neg_logprob * rewards, requires_grad=True)
        loss = torch.mean(total_loss) 
        return loss