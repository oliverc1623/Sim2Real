# %%
import matplotlib.pyplot as plt
import vista
import os
import torch
from torch import nn
from memory import Memory
from tqdm import tqdm
import numpy as np
from NeuralNetwork import Model
import torch
import torch.nn.functional as F
import torch.distributions as dist
import torch.optim as optim
import time
import datetime
from IPython import display as ipythondisplay

# %%
device = (
    "cuda"
    if torch.cuda.is_available()
    else "mps"
    if torch.backends.mps.is_available()
    else "cpu"
)
print(f"Using {device} device")

# %%
### Agent Memory ###

class Memory:
    def __init__(self):
        self.clear()

    # Resets/restarts the memory buffer
    def clear(self):
        self.observations = []
        self.actions = []
        self.rewards = []

    # Add observations, actions, rewards to memory
    def add_to_memory(self, new_observation, new_action, new_reward):
        self.observations.append(new_observation)
        self.actions.append(new_action)
        self.rewards.append(new_reward)

    def __len__(self):
        return len(self.actions)


# Instantiate a single Memory buffer
memory = Memory()

# %%
### Reward function ###

# Helper function that normalizes an np.array x
def normalize(x):
    x -= torch.mean(x)
    x /= torch.std(x)
    return x


# Compute normalized, discounted, cumulative rewards (i.e., return)
# Arguments:
#   rewards: reward at timesteps in episode
#   gamma: discounting factor
# Returns:
#   normalized discounted reward
def discount_rewards(rewards, gamma=0.95):
    discounted_rewards = torch.zeros_like(rewards)
    R = 0
    for t in reversed(range(0, len(rewards))):
        # update the total discounted reward
        R = R * gamma + rewards[t]
        discounted_rewards[t] = R

    return normalize(discounted_rewards)


# %%
### Training step (forward and backpropagation) ###


def train_step(
    model,
    loss_function,
    optimizer,
    observations,
    actions,
    discounted_rewards,
    running_loss,
    custom_fwd_fn=None,
):
    # Forward propagate through the agent network
    if custom_fwd_fn is not None:
        prediction = custom_fwd_fn(observations)
    else:
        prediction = model(observations)
    loss = loss_function(prediction, actions, discounted_rewards)
    loss.backward()

    nn.utils.clip_grad_norm_(model.parameters(), 2)
    optimizer.step()
    running_loss += loss.item()
    optimizer.zero_grad()
    return running_loss


# %%
### Set up VISTA simulator ###
trace_root = "../trace"
trace_path = [
    "20210726-154641_lexus_devens_center",
    "20210726-155941_lexus_devens_center_reverse",
    "20210726-184624_lexus_devens_center",
    "20210726-184956_lexus_devens_center_reverse",
]
trace_path = [os.path.join(trace_root, p) for p in trace_path]
world = vista.World(trace_path, trace_config={"road_width": 4})
car = world.spawn_agent(
    config={
        "length": 5.0,
        "width": 2.0,
        "wheel_base": 2.78,
        "steering_ratio": 14.7,
        "lookahead_road": True,
    }
)
camera = car.spawn_camera(config={"size": (200, 320)})
display = vista.Display(world, display_config={"gui_scale": 2, "vis_full_frame": False})


def vista_reset():
    world.reset()
    display.reset()


vista_reset()


# %%
def vista_step(curvature=None, speed=None):
    # Arguments:
    #   curvature: curvature to step with
    #   speed: speed to step with
    if curvature is None:
        curvature = car.trace.f_curvature(car.timestamp)
    if speed is None:
        speed = car.trace.f_speed(car.timestamp)

    car.step_dynamics(action=np.array([curvature, speed]), dt=1 / 15.0)
    car.step_sensors()


# %%
### Terminal State Check ###
def check_out_of_lane(car):
    distance_from_center = np.abs(car.relative_state.x)
    road_width = car.trace.road_width
    half_road_width = road_width / 2
    return distance_from_center > half_road_width


def check_exceed_max_rot(car):
    maximal_rotation = np.pi / 10.0
    current_rotation = np.abs(car.relative_state.yaw)
    return current_rotation > maximal_rotation


def check_crash(car):
    return check_out_of_lane(car) or check_exceed_max_rot(car) or car.done

# %%
## Data preprocessing functions ##

def preprocess(full_obs):
    # Extract ROI
    i1, j1, i2, j2 = camera.camera_param.get_roi()
    obs = full_obs[i1:i2, j1:j2]

    # Rescale to [0, 1]
    obs = obs / 255.0
    return obs


def grab_and_preprocess_obs(car):
    full_obs = car.observations[camera.name]
    obs = preprocess(full_obs)
    obs = torch.from_numpy(obs).to(torch.float32)
    return obs


# %%
import torch.nn.functional as F


class Model(nn.Module):
    def __init__(self):
        super(Model, self).__init__()

        self.conv1 = nn.Conv2d(in_channels=3, out_channels=32, kernel_size=5, stride=2)
        self.bn1 = nn.BatchNorm2d(32)

        self.conv2 = nn.Conv2d(in_channels=32, out_channels=48, kernel_size=3, stride=2)
        self.bn2 = nn.BatchNorm2d(48)

        self.conv3 = nn.Conv2d(in_channels=48, out_channels=64, kernel_size=3, stride=2)
        self.bn3 = nn.BatchNorm2d(64)

        self.conv4 = nn.Conv2d(in_channels=64, out_channels=64, kernel_size=3, stride=2)
        self.bn4 = nn.BatchNorm2d(64)

        self.conv5 = nn.Conv2d(in_channels=64, out_channels=128, kernel_size=1, stride=2)
        self.bn5 = nn.BatchNorm2d(128)

        self.conv6 = nn.Conv2d(in_channels=128, out_channels=256, kernel_size=1, stride=2)
        self.bn6 = nn.BatchNorm2d(256)

        self.fc1 = nn.Linear(in_features=256 * 1 * 1, out_features=128)
        self.bn7 = nn.BatchNorm1d(128)

        self.fc2 = nn.Linear(in_features=128, out_features=2)

    def forward(self, x):
        x = F.relu(self.bn1(self.conv1(x)))
        x = F.relu(self.bn2(self.conv2(x)))
        x = F.relu(self.bn3(self.conv3(x)))
        x = F.relu(self.bn4(self.conv4(x)))
        x = F.relu(self.bn5(self.conv5(x)))
        x = F.relu(self.bn6(self.conv6(x)))
        x = x.reshape(-1, 256 * 1 * 1)
        x = F.relu(self.bn7(self.fc1(x)))
        x = self.fc2(x)

        return x

# %%
## The self-driving learning algorithm ##

# hyperparameters
max_curvature = 1 / 8.0
max_std = 0.1


def run_driving_model(image):
    # Arguments:
    #   image: an input image
    # Returns:
    #   pred_dist: predicted distribution of control actions
    single_image_input = len(image.shape) == 3  # missing 4th batch dimension
    if single_image_input:
        image = image.unsqueeze(0)

    image = image.permute(0, 3, 1, 2)
    # print(f"input shape: {image.shape}")
    distribution = driving_model(image)

    mu, logsigma = torch.chunk(distribution, 2, dim=1)
    mu = max_curvature * torch.tanh(mu)  # conversion
    sigma = max_std * torch.sigmoid(logsigma) + 0.005  # conversion

    pred_dist = dist.Normal(mu, sigma)
    return pred_dist


def compute_driving_loss(dist, actions, rewards):
    # Arguments:
    #   logits: network's predictions for actions to take
    #   actions: the actions the agent took in an episode
    #   rewards: the rewards the agent received in an episode
    # Returns:
    #   loss
    neg_logprob = -1 * dist.log_prob(actions)
    loss = torch.mean(neg_logprob * rewards)
    return loss


# %%
class LossHistory:
    def __init__(self, smoothing_factor=0.0):
        self.alpha = smoothing_factor
        self.loss = []

    def append(self, value):
        self.loss.append(
            self.alpha * self.loss[-1] + (1 - self.alpha) * value
            if len(self.loss) > 0
            else value
        )

    def get(self):
        return self.loss


class PeriodicPlotter:
    def __init__(self, sec, xlabel="", ylabel="", scale=None):
        self.xlabel = xlabel
        self.ylabel = ylabel
        self.sec = sec
        self.scale = scale

        self.tic = time.time()

    def plot(self, data):
        if time.time() - self.tic > self.sec:
            plt.cla()

            if self.scale is None:
                plt.plot(data)
            elif self.scale == "semilogx":
                plt.semilogx(data)
            elif self.scale == "semilogy":
                plt.semilogy(data)
            elif self.scale == "loglog":
                plt.loglog(data)
            else:
                raise ValueError("unrecognized parameter scale {}".format(self.scale))

            plt.xlabel(self.xlabel)
            plt.ylabel(self.ylabel)
            ipythondisplay.clear_output(wait=True)
            ipythondisplay.display(plt.gcf())

            self.tic = time.time()


# %%
## Training parameters and initialization ##
## Re-run this cell to restart training from scratch ##

# instantiate driving agent
vista_reset()
driving_model = Model()

learning_rate = 0.001  # 5e-4
optimizer = optim.SGD(driving_model.parameters(), lr=learning_rate)

# to track our progress
# get current date and time
now = datetime.datetime.now()
filename = f"results/results_{now.strftime('%Y-%m-%d_%H-%M-%S')}.txt"
# open file and write rewards and losses to it
f = open(filename, "w") 
f.write("reward\tloss\n")

# instantiate Memory buffer
memory = Memory()

# %%
## Driving training! Main training block. ##
## Note: stopping and restarting this cell will pick up training where you
#        left off. To restart training you need to rerun the cell above as
#        well (to re-initialize the model and optimizer)

max_batch_size = 300
max_reward = float("-inf")  # keep track of the maximum reward acheived during training
if hasattr(tqdm, "_instances"):
    tqdm._instances.clear()  # clear if it exists
for i_episode in range(2):
    driving_model.eval() # set to eval mode because we pass in a single image - not a batch
    running_loss = 0
    # Restart the environment
    vista_reset()
    memory.clear()
    observation = grab_and_preprocess_obs(car)
    steps = 0
    print(f"Episode: {i_episode}")

    while True:
        curvature_dist = run_driving_model(observation)
        curvature_action = curvature_dist.sample()[0, 0]

        # Step the simulated car with the same action
        vista_step(curvature_action)
        observation = grab_and_preprocess_obs(car)

        reward = 1.0 if not check_crash(car) else 0.0

        # add to memory
        memory.add_to_memory(observation, curvature_action, reward)
        steps += 1

        # is the episode over? did you crash or do so well that you're done?
        if reward == 0.0:
            driving_model.train() # set to train as we pass in a batch
            # determine total reward and keep a record of this
            total_reward = sum(memory.rewards)
            print(f"reward: {total_reward}")

            # execute training step - remember we don't know anything about how the
            #   agent is doing until it has crashed! if the training step is too large
            #   we need to sample a mini-batch for this step.
            batch_size = min(len(memory), max_batch_size)
            i = torch.randint(len(memory), (batch_size,), dtype=torch.long)

            batch_observations = torch.stack(memory.observations, dim=0)
            batch_observations = torch.index_select(batch_observations, dim=0, index=i)

            batch_actions = torch.stack(memory.actions)
            batch_actions = torch.index_select(
                batch_actions, dim=0, index=i
            ) 

            batch_rewards = torch.tensor(memory.rewards)
            batch_rewards = discount_rewards(batch_rewards)[i]

            running_loss = train_step(
                driving_model,
                compute_driving_loss,
                optimizer,
                observations=batch_observations,
                actions=batch_actions,
                discounted_rewards=batch_rewards,
                running_loss=running_loss,
                custom_fwd_fn=run_driving_model,
            )
            # episodic loss
            episode_loss = running_loss / batch_size
            print(f"loss: {episode_loss}")
            
            # Write reward and loss to results txt file
            f.write(f"{total_reward}\t{episode_loss}\n")
            
            # reset the memory
            memory.clear()
            break

torch.save(driving_model.state_dict(), "models/basic_cnn.pth")