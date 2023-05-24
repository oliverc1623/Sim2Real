import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import vista
import os
import torch
from torch import nn
from tqdm import tqdm
import numpy as np
import torch
import torch.nn.functional as F
import torch.distributions as dist
import torch.optim as optim
import time
import datetime
import resnet
import rnn
# from torch.optim.lr_scheduler import lr_scheduler
import torchvision
import torch.nn.functional as F
import importlib
import torchvision.transforms as transforms
from PIL import Image
import cv2
import mycnn

device = ("cuda" if torch.cuda.is_available() else "mps" if torch.backends.mps.is_available() else "cpu")
device = torch.device(device)
print(f"Using {device} device")

models = {"ResNet18": resnet.ResNet18, 
          "ResNet34": resnet.ResNet34, 
          "ResNet50": resnet.ResNet50, 
          "ResNet101": resnet.ResNet101,
          "rnn": rnn.MyRNN,
          "CNN": mycnn.CNN,
          "LSTM": rnn.LSTMLaneFollower}

### Agent Memory ###
class Memory:
    def __init__(self):
        self.clear()

    # Resets/restarts the memory buffer
    def clear(self):
        self.observations = []
        self.actions = []
        self.rewards = []
        self.log_probs = []

    # Add observations, actions, rewards to memory
    def add_to_memory(self, new_observation, new_action, new_reward, new_log_prob):
        self.observations.append(new_observation)
        self.actions.append(new_action)
        self.rewards.append(new_reward)
        self.log_probs.append(new_log_prob)

    def __len__(self):
        return len(self.actions)

### Learner Class ###
class Learner:
    def __init__(
        self, model_name, learning_rate, episodes, clip, animate, algorithm, max_curvature=1/8.0, max_std=0.1
    ) -> None:
        # Set up VISTA simulator
        trace_root = "../trace"
        trace_path = [
            "20210726-154641_lexus_devens_center",
            "20210726-155941_lexus_devens_center_reverse",
            "20210726-184624_lexus_devens_center",
            "20210726-184956_lexus_devens_center_reverse",
        ]
        trace_path = [os.path.join(trace_root, p) for p in trace_path]
        self.world = vista.World(trace_path, trace_config={"road_width": 4})
        self.car = self.world.spawn_agent(
            config={
                "length": 5.0,
                "width": 2.0,
                "wheel_base": 2.78,
                "steering_ratio": 14.7,
                "lookahead_road": True,
            }
        )
        self.camera = self.car.spawn_camera(config={"size": (200, 320)})
        self.display = vista.Display(
            self.world, display_config={"gui_scale": 2, "vis_full_frame": False}
        )
        self.model_name = model_name
        self.driving_model = models[model_name]().to(device)
        self.max_batch_size = 300
        self.max_reward = float("-inf")  # keep track of the maximum reward acheived during training

        # hyperparameters
        self.learning_rate = learning_rate
        self.max_curvature = max_curvature
        self.max_std = max_std
        self.episodes = episodes
        self.clip = clip
        self.eps_clip = 0.2

        self.animate = animate
        self.algorithm = algorithm
        self._write_file_()

    def _write_file_(self):
        # open and write to file to track progress
        now = datetime.datetime.now()
        self.timestamp = now.strftime("%Y-%m-%d_%H-%M-%S")
        results_dir = "results"
        model_results_dir = results_dir + f"/{self.model_name}/"

        frames_dir = "frames"
        self.model_frame_dir = (
            frames_dir + f"/{self.model_name}_frames_{self.timestamp}/"
        )
        if self.animate and not os.path.exists(self.model_frame_dir):
            os.makedirs(self.model_frame_dir)

        if not os.path.exists(model_results_dir):
            os.makedirs(model_results_dir)
        filename = f"{self.model_name}_{self.algorithm}_{self.learning_rate}_{self.clip}_results_{self.timestamp}.txt"
        # Define the file path
        file_path = os.path.join(model_results_dir, filename)
        self.f = open(file_path, "w")
        self.f.write("reward\tsteps\n")

    def _vista_reset_(self):
        self.world.reset()
        self.display.reset()

    def _vista_step_(self, curvature=None, speed=None):
        if curvature is None:
            curvature = self.car.trace.f_curvature(self.car.timestamp)
        if speed is None:
            speed = self.car.trace.f_speed(self.car.timestamp)

        self.car.step_dynamics(action=np.array([curvature, speed]), dt=1 / 15.0)
        self.car.step_sensors()

    ### Reward function ###
    def _normalize_(self, x):
        x -= torch.mean(x)
        x /= torch.std(x)
        return x

    # Compute normalized, discounted, cumulative rewards (i.e., return)
    def _discount_rewards_(self, rewards, gamma=0.95):
        discounted_rewards = torch.zeros_like(rewards)
        R = 0
        for t in reversed(range(0, len(rewards))):
            # update the total discounted reward
            R = R * gamma + rewards[t]
            discounted_rewards[t] = R

        return self._normalize_(discounted_rewards)

    # Check if in terminal state
    def _check_out_of_lane_(self):
        distance_from_center = np.abs(self.car.relative_state.x)
        road_width = self.car.trace.road_width
        half_road_width = road_width / 2
        return distance_from_center > half_road_width

    def _check_exceed_max_rot_(self):
        maximal_rotation = np.pi / 10.0
        current_rotation = np.abs(self.car.relative_state.yaw)
        return current_rotation > maximal_rotation

    def _check_crash_(self):
        return (self._check_out_of_lane_() or self._check_exceed_max_rot_() or self.car.done)

    ## Data preprocessing functions ##
    def _preprocess_(self, full_obs):
        # Extract ROI
        i1, j1, i2, j2 = self.camera.camera_param.get_roi()
        obs = full_obs[i1:i2, j1:j2]
        return obs
    
    def _resize_image(self, img):
        resized_img = cv2.resize(img, (32, 30))
        return resized_img

    def _grab_and_preprocess_obs_(self, augment=True):
        full_obs = self.car.observations[self.camera.name]
        cropped_obs = self._preprocess_(full_obs)
        resized_obs = self._resize_image(cropped_obs)
        if augment:
            augmented_obs = self._augment_image(resized_obs)
            return augmented_obs.to(torch.float32)
        else:
            resized_obs_torch = resized_obs / 255.0
            return resized_obs, torch.from_numpy(resized_obs_torch).to(torch.float32).to(device)

    ## The self-driving learning algorithm ##
    def _run_driving_model_(self, image):
        torch.autograd.set_detect_anomaly(True)
        single_image_input = len(image.shape) == 3  # missing 4th batch dimension
        if single_image_input:
            image = image.unsqueeze(0)

        image = image.permute(0, 3, 1, 2)
        if self.model_name == "LSTM" or self.model_name == "resnet":
            image = image.unsqueeze(0)

        mu, logsigma = self.driving_model(image)
        mu = self.max_curvature * torch.tanh(mu)  # conversion
        sigma = self.max_std * torch.sigmoid(logsigma) + 0.005  # conversion

        pred_dist = dist.Normal(mu, sigma)
        return pred_dist
    
    def update_params(self, memory, K, optimizer):
        # Enable anomaly detection
        fixed_log_probs = torch.stack(memory.log_probs)
        batch_observations = torch.stack(memory.observations, dim=0)
        batch_actions = torch.stack(memory.actions).to(device)
        batch_rewards = torch.tensor(memory.rewards).to(device)
        batch_rewards = self._discount_rewards_(batch_rewards)
        
        batch_size = min(len(memory), self.max_batch_size)
        i = torch.randperm(len(memory))[:batch_size].to(device)

        fixed_log_probs_batch = torch.index_select(fixed_log_probs, dim=0, index=i)

        batch_observations = torch.index_select(batch_observations, dim=0, index=i)
        batch_actions = torch.index_select(batch_actions, dim=0, index=i)
        batch_rewards = batch_rewards[i]

        dist = self._run_driving_model_(batch_observations)
        log_probs = dist.log_prob(batch_actions)

        ratio = torch.exp(log_probs - fixed_log_probs_batch)
        surr1 = ratio * batch_rewards
        surr2 = torch.clamp(ratio, 1-self.eps_clip, 1+self.eps_clip) * batch_rewards
        loss = (-torch.min(surr1, surr2)).mean()

        optimizer.zero_grad()
        loss.backward()
        nn.utils.clip_grad_norm_(self.driving_model.parameters(), self.clip)
        optimizer.step()

    def learn(self):
        self._vista_reset_()
        optimizer = optim.Adam(self.driving_model.parameters(), lr=self.learning_rate, weight_decay=1e-5)
        running_loss = 0
        datasize = 0
        memory = Memory()
        K = 10
        
        ## Driving training! Main training block. ##
        for i_episode in range(self.episodes):
            # Restart the environment
            self._vista_reset_()
            # self.world.set_seed(47)
            memory.clear()
            _, observation = self._grab_and_preprocess_obs_(augment=False)     
            steps = 0
            print(f"Episode: {i_episode}")

            while True:
                self.driving_model.eval()
                curvature_dist = self._run_driving_model_(observation)
                memory_action = curvature_dist.sample()[0,0]
                curvature_action = memory_action.cpu().detach()
                log_prob = curvature_dist.log_prob(memory_action)
                self._vista_step_(curvature_action)
                np_obs, observation = self._grab_and_preprocess_obs_(augment=False)
                
                q_lat = np.abs(self.car.relative_state.x)
                road_width = self.car.trace.road_width
                z_lat = road_width / 2
                lane_reward = torch.round(torch.tensor(1 - (q_lat/z_lat)**2, dtype=torch.float32), decimals=3)
                reward = lane_reward if not self._check_crash_() else 0.0

                memory.add_to_memory(observation, memory_action, reward, log_prob)
                steps += 1

                if reward == 0.0:
                    self.driving_model.train()
                    total_reward = sum(memory.rewards)
                    print(f"steps: {steps}")
                    self.update_params(memory, K, optimizer)

                    self.f.write(f"{total_reward}\t{steps}\n")

                    # reset the memory
                    memory.clear()

                    # Check gradients norms
                    total_norm = 0
                    for p in self.driving_model.parameters():
                        param_norm = p.grad.data.norm(2) # calculate the L2 norm of gradients
                        total_norm += param_norm.item() ** 2 # accumulate the squared norm
                    total_norm = total_norm ** 0.5 # take the square root to get the total norm
                    print(f"Total gradient norm: {total_norm}")

                    break

    def save(self):
        torch.save(self.driving_model.state_dict(), f"models/{self.model_name}_{self.timestamp}_.pth")