import matplotlib.pyplot as plt
import vista
import os
from NeuralNetwork import NeuralNetwork, run_driving_model, compute_driving_loss, get_model
import torch
from torch import nn
from memory import Memory
from tqdm import tqdm
import numpy as np
from data_preprocessing import *
from terminal import *
from reward import *

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


def vista_reset():
    world.reset()
    display.reset()


vista_reset()


### Training step (forward and backpropagation) ###


def train_step(
    model,
    loss_function,
    optimizer,
    observations,
    actions,
    discounted_rewards,
    running_loss,
    loss_file,
    custom_fwd_fn=None,
):
    with torch.enable_grad():
        # zero the parameter gradients
        optimizer.zero_grad()
        # Forward propagate through the agent network
        prediction = custom_fwd_fn(observations, model)
        loss = loss_function(prediction, actions, discounted_rewards)
        loss.backward()
        running_loss += loss.item()
        optimizer.step()
        loss_file.write(str(loss.item())+"\n")


# instantiate driving agent
driving_model = nn.Sequential(
            nn.Conv2d(3, 32, 5),
            nn.SiLU(),
            nn.Conv2d(32, 48, 5),
            nn.SiLU(),
            nn.Conv2d(48, 64, 3),
            nn.SiLU(),
            nn.Conv2d(64, 64, 3),
            nn.SiLU(),
            nn.Flatten(),
            nn.Linear(302016, 128),
            nn.SiLU(),
            nn.Linear(128, 2)
        )  #NeuralNetwork()
driving_model.train()
print(driving_model)

learning_rate = 5e-4
optimizer = torch.optim.Adam(driving_model.parameters(), lr=learning_rate)

# hyperparameters
max_curvature = 1/8. 
max_std = 0.1 

loss_file = open("loss.txt", "w")
loss_file.write(f"loss,steps\n")

memory = Memory()

## Driving training! Main training block. ##

max_batch_size = 300
max_reward = float("-inf")  # keep track of the maximum reward acheived during training

if hasattr(tqdm, "_instances"):
    tqdm._instances.clear()  # clear if it exists
running_loss = 0.0
for i_episode in range(500):
    print(f"Episode: {i_episode}")
    # plotter.plot(smoothed_reward.get())
    # Restart the environment
    vista_reset()
    memory.clear()
    observation = grab_and_preprocess_obs(car, camera)
    steps = 0
    while True:
        curvature_dist = run_driving_model(observation, driving_model)
        curvature_action = curvature_dist.sample()[0, 0]
        vista_step(curvature_action)
        observation = grab_and_preprocess_obs(car, camera)
        reward = 1.0 if not check_crash(car) else 0.0

        # add to memory
        memory.add_to_memory(observation, curvature_action, reward)
        steps += 1

        if reward == 0.0:
            total_reward = sum(memory.rewards)

            batch_size = min(len(memory), max_batch_size)
            i = np.random.choice(len(memory), batch_size, replace=False)
            observation_batch = [memory.observations[indx] for indx in i]
            observation_batch = torch.stack(observation_batch, 0)

            # zero the parameter gradients
            optimizer.zero_grad()
            with torch.no_grad():
                # Forward propagate through the agent network
                observation_batch = observation_batch.permute(0,3,1,2)
                prediction = driving_model(observation_batch)
                mu, logsigma = torch.split(prediction, 1, dim=1)
                mu = max_curvature * torch.tanh(mu) # conversion
                sigma = max_std * torch.sigmoid(logsigma) + 0.005 # conversion
                pred_dist = torch.distributions.normal.Normal(loc=mu, scale=sigma)
                actions = torch.from_numpy(np.array(memory.actions)[i])
                discounted_rewards = discount_rewards(memory.rewards)[i],
            with torch.enable_grad():
                loss = compute_driving_loss(pred_dist, actions, discounted_rewards)                
                loss.backward()
                running_loss += loss.item()
                optimizer.step()

            # reset the memory
            memory.clear()
            break
    print(f"Epoch{i_episode}, running loss: {running_loss:.4f}")
    loss_file.write(f"{str(running_loss)}, {steps}\n")

