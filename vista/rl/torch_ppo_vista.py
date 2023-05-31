import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from torch.distributions import Normal
import vista
import os
from vista_helper import *
from replay_memory import Memory

device = ("cuda" if torch.cuda.is_available() else "mps" if torch.backends.mps.is_available() else "cpu")
device = torch.device(device)
print(f"Using {device} device")

#Hyperparameters
learning_rate  = 0.0005
gamma           = 0.95
lmbda           = 0.9
eps_clip        = 0.2
K_epoch         = 10
rollout_len    = 3
buffer_size    = 3
minibatch_size = 32

class PPO(nn.Module):
    def __init__(self):
        super(PPO, self).__init__()
        self.data = []
        
        self.conv1 = nn.Conv2d(3, 32, kernel_size=3, stride=1, padding=1)
        self.norm1 = nn.GroupNorm(8, 32)
        self.relu1 = nn.ReLU()
        self.conv2 = nn.Conv2d(32, 64, kernel_size=3, stride=1, padding=1)
        self.norm2 = nn.GroupNorm(16, 64)
        self.relu2 = nn.ReLU()
        self.conv3 = nn.Conv2d(64, 128, kernel_size=3, stride=1, padding=1)
        self.norm3 = nn.GroupNorm(32, 128)
        self.relu3 = nn.ReLU()
        self.conv4 = nn.Conv2d(128, 256, kernel_size=3, stride=1, padding=1)
        self.norm4 = nn.GroupNorm(64, 256)
        self.relu4 = nn.ReLU()
        self.conv5 = nn.Conv2d(256, 2, kernel_size=3, stride=1, padding=1)
        self.norm5 = nn.GroupNorm(1, 2)
        self.relu5 = nn.ReLU()
        self.fc = nn.Linear(2 * 32 * 30, 2)
        self.fc_v = nn.Linear(3 * 32 * 20, 1)

        self.optimizer = optim.Adam(self.parameters(), lr=learning_rate)
        self.optimization_step = 0

    def pi(self, x):
        single_image_input = len(x.shape) == 3  # missing 4th batch dimension
        if single_image_input:
            x = x.unsqueeze(0)
            x = x.permute(0, 3, 1, 2)
        else:
            x = x.view(x.shape[0] * x.shape[1], 3, 30, 32)
        x = self.relu1(self.norm1(self.conv1(x)))
        x = self.relu2(self.norm2(self.conv2(x)))
        x = self.relu3(self.norm3(self.conv3(x)))
        x = self.relu4(self.norm4(self.conv4(x)))
        x = self.relu5(self.norm5(self.conv5(x)))
        x = x.reshape(x.size(0), -1)
        x = self.fc(x)

        mu, log_sigma = torch.chunk(x, 2, dim=-1)
        mu = 1/8.0 * torch.tanh(mu)  # conversion
        sigma = 0.1 * torch.sigmoid(log_sigma) + 0.005  # conversion
        return mu, sigma
    
    def v(self, x):
        x = x.view(x.shape[0]*x.shape[1], 3, 30, 32)
        x = self.relu1(self.norm1(self.conv1(x)))
        x = self.relu2(self.norm2(self.conv2(x)))
        x = self.relu3(self.norm3(self.conv3(x)))
        x = self.relu4(self.norm4(self.conv4(x)))
        x = self.relu5(self.norm5(self.conv5(x)))
        x = x.reshape(x.size(0), -1)
        v = self.fc_v(x)
        return v
      
    def put_data(self, transition):
        self.data.append(transition)
        
    def make_batch(self):
        s_batch, a_batch, r_batch, s_prime_batch, prob_a_batch, done_batch = [], [], [], [], [], []
        data = []

        for j in range(buffer_size):
            for i in range(minibatch_size):
                rollout = self.data.pop()
                s_lst, a_lst, r_lst, s_prime_lst, prob_a_lst, done_lst = [], [], [], [], [], []

                for transition in rollout:
                    s, a, r, s_prime, prob_a, done = transition
                    
                    s_lst.append(s)
                    a_lst.append([a])
                    r_lst.append([r])
                    s_prime_lst.append(s_prime)
                    prob_a_lst.append([prob_a])
                    done_mask = 0 if done else 1
                    done_lst.append([done_mask])

                s_batch.append(s_lst)
                a_batch.append(a_lst)
                r_batch.append(r_lst)
                s_prime_batch.append(s_prime_lst)
                prob_a_batch.append(prob_a_lst)
                done_batch.append(done_lst)

            flatten_s_batch = [item for sublist in s_batch for item in sublist]
            flatten_s_batch = torch.stack(flatten_s_batch).view(len(s_batch), 3, 30, 32, 3)

            flatten_s_prime_batch = [item for sublist in s_prime_batch for item in sublist]
            # to shape (mini_batch_size, rollout_len, 30, 32, 3)
            flatten_s_prime_batch = torch.stack(flatten_s_prime_batch).view(len(s_batch), 3, 30, 32, 3)

            r = torch.tensor(r_batch, dtype=torch.float).to(device)
            r = r.view(r.shape[0]*r.shape[1], 1)
            d = torch.tensor(done_batch, dtype=torch.float).to(device)
            d = d.view(d.shape[0]*d.shape[1], 1)

            mini_batch = flatten_s_batch, torch.tensor(a_batch, dtype=torch.float).to(device), \
                          r, flatten_s_prime_batch, \
                          d, torch.tensor(prob_a_batch, dtype=torch.float).to(device)
            data.append(mini_batch)

        return data

    def calc_advantage(self, data):
        data_with_adv = []
        for mini_batch in data:
            s, a, r, s_prime, done_mask, old_log_prob = mini_batch
            with torch.no_grad():
                td_target = r + gamma * self.v(s_prime) * done_mask
                delta = td_target - self.v(s)
            delta = delta.cpu().numpy() 

            advantage_lst = []
            advantage = 0.0
            for delta_t in delta[::-1]:
                advantage = gamma * lmbda * advantage + delta_t[0]
                advantage_lst.append([advantage])
            advantage_lst.reverse()
            advantage = torch.tensor(advantage_lst, dtype=torch.float)
            data_with_adv.append((s, a, r, s_prime, done_mask, old_log_prob, td_target, advantage))

        return data_with_adv

    def train_net(self):
        if len(self.data) == minibatch_size * buffer_size:
            data = self.make_batch()
            data = self.calc_advantage(data)

            for i in range(K_epoch):
                for mini_batch in data:
                    s, a, r, s_prime, done_mask, old_log_prob, td_target, advantage = mini_batch
                    a = a.view(a.shape[0] * a.shape[1], 1)
                    mu, std = self.pi(s)
                    dist = Normal(mu, std)
                    log_prob = dist.log_prob(a)
                    old_log_prob = old_log_prob.view(old_log_prob.shape[0]*old_log_prob.shape[1], 1)
                    ratio = torch.exp(log_prob - old_log_prob)  # a/b == exp(log(a)-log(b))

                    surr1 = ratio * advantage.to(device)
                    surr2 = torch.clamp(ratio, 1-eps_clip, 1+eps_clip) * advantage.to(device)
                    loss = -torch.min(surr1, surr2) + F.smooth_l1_loss(self.v(s) , td_target)

                    self.optimizer.zero_grad()
                    loss.mean().backward()
                    nn.utils.clip_grad_norm_(self.parameters(), 1.0)
                    self.optimizer.step()
                    self.optimization_step += 1

def collect_samples(mini_batch_size, model, world, display, car, camera):
    log = dict()
    memory = Memory()
    num_steps = 0
    total_reward = 0
    num_episodes = 0
    while num_steps < mini_batch_size:
        vista_reset(world, display)
        s = grab_and_preprocess_obs(car, camera)
        reward_episode = 0
        for t in range(10000):
            mu, std = model.pi(s)
            dist = Normal(mu, std)
            a = dist.sample()
            log_prob = dist.log_prob(a)
            vista_step(car, curvature = a.item())
            s_prime = grab_and_preprocess_obs(car, camera).to(device)
            r = calculate_reward(car)
            reward_episode += r
            crash = check_crash(car)
            mask = 0 if crash else 1
            memory.push(s, a, mask, s_prime, r)
            if crash:
                break
            s = s_prime
        # log stats
        num_steps += (t + 1)
        num_episodes += 1
    log['num_steps'] = num_steps
    log['num_episodes'] = num_episodes
    log['total_reward'] = total_reward
    log['avg_reward'] = total_reward / num_episodes
    return memory, log

def main():
    # Set up VISTA simulator
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
    display = vista.Display(
        world, display_config={"gui_scale": 2, "vis_full_frame": False}
    )

    model = PPO().to(device)
    score = 0.0
    print_interval = 1
    rollout = []

    for n_epi in range(500):
        batch = collect_samples(minibatch_size, model, world, display, car, camera)

if __name__ == '__main__':
    main()