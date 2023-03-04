import torch
import numpy as np

## Data preprocessing functions ##


def preprocess(full_obs, camera):
    # Extract ROI
    i1, j1, i2, j2 = camera.camera_param.get_roi()
    obs = full_obs[i1:i2, j1:j2]

    # Rescale to [0, 1]
    obs = obs / 255.0
    return obs


def grab_and_preprocess_obs(car, camera):
    full_obs = car.observations[camera.name]
    obs = preprocess(full_obs, camera)
    obs = np.float32(obs)
    obs = torch.from_numpy(obs)
    
    return obs
