import argparse
import numpy as np
import os
import cv2

import vista
from vista.utils import transform
from vista.entities.agents.Dynamics import tireangle2curvature
import matplotlib.pyplot as plt


def main(args):
    world = vista.World(args.trace_path, trace_config={'road_width': 4})
    car = world.spawn_agent(
        config={
            'length': 5.,
            'width': 2.,
            'wheel_base': 2.78,
            'steering_ratio': 14.7,
            'lookahead_road': True
        })
    event_camera_config = {
        'name': 'event_camera_front',
        'rig_path': './RIG.xml',
        'original_size': (480, 640),
        'size': (240, 320),
        'optical_flow_root': 'data_prep/Super-SloMo/slowmo', # data_prep/Super-SloMo/slowmo
        'checkpoint': 'data_prep/Super-SloMo/ckpt/SuperSloMo.ckpt',
        'positive_threshold': 0.1,
        'sigma_positive_threshold': 0.02,
        'negative_threshold': -0.1,
        'sigma_negative_threshold': 0.02,
        'base_camera_name': 'camera_front',
        'base_size': (600, 960),
        'directional_light_intensity': 50,
    }
    event_camera = car.spawn_event_camera(event_camera_config)
    display = vista.Display(world)

    world.reset()
    display.reset()

    while not car.done:
        action = follow_human_trajectory(car)
        car.step_dynamics(action)
        car.step_sensors()

        vis_img = display.render()
        # cv2.imshow('Visualize event data', vis_img[:, :, ::-1])
        # cv2.waitKey(20)
        plt.pause(0.05)


def follow_human_trajectory(agent):
    action = np.array([
        agent.trace.f_curvature(agent.timestamp),
        agent.trace.f_speed(agent.timestamp)
    ])
    return action


if __name__ == '__main__':
    # Parse Arguments
    parser = argparse.ArgumentParser(
        description='Run the simulator with random actions')
    parser.add_argument('--trace-path',
                        type=str,
                        nargs='+',
                        help='Path to the traces to use for simulation')
    args = parser.parse_args()

    main(args)
