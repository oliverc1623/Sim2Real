import argparse
import numpy as np
import os
# import cv2

import vista
from vista.utils import transform
from vista.entities.agents.Dynamics import tireangle2curvature
import matplotlib.pyplot as plt

def follow_human_trajectory(agent):
    action = np.array([
        agent.trace.f_curvature(agent.timestamp),
        agent.trace.f_speed(agent.timestamp)
    ])
    return action

def pure_pursuit_controller(agent, lk = 5.):
    # hyperparameters
    lookahead_dist = lk
    Kp = 3.
    dt = 5. # 1 / 30.

    # get road in ego-car coordinates
    ego_pose = agent.ego_dynamics.numpy()[:3]
    road_in_ego = np.array([
        transform.compute_relative_latlongyaw(_v[:3], ego_pose)
        for _v in agent.road
    ])

    # find (lookahead) target
    dist = np.linalg.norm(road_in_ego[:, :2], axis=1)
    dist[road_in_ego[:, 1] < 0] = 9999.  # drop road in the back
    tgt_idx = np.argmin(np.abs(dist - lookahead_dist))
    dx, dy, dyaw = road_in_ego[tgt_idx]

    # simply follow human trajectory for speed
    speed = agent.human_speed

    # compute curvature
    arc_len = speed * dt
    curvature = (Kp * np.arctan2(-dx, dy) * dt) / arc_len
    curvature_bound = [
        tireangle2curvature(_v, agent.wheel_base)
        for _v in agent.ego_dynamics.steering_bound
    ]
    curvature = np.clip(curvature, *curvature_bound)

    return np.array([curvature, speed])

def terminal_condition(world, agent):
    def _check_out_of_lane():
        road_half_width = agent.trace.road_width / 2.
        return np.abs(agent.relative_state.x) > road_half_width

    def _check_exceed_max_rot():
        maximal_rotation = np.pi / 10.
        return np.abs(agent.relative_state.theta) > maximal_rotation

    out_of_lane = _check_out_of_lane()
    exceed_max_rot = False #_check_exceed_max_rot()
    done = out_of_lane or exceed_max_rot or agent.done
    other_info = {
        'done': done,
        'out_of_lane': out_of_lane,
        'exceed_max_rot': exceed_max_rot,
    }

    return done, other_info

def reward_fn(world, agent, **kwargs):
    """ An example definition of reward function. """
    reward = -1 if kwargs['done'] else 0  # simply encourage survival

    return reward, {}

def step(world, agent, action, config, dt=1/30):
    # Step agent and get observation
    action = np.array([action[0], agent.human_speed])
    agent.step_dynamics(action, dt=dt)
    agent.step_sensors()
    observations = agent.observations

    done, info_from_terminal_condition = config[0](world, agent)

    reward, _ = config[1](world, agent, **info_from_terminal_condition)

    return observations, reward, done, info_from_terminal_condition

def main(args):
    world = vista.World(args.trace_path, trace_config={'road_width': 4})
    car = world.spawn_agent(
        config={
            'length': 5.,
            'width': 2.,
            'wheel_base': 2.78,
            'steering_ratio': 14.7,
            'lookahead_road': True,
        })
    config = [terminal_condition, reward_fn]
    display = vista.Display(world)

    # world.reset()
    # display.reset()
    lk = 100
    rl_done = False
    i = 0
    while not rl_done:
        print(f"Generation: {i}")
        print(f"Look ahead: {lk}")
        world.set_seed(47)
        world.reset()
        display.reset()
        done = False
        while not done:
            action = pure_pursuit_controller(car, lk)
            observations, reward, done, info = step(world, car, action, config, dt=1/10)
            if car.done:
                rl_done = True
                break
            if reward == -1:
                lk -= 5
            vis_img = display.render()
            plt.pause(0.05)
        i += 1

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
