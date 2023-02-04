import vista
import numpy as np
import cv2
import matplotlib.pyplot as plt
from matplotlib import animation
import os, sys
from vista.utils import transform
from vista.entities.agents.Dynamics import tireangle2curvature


trace_path = ["./trace/20210726-131322_lexus_devens_center"]
world = vista.World(trace_path, trace_config={'road_width': 4})
car = world.spawn_agent(
    config={
        'length': 5.,
        'width': 2.,
        'wheel_base': 2.78,
        'steering_ratio': 14.7,
        'lookahead_road': True
    })
camera = car.spawn_camera(config={
    'name': 'camera_front',
    'rig_path': './RIG.xml',
    'size': (600,960), # (200, 320)
})

# car2 = world.spawn_agent(config={
#         'length': 5.,
#         'width': 2.,
#         'wheel_base': 2.78,
#         'steering_ratio': 14.7,
#         'lookahead_road': True
#     })
# car2.spawn_camera(config={
#     'name': 'camera_front',
#     'rig_path': './RIG.xml',
#     'size': (600,960), # (200, 320)
# })

lidar_config = {'name': 'lidar_3d',
                'yaw_res': 0.1,
                'pitch_res': 0.1,
                'yaw_fov': (-180., 180.)}
# lidar = car.spawn_lidar(lidar_config)

event_camera_config = {'name': 'event_camera_front',
                       'rig_path': './RIG.xml',
                       'original_size': (480, 640),
                       'size': (240, 320),
                       'optical_flow_root': 'data_prep/Super-SloMo/slowmo',
                       'checkpoint': 'data_prep/Super-SloMo/ckpt/SuperSloMo.ckpt',
                       'positive_threshold': 0.1,
                       'sigma_positive_threshold': 0.02,
                       'negative_threshold': -0.1,
                       'sigma_negative_threshold': 0.02,
                       'base_camera_name': 'camera_front',
                       'base_size': (600, 960),  # (600, 960)
                       'directional_light_intensity': 50,}

event_camera = car.spawn_event_camera(event_camera_config)
display = vista.Display(world)

def follow_human_trajectory(agent):
    action = np.array([
        agent.trace.f_curvature(agent.timestamp),
        agent.trace.f_speed(agent.timestamp)
    ])
    return action

def pure_pursuit_controller(agent):
    # hyperparameters
    lookahead_dist = 100.
    Kp = 3.
    dt = 1 / 30. # difference

    # get road in ego-car coordinates
    ego_pose = agent.ego_dynamics.numpy()[:3]
    road_in_ego = np.array([
        transform.compute_relative_latlongyaw(_v[:3], ego_pose)
        for _v in agent.road
    ])

    # find (lookahead) target
    dist = np.linalg.norm(road_in_ego[:,:2], axis=1)
    dist[road_in_ego[:,1] < 0] = 9999. # drop road in the back
    tgt_idx = np.argmin(np.abs(dist - lookahead_dist))
    dx, dy, dyaw = road_in_ego[tgt_idx]

    # simply follow human trajectory for speed
    speed = agent.human_speed # 20.0 for faster demo

    # compute curvature
    arc_len = speed * dt
    curvature = (Kp * np.arctan2(-dx, dy) * dt) / arc_len
    curvature_bound = [
        tireangle2curvature(_v, agent.wheel_base)
        for _v in agent.ego_dynamics.steering_bound]
    curvature = np.clip(curvature, *curvature_bound)

    return np.array([curvature, speed])

state_space_controller = [
    follow_human_trajectory,
    pure_pursuit_controller
][0]

world.reset()
display.reset()

initial_img = display.render()
image_frames = [initial_img]
i = 0
while not car.done:
    action = state_space_controller(car)
    car.step_dynamics(action)
    car.step_sensors()
    
    sensor_data = car.observations
    
    vis_img = display.render()
    plt.pause(0.05)
    # plt.savefig(f"saved_images/sim_frames/{i:04}.png")
    i +=1
    if i % 20 == 0:
        print(f"At step: {i:04}")
plt.show()
