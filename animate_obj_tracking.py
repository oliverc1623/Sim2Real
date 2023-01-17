import matplotlib.pyplot as plt
from matplotlib import animation
import numpy as np
import cv2
import os
import sys
import argparse
# from IPython.display import HTML
# plt.rcParams['animation.embed_limit'] = 2**128

def canny_edge(image):
    gray_conversion= cv2.cvtColor(image, cv2.COLOR_RGB2GRAY)
    blur_conversion = cv2.GaussianBlur(gray_conversion, (5,5),0)
    canny_conversion = cv2.Canny(blur_conversion, 50,150)
    return canny_conversion


def get_annotations(scene_number, isReal = True):
    if isReal:
        label_file = f"kitti/training/label_02/00{scene_number:02}.txt"

def get_scene_frames(scene_number = 1, isCannyEdge = False, annotate = False):
    kitti_scene = f"kitti/training/image_02/00{scene_number:02}/"
    vkitti_scene = f"vkitti/Scene{scene_number:02}/clone/frames/rgb/Camera_0/"
    sorted_rframe_files = sorted(os.listdir(kitti_scene))
    sorted_vframe_files = sorted(os.listdir(vkitti_scene))
    # rannotations = 
    frames = []
    for (rf, vf) in zip(sorted_rframe_files, sorted_vframe_files):
        # open real KITTI frame
        frame = cv2.imread(kitti_scene + rf)
        frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        # open virtual KITTI frame
        vframe = cv2.imread(vkitti_scene + vf)
        vframe = cv2.cvtColor(vframe, cv2.COLOR_BGR2RGB)
        if scene_number == 18:
            vframe = vframe[:374,:1238]
        elif scene_number == 20:
            vframe = vframe[:374,:1241]
        # concatenate frames
        pair = np.concatenate((frame, vframe))
        if isCannyEdge:
            pair = canny_edge(pair)
        frames.append(pair)
    return frames

def animate_scene(frames, frame_frequency = 1):
    fig, ax = plt.subplots()
    ims = []
    for f in frames[::frame_frequency]:
        im = ax.imshow(f, animated=True)
        ims.append([im])
    ani = animation.ArtistAnimation(fig, ims, interval=100, blit=True,
                                repeat_delay=1000)
    return ani

def main(args):
    frames = get_scene_frames(scene_number = args.scene_number,
                             isCannyEdge = args.canny,
                             annotate = args.annotate)
    ani = animate_scene(frames, args.frame_freq)
    writer = animation.PillowWriter(fps=15, metadata=dict(artist='Me'), bitrate=1800)
    ani.save(args.outputfile + ".gif", writer=writer)
    
if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Visualize Virtual and Real KITTI Scenes")
    parser.add_argument('--scene_number', '-sn', help='Scene Number', type=int, required=False, default=1)
    parser.add_argument('--pair', '-p', help='Pair Scenes for comparison', action=argparse.BooleanOptionalAction)
    parser.add_argument('--canny', '-c', help='Apply Canny Edge Detection', action=argparse.BooleanOptionalAction)
    parser.add_argument('--frame_freq', '-ff', help='Frame Frequency', type=int, required=False, default=1)
    parser.add_argument('--annotate', '-a', help='Apply BBoxes to Objects', action=argparse.BooleanOptionalAction)
    parser.add_argument('--outputfile', '-o', help='Output filename', default="movie")
    args = parser.parse_args()
    main(args)