from matplotlib.animation import FuncAnimation
import matplotlib.pyplot as plt
import numpy as np
import os
import argparse
import cv2


def main(args):
    frames = os.listdir(args.frames_folder)
    frames.sort()
    image_frames = []
    for f in frames:
        img_name = args.frames_folder + '/' + f
        img = cv2.imread(img_name)
        img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
        image_frames.append(img)

    fig, ax = plt.subplots()
    ln = plt.imshow(image_frames[0])
    plt.axis('off')
    def init():
        ln.set_data(image_frames[0])
        return [ln]

    def update(frame):
        ln.set_array(frame)
        return [ln]

    ani = FuncAnimation(fig, update, image_frames, init_func=init, blit=True)
    ani.save("movie.mp4", fps=20)

   
if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Animator to visualize VISTA frames")
    parser.add_argument('--frames_folder')
    args = parser.parse_args()
    main(args)