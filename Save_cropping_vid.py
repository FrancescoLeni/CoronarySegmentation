import matplotlib.pyplot as plt
import torch
import torch.nn as nn
from pathlib import Path
import numpy as np
import hcatnetwork
from PIL import Image, ImageDraw
import matplotlib.patches as patches
import imageio

import cv2

from utils.ASOCA_handler.general import load_vol_lab_graph_and_align, get_slices_with_centerline
from utils.augmentation import get_grid_patches_3d, adjust_idxs
from utils.ASOCA_handler.clustering import get_subvolumes_centroid, build_centerline_per_slice_dict, floor_or_ceil
from utils.ASOCA_handler.visualize import plot_slice_with_mask_and_centers, plot_slice_with_mask

from utils.loaders import LoaderFromPath


data_path = 'ASOCA_DATASET/test/Normal/CTCA/Normal_20.nrrd'
crop_depth = 32
crop_size = 128

v, l, g = load_vol_lab_graph_and_align(data_path, 'ijk')

# hcatnetwork.draw.draw_centerlines_graph_3d(g)

idxs = get_slices_with_centerline(g)

start_id, last_id = adjust_idxs(v, idxs, crop_depth)

clusters_list, kij_list = get_subvolumes_centroid(g, start_id, last_id, crop_depth, closeness=64.)

c, p = clusters_list[3], kij_list[3]
print(start_id, last_id)
print(len(kij_list), kij_list[0].shape)

dst = r'runs'

# Video writer setup
frame_height, frame_width = 512, 512 # Dimensions of each slice
fps = 16  # Frames per second
# video_writer = cv2.VideoWriter('output_video.mp4', cv2.VideoWriter_fourcc(*'mp4v'), fps, (frame_width, frame_height))

frames = []

crop_num = (last_id - start_id + 1) // crop_depth

cent_indxs = [i for i in range(crop_num)]
cent_indxs = [element for element in cent_indxs for _ in range(crop_depth)]


z_dict = build_centerline_per_slice_dict(g)

# only red dots
for i, (idx, start) in enumerate(zip(range(start_id, last_id, 1), cent_indxs)):
    fig, ax = plt.subplots(1, 1, figsize=(19.2, 10.8), dpi=100)
    # ax = ax.flatten()
    plot_slice_with_mask(v, l, slice_num=idx, ax=ax, show=False)

    c, p = clusters_list[start], kij_list[start]

    p_here = p[p[:,0] == idx]
    if p_here.any():
        ax.scatter(p_here[:, 2], p_here[:, 1], c='red', s=6)
    p_here = None
    # ax.scatter(c[:, 1], c[:, 0], c='blue', s=6)
    # for ij in c:
    #     top_left_x = max(ij[1] - crop_size // 2, 0)
    #     top_left_y = max(ij[0] - crop_size // 2, 0)
    #
    #     bottom_right_x = min(ij[1] + crop_size // 2, 512)
    #     bottom_right_y = min(ij[0] + crop_size // 2, 512)
    #
    #
    #     rect = patches.Rectangle((top_left_x, top_left_y), bottom_right_x - top_left_x, bottom_right_y - top_left_y,
    #                              linewidth=2, edgecolor='red', facecolor='none')
    #
    #     ax.add_patch(rect)

    # for x in range(0, 512, 128):
    #     if x != 0:
    #         ax.axvline(x, color='red', linewidth=2)  # Vertical lines
    # for y in range(0, 512, 128):
    #     if y != 0:
    #         ax.axhline(y, color='red', linewidth=2)  # Horizontal lines

    ax.axis('off')
    plt.tight_layout()

    fig.canvas.draw()
    frame = np.frombuffer(fig.canvas.buffer_rgba(), dtype=np.uint8)  # Updated to use buffer_rgba
    frame = frame.reshape(fig.canvas.get_width_height()[::-1] + (4,))  # Convert to proper shape (RGBA)
    frame = frame[:, :, :3]

    frames.append(frame)

    plt.close(fig)


# Boxes
for i, (idx, start) in enumerate(zip(range(start_id, last_id, 1), cent_indxs)):
    fig, ax = plt.subplots(1, 1, figsize=(19.2, 10.8), dpi=100)
    # ax = ax.flatten()
    plot_slice_with_mask(v, l, slice_num=idx, ax=ax, show=False)

    c, p = clusters_list[start], kij_list[start]

    p_here = p[p[:,0] == idx]
    if p_here.any():
        ax.scatter(p_here[:, 2], p_here[:, 1], c='red', s=6)
    p_here = None
    ax.scatter(c[:, 1], c[:, 0], c='blue', s=8)
    for ij in c:
        top_left_x = max(ij[1] - crop_size // 2, 0)
        top_left_y = max(ij[0] - crop_size // 2, 0)

        bottom_right_x = min(ij[1] + crop_size // 2, 512)
        bottom_right_y = min(ij[0] + crop_size // 2, 512)


        rect = patches.Rectangle((top_left_x, top_left_y), bottom_right_x - top_left_x, bottom_right_y - top_left_y,
                                 linewidth=2, edgecolor='red', facecolor='none')

        ax.add_patch(rect)

    # for x in range(0, 512, 128):
    #     if x != 0:
    #         ax.axvline(x, color='red', linewidth=2)  # Vertical lines
    # for y in range(0, 512, 128):
    #     if y != 0:
    #         ax.axhline(y, color='red', linewidth=2)  # Horizontal lines

    ax.axis('off')
    plt.tight_layout()

    fig.canvas.draw()
    frame = np.frombuffer(fig.canvas.buffer_rgba(), dtype=np.uint8)  # Updated to use buffer_rgba
    frame = frame.reshape(fig.canvas.get_width_height()[::-1] + (4,))  # Convert to proper shape (RGBA)
    frame = frame[:, :, :3]

    frames.append(frame)

    plt.close(fig)

# GRID
for i, (idx, start) in enumerate(zip(range(start_id, last_id, 1), cent_indxs)):
    fig, ax = plt.subplots(1, 1, figsize=(19.2, 10.8), dpi=100)
    # ax = ax.flatten()
    plot_slice_with_mask(v, l, slice_num=idx, ax=ax, show=False)

    c, p = clusters_list[start], kij_list[start]

    p_here = p[p[:,0] == idx]
    if p_here.any():
        ax.scatter(p_here[:, 2], p_here[:, 1], c='red', s=6)
    p_here = None
    # ax.scatter(c[:, 1], c[:, 0], c='blue', s=6)
    # for ij in c:
    #     top_left_x = max(ij[1] - crop_size // 2, 0)
    #     top_left_y = max(ij[0] - crop_size // 2, 0)
    #
    #     bottom_right_x = min(ij[1] + crop_size // 2, 512)
    #     bottom_right_y = min(ij[0] + crop_size // 2, 512)
    #
    #
    #     rect = patches.Rectangle((top_left_x, top_left_y), bottom_right_x - top_left_x, bottom_right_y - top_left_y,
    #                              linewidth=2, edgecolor='red', facecolor='none')
    #
    #     ax.add_patch(rect)

    for x in range(0, 512, 128):
        if x != 0:
            ax.axvline(x, color='red', linewidth=2)  # Vertical lines
    for y in range(0, 512, 128):
        if y != 0:
            ax.axhline(y, color='red', linewidth=2)  # Horizontal lines

    ax.axis('off')
    plt.tight_layout()

    fig.canvas.draw()
    frame = np.frombuffer(fig.canvas.buffer_rgba(), dtype=np.uint8)  # Updated to use buffer_rgba
    frame = frame.reshape(fig.canvas.get_width_height()[::-1] + (4,))  # Convert to proper shape (RGBA)
    frame = frame[:, :, :3]

    frames.append(frame)

    plt.close(fig)

imageio.mimwrite(dst+'\everything_N20.mp4', frames, fps=fps, codec='libx264')

print("Video saved")




