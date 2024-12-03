import matplotlib.pyplot as plt
import torch
import torch.nn as nn
from pathlib import Path
import numpy as np
import hcatnetwork
from PIL import Image, ImageDraw
import matplotlib.patches as patches

import cv2

from utils.ASOCA_handler.general import load_vol_lab_graph_and_align, get_slices_with_centerline
from utils.augmentation import get_grid_patches_3d, adjust_idxs
from utils.ASOCA_handler.clustering import get_subvolumes_centroid, build_centerline_per_slice_dict, floor_or_ceil
from utils.ASOCA_handler.visualize import plot_slice_with_mask_and_centers, plot_slice_with_mask

from utils.loaders import LoaderFromPath


data_path = 'ASOCA_DATASET/train/Diseased/CTCA/Diseased_1.nrrd'
crop_depth = 32
crop_size = 128

v, l, g = load_vol_lab_graph_and_align(data_path, 'ijk')

# hcatnetwork.draw.draw_centerlines_graph_3d(g)

idxs = get_slices_with_centerline(g)

start_id, last_id = adjust_idxs(v, idxs, crop_depth)

clusters_list, kij_list = get_subvolumes_centroid(g, start_id, last_id, crop_depth, closeness=64.)

f, ax = plt.subplots(2, 4)
ax = ax.flatten()

c, p = clusters_list[3], kij_list[3]

z_dict = build_centerline_per_slice_dict(g)

for i, start in enumerate(range(start_id, last_id, crop_depth)):
    plot_slice_with_mask(v, l, slice_num=start, ax=ax[i], show=False)

    c, p = clusters_list[i], kij_list[i]

    ax[i].scatter(p[:, 2], p[:, 1], c='red', s=4)
    ax[i].scatter(c[:, 1], c[:, 0], c='blue', s=5)
    for ij in c:
        top_left_x = max(ij[1] - crop_size // 2, 0)
        top_left_y = max(ij[0] - crop_size // 2, 0)

        bottom_right_x = min(ij[1] + crop_size // 2, 512)
        bottom_right_y = min(ij[0] + crop_size // 2, 512)


        rect = patches.Rectangle((top_left_x, top_left_y), bottom_right_x - top_left_x, bottom_right_y - top_left_y,
                                 linewidth=1, edgecolor='red', facecolor='none')

        ax[i].add_patch(rect)

plt.show()

