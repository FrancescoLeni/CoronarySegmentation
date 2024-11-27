import networkx as nx
import hcatnetwork
import os
import nrrd
import copy
import math

from HearticDatasetManager.asoca import AsocaImageCT
from HearticDatasetManager.asoca.dataset import DATASET_ASOCA_IMAGES_DICT
from HearticDatasetManager.affine import apply_affine_3d

import matplotlib.pyplot as plt
import numpy as np

from ASOCA_handler.visualize import plot_centerline_over_CT, plot_slice_with_mask_and_centers
from ASOCA_handler.general import load_centerline, load_single_volume, align_centerline_to_image, floor_or_ceil, \
                                  build_centerline_per_slice_dict


# Path to your .gml file
file_path = "ASOCA/Diseased/Centerlines_graphs/Diseased_1_0.5mm.GML"

asoca_path = 'ASOCA'

graph = load_centerline(file_path)

image, labs = load_single_volume(asoca_path, 'Diseased', 0)

# print(image.name)
# print(image.path)
# print(image.data.shape) # the actual CT data in (i, j, k) (i ~ x, k ~ y, k ~ z)
# print(image.bounding_box)
# print(image.origin)     # In the RAS coordinate system, this is the origin of the image
# print(image.spacing)    # Pixel spacing in the x, y and z directions (in mm)

graph = align_centerline_to_image(image, graph, 'RAS')

# whole 3D plot with bound slices
plot_centerline_over_CT(image, graph)

# single sclice plot
# plot_slice_with_mask_and_centers(image, labs, graph, 50)
