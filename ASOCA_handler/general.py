import hcatnetwork
import os
import nrrd
import math

from HearticDatasetManager.asoca import AsocaImageCT
from HearticDatasetManager.asoca.dataset import DATASET_ASOCA_IMAGES_DICT
from HearticDatasetManager.affine import apply_affine_3d

import matplotlib.pyplot as plt
import numpy as np
from sklearn.cluster import DBSCAN


def min_max(x):
    max = x.max()
    min = x.min()

    return (x - min) / (max - min)


def floor_or_ceil(value):
    decimal_part = value - math.floor(value)

    if decimal_part < 0.5:
        return math.floor(value)
    else:
        return math.ceil(value)


def load_single_volume(path_to_ASOCA, folder, vol_n):
    """
    args:
        - path_to_ASOCA: path to ASOCA folder
        - folder: 'Normal' or 'Diseased'
        - vol_n: the number relative to the volume to load ([0, 19] becouse 21 images are provided)

    return:
        - AsocaImageCT obj  (HearticDatasetManager)
        - relative labels np.array volume

    """

    assert vol_n in range(0, 20)
    assert folder in ['Diseased', 'Normal']

    asoca_graph_file = os.path.join(
        path_to_ASOCA,
        DATASET_ASOCA_IMAGES_DICT[folder][vol_n]
    )

    image = AsocaImageCT(asoca_graph_file)
    lab_path = image.path.replace('CTCA', 'Annotations')
    labels, _ = nrrd.read(lab_path)

    return image, labels


def load_centerline(path_to_centerline):
    graph_hcatnetwork = hcatnetwork.io.load_graph(path_to_centerline, output_type=hcatnetwork.graph.SimpleCenterlineGraph)
    return graph_hcatnetwork


def align_centerline_to_image(image, centerline_hcatgraph, coord):
    """

    args:
        - image: AsocaImageCT obj image
        - centerline_hcatgraph: hcatnetwork.graph.SimpleCenterlineGraph centerline
        - coord ('RAS', 'ijk'): whether to align to RAS coords. or image_space (ijk) coords.
    return:
        - transformed centerline in same image coord. space (RAS)
    """

    assert coord in ['RAS', 'ijk']

    transform = image.affine_centerlines2ras

    for n in centerline_hcatgraph.nodes:
        x = centerline_hcatgraph.nodes[n]['x']
        y = centerline_hcatgraph.nodes[n]['y']
        z = centerline_hcatgraph.nodes[n]['z']

        xyz = np.array([x, y, z])

        xyz_ras = apply_affine_3d(transform, xyz.T).T.squeeze()

        if coord == 'RAS':
            xyz_out = xyz_ras
        elif coord == 'ijk':
            t = image.affine_ras2ijk
            xyz_out = apply_affine_3d(t, xyz_ras.T).T.squeeze()
        else:
            raise AttributeError(f'"{coord}" name not valid')

        centerline_hcatgraph.nodes[n]['x'] = xyz_out[0]
        centerline_hcatgraph.nodes[n]['y'] = xyz_out[1]
        centerline_hcatgraph.nodes[n]['z'] = xyz_out[2]

    return centerline_hcatgraph


def build_centerline_per_slice_dict(ijkgraph):
    """

    args:
        - ijkgraph: centerline graph in ijk (image_space) coordinates (VERY IMPORTANT!!!!)
    return:
        - a dictionary containings the ids of all the nodes of the line seenable per slice {slice: [ids...]}
    """

    z_dict = {}

    for node_id, data in ijkgraph.nodes(data=True):
        # Floor-round the 'z' value
        z_floored = math.floor(data['z'])

        # Add the node to the dictionary
        if z_floored not in z_dict:
            z_dict[z_floored] = []  # Initialize the list if the key doesn't exist
        z_dict[z_floored].append(node_id)

    return {k: z_dict[k] for k in sorted(z_dict.keys())}


def get_slices_with_centerline(ijkgraph):
    z_dict = build_centerline_per_slice_dict(ijkgraph)

    idx = list(z_dict.keys())

    return idx

