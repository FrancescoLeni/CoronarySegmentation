import hcatnetwork
import numpy as np
import matplotlib.pyplot as plt
import nrrd
from .general import min_max, floor_or_ceil, build_centerline_per_slice_dict

def plot_centerline_over_CT(image, graph_RAS):
    """
        !!!!! NEEDS GRAPH IN RAS COORDINATES !!!!!
    args:
        - image: AsocaImageCT obj image
        - graph_RAS: hcatnetwork.graph.SimpleCenterlineGraph graph in "RAS" coord. (IMPORTANT !!!!)

    """

    lab_path = image.path.replace('CTCA', 'Annotations')
    data, _ = nrrd.read(lab_path)

    idx = data[:, :, :].reshape(512*512, -1).max(axis=0) == 1
    idx_nums = np.where(idx)
    idx_nums = idx_nums[0]

    xyz_spacing = image.spacing

    fig, ax = plt.subplots(1, 1, subplot_kw={"projection": "3d"})
    xl, yl, zl = image.bounding_box.get_xlim(), image.bounding_box.get_ylim(), image.bounding_box.get_zlim()
    ax.set_xlim([xl[0] - 10, xl[1] + 10])
    ax.set_ylim([yl[0] - 10, yl[1] + 10])
    ax.set_zlim([zl[0] - 10, zl[1] + 10])
    ax.add_artist(image.bounding_box.get_artist())
    ax.scatter(image.origin[0], image.origin[1], image.origin[2], c="r")

    first = idx_nums[0] * xyz_spacing[2] + zl[0]  # first slice with coronary visible
    last = idx_nums[-1] * xyz_spacing[2] + zl[0]  # last slice with coronary visible

    for zs in [first, last]:
        points_to_sample = []
        for xs in range(-110, 110, 2):
            for ys in range(-70, 130, 2):
                points_to_sample.append([xs, ys, zs])
        points_to_sample = np.array(points_to_sample)  # N x 3
        samples = image.sample(points_to_sample.T, interpolation="nearest")
        # plot them
        ax.scatter(points_to_sample[:, 0], points_to_sample[:, 1], points_to_sample[:, 2], c=samples, cmap="gray")

    hcatnetwork.draw.draw_centerlines_graph_3d(graph_RAS, ax)


def plot_slice_with_mask_and_centers(image, masks, graph_ijk, slice_num, ax=None, show=True, crop_center=False):
    """
        !!!!! NEEDS GRAPH IN IJK COORDINATES !!!!!
    args:
        - image: image as np.array
        - masks: masks np.array (same shape as image.data)
        - graph_ijk: hcatnetwork.graph.SimpleCenterlineGraph graph in "ijk" coord. (IMPORTANT !!!!)
        - slice_num: number of the slice to display. must be between (0, image.shape[3] -1)

    """


    z_dict = build_centerline_per_slice_dict(graph_ijk)

    xy = np.array([[floor_or_ceil(graph_ijk.nodes[i]['x']), floor_or_ceil(graph_ijk.nodes[i]['y'])] for i in z_dict[slice_num]])

    if crop_center:
        dx = image.shape[0] // 2
        c_i, c_j = crop_center
        top_left_i, top_left_j = c_i - dx, c_j - dx
        translated_points = [(xy_vect[0] - top_left_i, xy_vect[1] - top_left_j) for xy_vect in xy]
        translated_points = np.array([[i, j] for i, j in translated_points if 0 <= j < image.shape[0] and 0 <= i < image.shape[0]])

        xy = np.array(translated_points)

        img = min_max(image)
        lab = masks
    else:
        img = min_max(image[:, :, slice_num])
        lab = masks[:, :, slice_num].astype(np.uint8)

    img[lab == 1] = 1
    if not ax:
        f, ax = plt.subplots(1, 1)

    ax.imshow(img, cmap='gray')
    ax.scatter(xy[:, 1], xy[:, 0], color='red', s=5)

    if show:
        plt.show()


def plot_slice_with_mask(image, masks, slice_num, ax=None, show=True):
    """
        !!!!! NEEDS GRAPH IN IJK COORDINATES !!!!!
    args:
        - image: image as np.array
        - masks: masks np.array (same shape as image.data)
        - slice_num: num of slice to plot

    """

    img = min_max(image[:, :, slice_num])
    lab = masks[:, :, slice_num].astype(np.uint8)

    img[lab == 1] = 1
    if not ax:
        f, ax = plt.subplots(1, 1)

    ax.imshow(img, cmap='gray')

    if show:
        plt.show()