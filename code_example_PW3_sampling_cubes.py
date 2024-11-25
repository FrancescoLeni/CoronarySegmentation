import os
import numpy
import torch
import matplotlib.pyplot as plt
import hcatnetwork
import HearticDatasetManager

DATA_FOLDER = "E:/MatteoLeccardi/HearticData/ASOCA/"

# open image
image_file = os.path.join(
    DATA_FOLDER,
    HearticDatasetManager.asoca.DATASET_ASOCA_IMAGES_DICT["Normal"][0] 
)
image = HearticDatasetManager.asoca.AsocaImageCT(image_file)

# open graph and convert its coordinates to RAS right away
graph_file = os.path.join(
    DATA_FOLDER,
    HearticDatasetManager.asoca.DATASET_ASOCA_GRAPHS_RESAMPLED_05MM_DICT["Normal"][0]
)
graph = hcatnetwork.io.load_graph(
    graph_file,
    output_type=hcatnetwork.graph.SimpleCenterlineGraph
)
for node_id in graph.nodes:
    old_coords = numpy.array(
        [graph.nodes[node_id]["x"], graph.nodes[node_id]["y"], graph.nodes[node_id]["z"]]
    )
    new_coords = HearticDatasetManager.affine.apply_affine_3d(image.affine_centerlines2ras, old_coords)
    graph.nodes[node_id]["x"] = new_coords[0]   
    graph.nodes[node_id]["y"] = new_coords[1]
    graph.nodes[node_id]["z"] = new_coords[2]
    
# extract a cube from the original image
CUBE_SIDE_MM = 12
CUBE_ISOTROPIC_SPACING_MM = 0.3
CUBE_SIDE_N_SAMPLES = int(CUBE_SIDE_MM * (1/CUBE_ISOTROPIC_SPACING_MM)) # you can define this as you prefer, no need to stick to this formula.

def get_cube_sample_points(center: numpy.ndarray, side_mm: float, n_samples_per_side: int):
    """Sample a cube centered in center with side side, n_samples points."""
    xs = numpy.linspace(center[0] - side_mm/2, center[0] + side_mm/2, n_samples_per_side)
    ys = numpy.linspace(center[1] - side_mm/2, center[1] + side_mm/2, n_samples_per_side)
    zs = numpy.linspace(center[2] - side_mm/2, center[2] + side_mm/2, n_samples_per_side)
    return numpy.array(numpy.meshgrid(xs, ys, zs)).reshape(3, -1).T

def cube_samples_to_array(samples: numpy.ndarray, n_samples_per_side: int) -> numpy.ndarray:
    """Convert sampoles from a cube to a numpy array."""
    return samples.reshape(n_samples_per_side, n_samples_per_side, n_samples_per_side)

def cube_samples_to_tensor(samples: numpy.ndarray, n_samples_per_side: int) -> torch.Tensor:
    """Convert a sampled cube to a graph."""
    return torch.from_numpy(cube_samples_to_array(samples, n_samples_per_side)).float()

def get_input_data_from_vertex_ras_position(
        image: HearticDatasetManager.cat08.Cat08ImageCT|HearticDatasetManager.asoca.AsocaImageCT,
        position: numpy.ndarray,
        side_mm: float,
        n_samples_per_side: int,
        affine=numpy.eye(4)
    ) -> numpy.ndarray:
    """Get the input data from a vertex position expressed in RAS coordinates system.

    Parameters
    ----------
    image : HearticDatasetManager.cat08.Cat08ImageCT | HearticDatasetManager.asoca.AsocaImageCT
        The image from which to extract the data.
    position : numpy.ndarray
        The position of the cube center in RAS coordinates system.
    side_mm : float
        The side of the cube in mm.
    n_samples_per_side : int
        The number of samples per side.
    affine : numpy.ndarray, optional
        The affine transformation to apply to the position of the samples used to create the cube, by default numpy.eye(4) (which does nothing).
        This is useful in data augmemtation, if you want to rotate, flip, or do whatever operation
        on the cube sample points, you can do it by passing the affine transformation here.
        For example, HearticDatasetManager.affine.get_affine_3d_rotation_around_vector() will rotate the cube (see the function docs).
    """
    # Get the cube sample points
    cube_pos = get_cube_sample_points(position, side_mm, n_samples_per_side)
    # Apply transformation affine if any
    if affine is None:
        affine = numpy.eye(4)
    cube_pos = HearticDatasetManager.affine.apply_affine_3d(affine, cube_pos)
    # Sample the image
    samples = image.sample(cube_pos, interpolation="linear").T
    # Convert to ndarray
    cube_array = cube_samples_to_array(samples, n_samples_per_side)
    return cube_array

# Example 1: without rotation
# ---------------------------

# - choose a random node of the graph
node_id = numpy.random.choice(list(graph.nodes.keys()))
# - get the position of the node in RAS
node_position = numpy.array(
    [graph.nodes[node_id]["x"], graph.nodes[node_id]["y"], graph.nodes[node_id]["z"]]
)
# - move it a bit in a random direction
r = numpy.random.uniform(0, 2.5644) # or whatever value you want
theta = numpy.random.uniform(0, 2*numpy.pi)
phi = numpy.random.uniform(0, numpy.pi)
node_position += numpy.array(
    [r*numpy.sin(phi)*numpy.cos(theta), r*numpy.sin(phi)*numpy.sin(theta), r*numpy.cos(phi)]
).reshape(3,1)
# - now, each voxel of the sampled cube will be sampled at the position shown below
#   (the cube is centered in the node position)
cube_samples_position = get_cube_sample_points(node_position, CUBE_SIDE_MM, CUBE_SIDE_N_SAMPLES)
print(f"cube samples position array shape: {cube_samples_position.shape}")
if 1:
    # plot the cube samples position with respect to the image bounding box
    fig = plt.figure()
    ax = fig.add_subplot(111, projection='3d')
    ax.set_xlim(image.bounding_box.get_xlim()[0], image.bounding_box.get_xlim()[1])
    ax.set_ylim(image.bounding_box.get_ylim()[0], image.bounding_box.get_ylim()[1])
    ax.set_zlim(image.bounding_box.get_zlim()[0], image.bounding_box.get_zlim()[1])
    ax.set_title("Cube samples position")
    ax.add_artist(image.bounding_box.get_artist())
    ax.scatter(cube_samples_position[:, 0], cube_samples_position[:, 1], cube_samples_position[:, 2])
    plt.show()

# - sampling works as follows
samples = image.sample(cube_samples_position.T, interpolation="linear").T
print(f"samples shape: {samples.shape}")
if 1:
    # plot the cube samples position with respect to the image bounding box
    fig = plt.figure()
    ax = fig.add_subplot(111, projection='3d')
    ax.set_xlim(image.bounding_box.get_xlim()[0], image.bounding_box.get_xlim()[1])
    ax.set_ylim(image.bounding_box.get_ylim()[0], image.bounding_box.get_ylim()[1])
    ax.set_zlim(image.bounding_box.get_zlim()[0], image.bounding_box.get_zlim()[1])
    ax.set_title("Sampled cube wrt image BB")
    ax.add_artist(image.bounding_box.get_artist())
    ax.scatter(cube_samples_position[:, 0], cube_samples_position[:, 1], cube_samples_position[:, 2], c=samples, cmap="gray")
    plt.show()
    # plot the cube samples positions colored with the sampled values
    fig = plt.figure()
    ax = fig.add_subplot(111, projection='3d')
    ax.set_title("Sampled cube zoomed in")
    ax.scatter(cube_samples_position[:, 0], cube_samples_position[:, 1], cube_samples_position[:, 2], c=samples, cmap="gray")
    plt.show()

# - convert the samples to a numpy array
cube_array = cube_samples_to_array(samples, CUBE_SIDE_N_SAMPLES)
print(f"cube array shape: {cube_array.shape}")
if 1:
    # plot a slice of the cube
    plt.imshow(cube_array[:, ::-1, int(CUBE_SIDE_N_SAMPLES/2)], cmap="gray")
    plt.show()

# - if you want, this function condenses everything we did above into a single function
cube_array = get_input_data_from_vertex_ras_position(
    image,
    node_position,
    CUBE_SIDE_MM,
    CUBE_SIDE_N_SAMPLES
)
if 1:
    # plot a slice of the cube
    plt.imshow(cube_array[:, ::-1, int(CUBE_SIDE_N_SAMPLES/2)], cmap="gray")
    plt.show()


# Example 2: with rotation
# ------------------------
vector_axis_of_rotation = numpy.array([0, 0.23, 1]) # each point will rotate around this vector
transformation_to_apply = HearticDatasetManager.affine.get_affine_3d_rotation_around_vector(
    vector=vector_axis_of_rotation,
    vector_source=node_position.reshape(3,1), # the center of the cube
    rotation=37.0, # degrees
    rotation_units="deg"
)
if 1:
    # plot the cube samples position with respect to the image bounding box
    # before the rotation
    # Let's also plot the arrow of the rotation
    fig = plt.figure()
    ax = fig.add_subplot(111, projection='3d')
    ax.set_xlim(image.bounding_box.get_xlim()[0], image.bounding_box.get_xlim()[1])
    ax.set_ylim(image.bounding_box.get_ylim()[0], image.bounding_box.get_ylim()[1])
    ax.set_zlim(image.bounding_box.get_zlim()[0], image.bounding_box.get_zlim()[1])
    ax.set_title("Cube samples position + axis of rotation")
    ax.add_artist(image.bounding_box.get_artist())
    ax.scatter(cube_samples_position[:, 0], cube_samples_position[:, 1], cube_samples_position[:, 2], c="blue", s=1, alpha=0.05)
    ax.scatter(node_position[0], node_position[1], node_position[2], c="red", s = 15)
    ax.quiver(
        node_position[0], node_position[1], node_position[2],
        30*vector_axis_of_rotation[0], 30*vector_axis_of_rotation[1], 30*vector_axis_of_rotation[2],
        color="green"
    )
    plt.show()

# - sampling works as follows
cube_array = get_input_data_from_vertex_ras_position(
    image,
    node_position,
    CUBE_SIDE_MM,
    CUBE_SIDE_N_SAMPLES,
    affine=transformation_to_apply
)
if 1:
    # plot a slice of the cube
    plt.imshow(cube_array[:, ::-1, int(CUBE_SIDE_N_SAMPLES/2)], cmap="gray")
    plt.show()

quit()