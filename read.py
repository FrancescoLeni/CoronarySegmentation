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

from utils.ASOCA_handler.visualize import plot_centerline_over_CT, plot_slice_with_mask_and_centers
from utils.ASOCA_handler.general import load_centerline, load_single_volume, align_centerline_to_image, floor_or_ceil, \
                                        build_centerline_per_slice_dict


# MEMO
# ho aggiunto:
#   self.path in HearticDatasetManager.asoca.py ~ AsocaImageCT
#
#   cambiato questa classe in hcatnetwork.draw e l'ho messa nell'init per importarla
#
#   def draw_centerlines_graph_3d(graph, ax=None, transform=None):
#     """ NOT READY YET, DO NOT USE
#     Assumes this kind on dictionaries:
#         nodes: hcatnetwork.node.SimpleCenterlineNodeAttributes
#         edges: hcatnetwork.edge.SimpleCenterlineEdgeAttributes
#         graph: hcatnetwork.graph.SimpleCenterlineGraph
#     """
#     if not ax:
#         ax = plt.subplot(111, projection="3d")
#     # plot nodes
#     c_in = []
#     c_out = []
#     s_in = 25
#     s_out = []
#     positions = []
#     for n in graph.nodes:
#         n_ = SimpleCenterlineNodeAttributes(**(graph.nodes[n]))
#         c_in.append(NODE_FACECOLOR_RCA if n_["side"].value == ArteryNodeSide.RIGHT.value else NODE_FACECOLOR_LCA)
#         if n_["topology"].value == ArteryNodeTopology.OSTIUM.value:
#             c_out.append(NODE_EDGEECOLOR_START)
#             s_out.append(2.5)
#         elif n_["topology"].value == ArteryNodeTopology.ENDPOINT.value:
#             c_out.append(NODE_EDGEECOLOR_END)
#             s_out.append(2.5)
#         elif n_["topology"].value == ArteryNodeTopology.INTERSECTION.value:
#             c_out.append(NODE_EDGEECOLOR_CROSS)
#             s_out.append(2)
#         else:
#             c_out.append(NODE_EDGEECOLOR_DEFAULT)
#             s_out.append(0.0)
#
#         if isinstance(transform, numpy.ndarray):
#             nn = n_.get_vertex_numpy_array()
#             positions.append(apply_affine_3d(transform,nn.T).tolist())
#         else:
#             positions.append(n_.get_vertex_list())
#     # - convert to numpy
#     c_in  = numpy.array(c_in)
#     c_out = numpy.array(c_out)
#     s_out = numpy.array(s_out)
#     positions = numpy.array(positions)
#
#     # - plot
#     below_idx_ = [i for i in range(len(c_out)) if c_out[i] == NODE_EDGEECOLOR_DEFAULT]
#     above_idx_ = [i for i in range(len(c_out)) if c_out[i] != NODE_EDGEECOLOR_DEFAULT]
#     ax.scatter( # - below
#         positions[below_idx_,0],
#         positions[below_idx_,1],
#         positions[below_idx_,2],
#         c=c_in[below_idx_],
#         s=s_in,
#         zorder=1.5,
#         linewidths=s_out[below_idx_],
#         edgecolors=c_out[below_idx_]
#     )
#     ax.scatter( # - above
#         positions[above_idx_,0],
#         positions[above_idx_,1],
#         positions[above_idx_,2],
#         c=c_in[above_idx_],
#         s=s_in*4,
#         zorder=2,
#         linewidths=s_out[above_idx_],
#         edgecolors=c_out[above_idx_]
#     )
#     # plot undirected edges
#     segs = []
#     for u_,v_,a in graph.edges(data=True):
#         uu = SimpleCenterlineNodeAttributes(**(graph.nodes[u_])).get_vertex_list()
#         vv = SimpleCenterlineNodeAttributes(**(graph.nodes[v_])).get_vertex_list()
#         segs.append(numpy.array([uu[:3],vv[:3]]))
#     line_segments = Line3DCollection(segs, zorder=1, linewidth=0.4, color=EDGE_FACECOLOR_DEFAULT)
#     ax.add_collection(line_segments)
#     # legend
#     legend_elements = [
#         Line2D([0], [0], marker='o', markerfacecolor=NODE_FACECOLOR_RCA, color="w",                 markersize=10, lw=0),
#         Line2D([0], [0], marker='o', markerfacecolor=NODE_FACECOLOR_LCA, color="w",                 markersize=10, lw=0),
#         Line2D([0], [0], marker='o', markerfacecolor="w",          color=NODE_EDGEECOLOR_START,    markersize=10, lw=0),
#         Line2D([0], [0], marker='o', markerfacecolor="w",          color=NODE_EDGEECOLOR_CROSS, markersize=10, lw=0),
#         Line2D([0], [0], marker='o', markerfacecolor="w",          color=NODE_EDGEECOLOR_END,  markersize=10, lw=0)
#     ]
#     ax.legend(
#         legend_elements,
#         ["RCA",
#          "LCA",
#          "OSTIA",
#          "INTERSECTIONS",
#          "ENDPOINTS"]
#     )
#     # axis
#     ax.set_xlabel("mm")
#     ax.set_ylabel("mm")
#     ax.set_zlabel("mm")
#     ax.grid(color='gray', linestyle='dashed')
#     ax.set_title(graph.graph["image_id"])
#     # out
#     plt.tight_layout()
#     plt.show()


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
# plot_centerline_over_CT(image, graph)

# single sclice plot
# plot_slice_with_mask_and_centers(image, labs, graph, 50)


