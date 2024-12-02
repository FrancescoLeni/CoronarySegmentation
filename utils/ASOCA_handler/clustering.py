
import numpy as np
from sklearn.cluster import DBSCAN

from .general import build_centerline_per_slice_dict, floor_or_ceil


def cluster_centerlines_in_slice(graph, n_slice, eps=5):
    """
        uses DBSCAN to find clusters' centroids of centerlines points inside a givent slice

        args:
            - graph: graph in 'ijk'
            - n_slice: number of current slice
        returns:
            - cluster_centroids: a np.array with centroids of the clusters of centerline's points (i, j) coord
    """
    z_dict = build_centerline_per_slice_dict(graph)

    xy = np.array([[floor_or_ceil(graph.nodes[i]['x']), floor_or_ceil(graph.nodes[i]['y'])] for i in z_dict[n_slice]])

    db = DBSCAN(eps=eps, min_samples=1).fit(xy)

    # Extract cluster labels (-1 means noise)
    labels = db.labels_
    unique_labels = np.unique(labels)

    cluster_centroids = []
    for label in unique_labels:

        cluster_points = xy[labels == label]
        cluster_centroid = cluster_points.mean(axis=0)
        cluster_centroids.append(cluster_centroid)  # row, col

    cluster_centroids = np.array(cluster_centroids)

    # centroid of clusters
    return cluster_centroids, xy


def get_new_centroids(closest, old_centroids):
    """

    recreates the centroid for given clouds of closest points

    args:
        - closest: a np.array with N pairs of (i,j) coordinates [shape = Nx2]
        - old_centroids: a np.array with old centroids

    returns:
        - clusters: clustered old centorids
        - updated_centroids: array with i,j coords of final centroids
    """

    clusters = []

    def find_cluster(point_index, clusters):
        for idx, cluster in enumerate(clusters):
            if point_index in cluster:
                return idx
        return -1  # If the point is not in any cluster

    # Create clusters based on close points
    for i, j in closest:
        cluster_i = find_cluster(i, clusters)
        cluster_j = find_cluster(j, clusters)

        if cluster_i == -1 and cluster_j == -1:  # Both points are not in any cluster
            clusters.append([i, j])
        elif cluster_i == -1:  # Only point i is not in a cluster
            clusters[cluster_j].append(i)
        elif cluster_j == -1:  # Only point j is not in a cluster
            clusters[cluster_i].append(j)
        elif cluster_i != cluster_j:  # Merge two different clusters
            clusters[cluster_i].extend(clusters[cluster_j])
            clusters.pop(cluster_j)

    new_centroids = []
    for cluster in clusters:
        cluster_points = old_centroids[cluster]
        centroid = np.mean(cluster_points, axis=0)
        new_centroids.append(centroid)

    # removing closest centroid and substituting them with the new ones
    to_replace = np.unique([i for i, _ in closest] + [j for _, j in closest])
    updated_centroids = [x for i, x in enumerate(old_centroids) if i not in to_replace] + new_centroids

    return clusters, np.array(updated_centroids)


def get_slice_centroids(n_slice, graph, eps=5, closeness=50.):
    """

    args:
        - n_slice: number of current slice
        - graph: graph in 'ijk'
        - eps: DBSCAN eps (max distance for points inside cluster)
        - closeness: minimum distance between centroids
    returns:
        - updated_centroids: final set of centroids (np.array of shape Nc x 2) (i,j coord)
    """
    centroids, xy = cluster_centerlines_in_slice(graph, n_slice, eps)
    distances = np.sqrt(((centroids[:, np.newaxis, :] - centroids[np.newaxis, :, :]) ** 2).sum(axis=2))
    distances = np.triu(distances)

    closest = []
    for i in range(distances.shape[0]):
        for j, d in enumerate(distances[i]):
            if d <= closeness and d != 0.:
                closest.append((i, j))
    if closest:
        _, updated_centroids = get_new_centroids(np.array(closest), centroids)
    else:
        updated_centroids = centroids

    return updated_centroids.astype(int), xy



