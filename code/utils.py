import numpy as np
from sklearn.metrics import silhouette_score, davies_bouldin_score, calinski_harabasz_score, adjusted_rand_score
from sklearn.metrics.cluster import mutual_info_score
from sklearn.cluster import SpectralClustering
from sklearn.neighbors import NearestNeighbors
from scipy.stats import entropy
import networkx as nx
from collections import defaultdict
import warnings
warnings.filterwarnings('ignore')


def variation_of_information(labels_true, labels_pred):
    """
    Compute Variation of Information between two clusterings.
    VI = H(X) + H(Y) - 2*I(X,Y)
    """
    # Compute entropies
    _, counts_true = np.unique(labels_true, return_counts=True)
    h_true = entropy(counts_true / len(labels_true))

    _, counts_pred = np.unique(labels_pred, return_counts=True)
    h_pred = entropy(counts_pred / len(labels_pred))

    # Mutual information
    mi = mutual_info_score(labels_true, labels_pred)

    vi = h_true + h_pred - 2 * mi
    return vi


def compute_k_distance(X, k):
    """
    Compute k-distance for DBSCAN eps selection.
    """
    nbrs = NearestNeighbors(n_neighbors=k).fit(X)
    distances, _ = nbrs.kneighbors(X)
    k_distances = np.sort(distances[:, -1])
    return k_distances


def tune_dbscan_eps(X, min_samples):
    """
    Tune eps using k-distance plot (elbow method approximation).
    """
    k_distances = compute_k_distance(X, min_samples)
    # Simple elbow: find point where slope changes
    diffs = np.diff(k_distances)
    elbow_idx = np.argmax(diffs)  # Approximate
    eps = k_distances[elbow_idx]
    return eps


def cluster_sizes(labels):
    """
    Compute sizes of clusters.
    """
    unique, counts = np.unique(labels, return_counts=True)
    sizes = dict(zip(unique, counts))
    return sizes


def compute_dbcv(X, labels):
    """
    Compute Density-Based Clustering Validation (DBCV) index.

    Parameters:
    - X: array-like, shape (n_samples, n_features)
    - labels: array-like, shape (n_samples,)

    Returns:
    - float: DBCV score
    """
    if len(set(labels)) <= 1 or -1 in labels:
        return np.nan

    unique_labels = set(labels)
    if -1 in unique_labels:
        unique_labels.remove(-1)

    # Compute density for each point
    nbrs = NearestNeighbors(n_neighbors=2).fit(X)
    distances, _ = nbrs.kneighbors(X)

    # Core distance: distance to k-th nearest neighbor
    core_distances = distances[:, 1]

    # Mutual reachability distance
    mrd = np.zeros((len(X), len(X)))
    for i in range(len(X)):
        for j in range(len(X)):
            mrd[i, j] = max(core_distances[i], core_distances[j], np.linalg.norm(X[i] - X[j]))

    # For each cluster, compute validity
    cluster_validities = []
    for label in unique_labels:
        cluster_points = [i for i, l in enumerate(labels) if l == label]
        if len(cluster_points) < 2:
            continue

        # Build MST for the cluster using MRD
        G = nx.Graph()
        for i in cluster_points:
            for j in cluster_points:
                if i != j:
                    G.add_edge(i, j, weight=mrd[i, j])

        mst = nx.minimum_spanning_tree(G)
        edges = list(mst.edges(data=True))

        # Density of cluster
        density = 0
        for _, _, d in edges:
            density += 1 / d['weight']
        density /= len(cluster_points)

        cluster_validities.append(density)

    if not cluster_validities:
        return np.nan

    # Overall DBCV
    return np.mean(cluster_validities)


def compute_all_metrics(X, labels):
    """
    Compute all validation metrics.

    Parameters:
    - X: array-like
    - labels: array-like

    Returns:
    - dict: Metrics
    """
    metrics = {}
    try:
        metrics['Silhouette Score'] = silhouette_score(X, labels)
    except (ValueError, RuntimeError):
        metrics['Silhouette Score'] = np.nan

    try:
        metrics['Davies-Bouldin Index'] = davies_bouldin_score(X, labels)
    except (ValueError, RuntimeError):
        metrics['Davies-Bouldin Index'] = np.nan

    try:
        metrics['Calinski-Harabasz Index'] = calinski_harabasz_score(X, labels)
    except (ValueError, RuntimeError):
        metrics['Calinski-Harabasz Index'] = np.nan

    # DBCV removed due to computational instability (inf/nan issues)
    # Future work: implement robust density-based validation

    return metrics


def cspa_ensemble(base_labels_list, n_clusters_final=5):
    """
    Cluster-based Similarity Partitioning Algorithm (CSPA).

    Parameters:
    - base_labels_list: list of arrays, each shape (n_samples,)
    - n_clusters_final: int

    Returns:
    - array: Final labels
    """
    n_samples = len(base_labels_list[0])
    n_models = len(base_labels_list)

    # Build similarity matrix
    similarity = np.zeros((n_samples, n_samples))
    for i in range(n_samples):
        for j in range(n_samples):
            sim = sum(1 for labels in base_labels_list if labels[i] == labels[j])
            similarity[i, j] = sim

    # Normalize
    similarity /= n_models

    # Spectral clustering
    sc = SpectralClustering(n_clusters=n_clusters_final, affinity='precomputed', random_state=42)
    labels = sc.fit_predict(similarity)

    return labels


def hgpa_ensemble(base_labels_list, n_clusters_final=5):
    """
    HyperGraph Partitioning Algorithm (HGPA) approximation.

    Parameters:
    - base_labels_list: list of arrays
    - n_clusters_final: int

    Returns:
    - array: Final labels
    """
    n_samples = len(base_labels_list[0])

    # Build hypergraph: each cluster is a hyperedge
    hyperedges = []
    for labels in base_labels_list:
        unique_labels = set(labels)
        if -1 in unique_labels:
            unique_labels.remove(-1)
        for label in unique_labels:
            hyperedge = [i for i, l in enumerate(labels) if l == label]
            if len(hyperedge) > 1:
                hyperedges.append(hyperedge)

    # Approximate partitioning: use graph cut heuristic
    # Build a graph where nodes are samples, edges based on shared hyperedges
    G = nx.Graph()
    G.add_nodes_from(range(n_samples))

    for hyperedge in hyperedges:
        for i in hyperedge:
            for j in hyperedge:
                if i != j:
                    if G.has_edge(i, j):
                        G[i][j]['weight'] += 1
                    else:
                        G.add_edge(i, j, weight=1)

    # Partition using spectral clustering on the graph
    adjacency = nx.to_numpy_array(G)
    sc = SpectralClustering(n_clusters=n_clusters_final, affinity='precomputed', random_state=42)
    labels = sc.fit_predict(adjacency + np.eye(n_samples))  # Add self-loops

    return labels


def mcla_ensemble(base_labels_list, n_clusters_final=5):
    """
    Meta-CLustering Algorithm (MCLA).

    Parameters:
    - base_labels_list: list of arrays
    - n_clusters_final: int

    Returns:
    - array: Final labels
    """
    # Collect all clusters
    all_clusters = []
    cluster_to_model = []
    for model_idx, labels in enumerate(base_labels_list):
        unique_labels = set(labels)
        if -1 in unique_labels:
            unique_labels.remove(-1)
        for label in unique_labels:
            cluster_points = [i for i, l in enumerate(labels) if l == label]
            all_clusters.append(cluster_points)
            cluster_to_model.append(model_idx)

    n_clusters_total = len(all_clusters)

    # Similarity between clusters: Jaccard
    cluster_sim = np.zeros((n_clusters_total, n_clusters_total))
    for i in range(n_clusters_total):
        for j in range(n_clusters_total):
            set_i = set(all_clusters[i])
            set_j = set(all_clusters[j])
            inter = len(set_i & set_j)
            union = len(set_i | set_j)
            cluster_sim[i, j] = inter / union if union > 0 else 0

    # Meta-clustering: simple hierarchical on clusters
    from sklearn.cluster import AgglomerativeClustering
    meta_clust = AgglomerativeClustering(n_clusters=n_clusters_final, linkage='ward')
    meta_labels = meta_clust.fit_predict(cluster_sim)

    # Assign points to meta-clusters
    point_assignments = {}
    for cluster_idx, meta_label in enumerate(meta_labels):
        model_idx = cluster_to_model[cluster_idx]
        for point in all_clusters[cluster_idx]:
            if point not in point_assignments:
                point_assignments[point] = defaultdict(int)
            point_assignments[point][meta_label] += 1

    final_labels = np.full(len(base_labels_list[0]), -1)
    for point, counts in point_assignments.items():
        final_labels[point] = max(counts, key=counts.get)

    return final_labels


def bootstrap_stability(X, clustering_func, n_boot=20):
    """
    Compute bootstrap stability.

    Parameters:
    - X: array-like
    - clustering_func: function that takes X and returns labels
    - n_boot: int

    Returns:
    - list of ARI, list of VI
    """
    n_samples = len(X)
    base_labels = clustering_func(X)

    ari_scores = []
    vi_scores = []

    for _ in range(n_boot):
        indices = np.random.choice(n_samples, n_samples, replace=True)
        X_boot = X[indices]
        labels_boot = clustering_func(X_boot)

        # Map back to original indices? For simplicity, assume same order
        ari = adjusted_rand_score(base_labels[indices], labels_boot)
        vi = variation_of_information(base_labels[indices], labels_boot)

        ari_scores.append(ari)
        vi_scores.append(vi)

    return ari_scores, vi_scores


def noise_injection_stability(X, clustering_func, sigma=0.05):
    """
    Compute stability under noise injection.

    Parameters:
    - X: array-like
    - clustering_func: function
    - sigma: float

    Returns:
    - ari, vi
    """
    base_labels = clustering_func(X)

    noise = np.random.normal(0, sigma, X.shape)
    X_noisy = X + noise

    noisy_labels = clustering_func(X_noisy)

    ari = adjusted_rand_score(base_labels, noisy_labels)
    vi = variation_of_information(base_labels, noisy_labels)

    return ari, vi