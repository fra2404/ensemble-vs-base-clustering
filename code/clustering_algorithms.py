from sklearn.cluster import KMeans, AgglomerativeClustering, DBSCAN
import hdbscan


def run_kmeans(X, n_clusters):
    """Run K-Means clustering."""
    kmeans = KMeans(n_clusters=n_clusters, random_state=42, n_init=10)
    labels = kmeans.fit_predict(X)
    return labels, kmeans


def run_agglomerative(X, n_clusters):
    """Run Agglomerative clustering."""
    agg = AgglomerativeClustering(n_clusters=n_clusters)
    labels = agg.fit_predict(X)
    return labels, agg


def run_dbscan(X, eps, min_samples):
    """Run DBSCAN clustering."""
    dbscan = DBSCAN(eps=eps, min_samples=min_samples)
    labels = dbscan.fit_predict(X)
    return labels, dbscan


def run_hdbscan(X):
    """Run HDBSCAN clustering."""
    clusterer = hdbscan.HDBSCAN(min_cluster_size=5, min_samples=3)
    labels = clusterer.fit_predict(X)
    return labels, clusterer


def run_all_base_algorithms(X, n_clusters, eps, min_samples):
    """
    Run all base clustering algorithms.

    Returns:
    - results: dict with labels and models
    """
    results = {}
    results['K-Means'], results['K-Means Model'] = run_kmeans(X, n_clusters)
    results['Agglomerative'], results['Agglomerative Model'] = run_agglomerative(X, n_clusters)
    results['DBSCAN'], results['DBSCAN Model'] = run_dbscan(X, eps, min_samples)
    results['HDBSCAN'], results['HDBSCAN Model'] = run_hdbscan(X)
    return results