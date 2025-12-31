from utils import compute_all_metrics, bootstrap_stability, noise_injection_stability
import pandas as pd


def evaluate_all_algorithms(X, all_labels):
    """
    Compute metrics for all algorithms.

    Returns:
    - metrics_df: DataFrame with metrics
    """
    metrics = {}
    for name, labels in all_labels.items():
        metrics[name] = compute_all_metrics(X, labels)
    metrics_df = pd.DataFrame(metrics).T
    return metrics_df


from sklearn.cluster import KMeans, AgglomerativeClustering, DBSCAN
import hdbscan
from utils import cspa_ensemble, hgpa_ensemble, mcla_ensemble


def perform_stability_analysis(X, all_labels, *, n_clusters=5, eps=0.5, min_samples=5):
    """
    Perform stability analysis.

    Returns:
    - stability_results: dict with ARI and VI
    """
    # Define clustering functions
    funcs = {
        'K-Means': lambda data: KMeans(n_clusters=n_clusters, random_state=42).fit_predict(data),
        'Agglomerative': lambda data: AgglomerativeClustering(n_clusters=n_clusters, linkage='ward').fit_predict(data),
        'DBSCAN': lambda data: DBSCAN(eps=eps, min_samples=min_samples).fit_predict(data),
        'HDBSCAN': lambda data: hdbscan.HDBSCAN(min_cluster_size=5, min_samples=3).fit_predict(data),
        'CSPA': lambda data: cspa_ensemble([
            KMeans(n_clusters=n_clusters, random_state=42).fit_predict(data),
            AgglomerativeClustering(n_clusters=n_clusters, linkage='ward').fit_predict(data),
            DBSCAN(eps=eps, min_samples=min_samples).fit_predict(data)
        ], n_clusters),
        'HGPA': lambda data: hgpa_ensemble([
            KMeans(n_clusters=n_clusters, random_state=42).fit_predict(data),
            AgglomerativeClustering(n_clusters=n_clusters, linkage='ward').fit_predict(data),
            DBSCAN(eps=eps, min_samples=min_samples).fit_predict(data)
        ], n_clusters),
        'MCLA': lambda data: mcla_ensemble([
            KMeans(n_clusters=n_clusters, random_state=42).fit_predict(data),
            AgglomerativeClustering(n_clusters=n_clusters, linkage='ward').fit_predict(data),
            DBSCAN(eps=eps, min_samples=min_samples).fit_predict(data)
        ], n_clusters)
    }

    stability_results = {}
    for name in all_labels.keys():
        if name in funcs:
            ari, vi = bootstrap_stability(X, funcs[name], n_boot=50)
            stability_results[name] = {'ARI': ari, 'VI': vi}
    return stability_results


def perform_noise_stability_analysis(X, all_labels, *, n_clusters=5, eps=0.5, min_samples=5, sigma=0.05):
    """
    Perform noise injection stability.

    Returns:
    - noise_stability: dict with ARI and VI
    """
    # Define clustering functions (same as above)
    funcs = {
        'K-Means': lambda data: KMeans(n_clusters=n_clusters, random_state=42).fit_predict(data),
        'Agglomerative': lambda data: AgglomerativeClustering(n_clusters=n_clusters, linkage='ward').fit_predict(data),
        'DBSCAN': lambda data: DBSCAN(eps=eps, min_samples=min_samples).fit_predict(data),
        'HDBSCAN': lambda data: hdbscan.HDBSCAN(min_cluster_size=5, min_samples=3).fit_predict(data),
        'CSPA': lambda data: cspa_ensemble([
            KMeans(n_clusters=n_clusters, random_state=42).fit_predict(data),
            AgglomerativeClustering(n_clusters=n_clusters, linkage='ward').fit_predict(data),
            DBSCAN(eps=eps, min_samples=min_samples).fit_predict(data)
        ], n_clusters),
        'HGPA': lambda data: hgpa_ensemble([
            KMeans(n_clusters=n_clusters, random_state=42).fit_predict(data),
            AgglomerativeClustering(n_clusters=n_clusters, linkage='ward').fit_predict(data),
            DBSCAN(eps=eps, min_samples=min_samples).fit_predict(data)
        ], n_clusters),
        'MCLA': lambda data: mcla_ensemble([
            KMeans(n_clusters=n_clusters, random_state=42).fit_predict(data),
            AgglomerativeClustering(n_clusters=n_clusters, linkage='ward').fit_predict(data),
            DBSCAN(eps=eps, min_samples=min_samples).fit_predict(data)
        ], n_clusters)
    }

    noise_stability = {}
    for name in all_labels.keys():
        if name in funcs:
            ari, vi = noise_injection_stability(X, funcs[name], sigma=sigma)
            noise_stability[name] = {'ARI': ari, 'VI': vi}
    return noise_stability