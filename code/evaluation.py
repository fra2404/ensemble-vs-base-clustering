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


def perform_stability_analysis(X, all_labels):
    """
    Perform stability analysis.

    Returns:
    - stability_results: dict with ARI and VI
    """
    # Define clustering functions
    funcs = {
        'K-Means': lambda data: KMeans(n_clusters=5, random_state=42).fit_predict(data),
        'Agglomerative': lambda data: AgglomerativeClustering(n_clusters=5, linkage='ward').fit_predict(data),
        'DBSCAN': lambda data: DBSCAN(eps=0.5, min_samples=5).fit_predict(data),
        'HDBSCAN': lambda data: hdbscan.HDBSCAN(min_cluster_size=5, min_samples=3).fit_predict(data),
        'CSPA': lambda data: cspa_ensemble([
            KMeans(n_clusters=5, random_state=42).fit_predict(data),
            AgglomerativeClustering(n_clusters=5, linkage='ward').fit_predict(data),
            DBSCAN(eps=0.5, min_samples=5).fit_predict(data)
        ], 5),
        'HGPA': lambda data: hgpa_ensemble([
            KMeans(n_clusters=5, random_state=42).fit_predict(data),
            AgglomerativeClustering(n_clusters=5, linkage='ward').fit_predict(data),
            DBSCAN(eps=0.5, min_samples=5).fit_predict(data)
        ], 5),
        'MCLA': lambda data: mcla_ensemble([
            KMeans(n_clusters=5, random_state=42).fit_predict(data),
            AgglomerativeClustering(n_clusters=5, linkage='ward').fit_predict(data),
            DBSCAN(eps=0.5, min_samples=5).fit_predict(data)
        ], 5)
    }

    stability_results = {}
    for name in all_labels.keys():
        if name in funcs:
            ari, vi = bootstrap_stability(X, funcs[name], n_boot=50)
            stability_results[name] = {'ARI': ari, 'VI': vi}
    return stability_results


def perform_noise_stability_analysis(X, all_labels):
    """
    Perform noise injection stability.

    Returns:
    - noise_stability: dict with ARI and VI
    """
    # Define clustering functions (same as above)
    funcs = {
        'K-Means': lambda data: KMeans(n_clusters=5, random_state=42).fit_predict(data),
        'Agglomerative': lambda data: AgglomerativeClustering(n_clusters=5, linkage='ward').fit_predict(data),
        'DBSCAN': lambda data: DBSCAN(eps=0.5, min_samples=5).fit_predict(data),
        'HDBSCAN': lambda data: hdbscan.HDBSCAN(min_cluster_size=5, min_samples=3).fit_predict(data),
        'CSPA': lambda data: cspa_ensemble([
            KMeans(n_clusters=5, random_state=42).fit_predict(data),
            AgglomerativeClustering(n_clusters=5, linkage='ward').fit_predict(data),
            DBSCAN(eps=0.5, min_samples=5).fit_predict(data)
        ], 5),
        'HGPA': lambda data: hgpa_ensemble([
            KMeans(n_clusters=5, random_state=42).fit_predict(data),
            AgglomerativeClustering(n_clusters=5, linkage='ward').fit_predict(data),
            DBSCAN(eps=0.5, min_samples=5).fit_predict(data)
        ], 5),
        'MCLA': lambda data: mcla_ensemble([
            KMeans(n_clusters=5, random_state=42).fit_predict(data),
            AgglomerativeClustering(n_clusters=5, linkage='ward').fit_predict(data),
            DBSCAN(eps=0.5, min_samples=5).fit_predict(data)
        ], 5)
    }

    noise_stability = {}
    for name in all_labels.keys():
        if name in funcs:
            ari, vi = noise_injection_stability(X, funcs[name], sigma=0.05)
            noise_stability[name] = {'ARI': ari, 'VI': vi}
    return noise_stability