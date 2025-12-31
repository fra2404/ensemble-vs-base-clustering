from __future__ import annotations

from pathlib import Path

import matplotlib.pyplot as plt
import seaborn as sns
import pandas as pd
import numpy as np
from sklearn.decomposition import PCA
from sklearn.cluster import KMeans


def plot_clusters_pca(X, labels, title, filename, output_dir="plots"):
    """
    Generate a 2D scatter plot using PCA for clustering results.

    Parameters:
    - X: array-like, shape (n_samples, n_features)
    - labels: array-like, shape (n_samples,)
    - title: str
    - filename: str
    """
    pca = PCA(n_components=2, random_state=42)
    X_pca = pca.fit_transform(X)

    df = pd.DataFrame(X_pca, columns=['PC1', 'PC2'])
    df['Cluster'] = labels

    plt.figure(figsize=(8, 6))
    sns.scatterplot(data=df, x='PC1', y='PC2', hue='Cluster', palette='viridis')
    plt.title(title)
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    plt.savefig(output_dir / f"{filename}.png")
    plt.close()


def plot_similarity_heatmap(similarity_matrix, title, filename, output_dir="plots"):
    """
    Plot heatmap of similarity matrix.

    Parameters:
    - similarity_matrix: array-like
    - title: str
    - filename: str
    """
    plt.figure(figsize=(8, 6))
    sns.heatmap(similarity_matrix, cmap='viridis', square=True)
    plt.title(title)
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    plt.savefig(output_dir / f"{filename}.png")
    plt.close()


def plot_metrics_comparison(metrics_df, filename, output_dir="plots"):
    """
    Bar chart for metrics comparison.

    Parameters:
    - metrics_df: DataFrame
    - filename: str
    """
    metrics_df.plot(kind='bar', figsize=(12, 8))
    plt.title('Clustering Metrics Comparison')
    plt.ylabel('Score')
    plt.xticks(rotation=45)
    plt.tight_layout()
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    plt.savefig(output_dir / f"{filename}.png")
    plt.close()


def plot_stability_boxplot(stability_data, filename, output_dir="plots"):
    """
    Boxplot for stability metrics.

    Parameters:
    - stability_data: dict with keys as algorithms, values as dict of lists
    - filename: str
    """
    _, axes = plt.subplots(2, 1, figsize=(6, 12))

    # ARI
    ari_data = []
    ari_labels = []
    for alg, data in stability_data.items():
        ari_data.extend(data['ARI'])
        ari_labels.extend([alg] * len(data['ARI']))

    df_ari = pd.DataFrame({'ARI': ari_data, 'Algorithm': ari_labels})
    sns.boxplot(data=df_ari, x='Algorithm', y='ARI', ax=axes[0])
    axes[0].set_title('ARI Stability (Bootstrap)')
    axes[0].tick_params(axis='x', rotation=45)

    # VI
    vi_data = []
    vi_labels = []
    for alg, data in stability_data.items():
        vi_data.extend(data['VI'])
        vi_labels.extend([alg] * len(data['VI']))

    df_vi = pd.DataFrame({'VI': vi_data, 'Algorithm': vi_labels})
    sns.boxplot(data=df_vi, x='Algorithm', y='VI', ax=axes[1])
    axes[1].set_title('VI Stability (Bootstrap)')
    axes[1].tick_params(axis='x', rotation=45)

    plt.tight_layout()
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    plt.savefig(output_dir / f"{filename}.png", dpi=300)
    plt.close()


def plot_noise_stability(noise_data, filename, output_dir="plots"):
    """
    Bar chart for noise injection stability.

    Parameters:
    - noise_data: dict
    - filename: str
    """
    df = pd.DataFrame(noise_data).T
    df.plot(kind='bar', figsize=(10, 6))
    plt.title('Stability under Noise Injection (σ=0.05)')
    plt.ylabel('Score')
    plt.xticks(rotation=45)
    plt.tight_layout()
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    plt.savefig(output_dir / f"{filename}.png")
    plt.close()


def plot_metrics_paper_version(results_df, dataset_name, output_dir="plots"):
    """
    Create paper-quality metrics comparison plots with appropriate scales.

    Parameters:
    - results_df: DataFrame with algorithms as index, metrics as columns
    - dataset_name: str, name of the dataset for filename
    """
    fig, axes = plt.subplots(2, 2, figsize=(16, 12))
    fig.suptitle(f'Metrics Comparison - {dataset_name.replace("_", " ").title()}',
                 fontsize=16, fontweight='bold')

    # Colors for consistency
    colors = ['#1f77b4', '#ff7f0e', '#2ca02c', '#d62728', '#9467bd', '#8c564b', '#e377c2']

    # Top-left: Silhouette Score
    results_df['Silhouette Score'].plot(kind='bar', ax=axes[0,0], color=colors[:len(results_df)],
                                  edgecolor='black', linewidth=0.5)
    axes[0,0].set_title('Silhouette Score\n(-1 to 1, higher better)', fontsize=12, fontweight='bold')
    axes[0,0].set_ylabel('Score', fontsize=10)
    axes[0,0].tick_params(axis='x', rotation=45)
    axes[0,0].grid(True, alpha=0.3)

    # Top-right: Davies-Bouldin Index
    results_df['Davies-Bouldin Index'].plot(kind='bar', ax=axes[0,1], color=colors[:len(results_df)],
                                      edgecolor='black', linewidth=0.5)
    axes[0,1].set_title('Davies-Bouldin Index\n(lower better)', fontsize=12, fontweight='bold')
    axes[0,1].set_ylabel('Score', fontsize=10)
    axes[0,1].tick_params(axis='x', rotation=45)
    axes[0,1].grid(True, alpha=0.3)

    # Bottom-left: Calinski-Harabasz Index (log scale)
    results_df['Calinski-Harabasz Index'].plot(kind='bar', ax=axes[1,0], color=colors[:len(results_df)],
                                         edgecolor='black', linewidth=0.5, logy=True)
    axes[1,0].set_title('Calinski-Harabasz Index\n(log scale, higher better)',
                        fontsize=12, fontweight='bold')
    axes[1,0].set_ylabel('Score (log scale)', fontsize=10)
    axes[1,0].tick_params(axis='x', rotation=45)
    axes[1,0].grid(True, alpha=0.3)

    # Bottom-right: Normalized comparison (0-1 scale for all metrics)
    normalized_df = results_df.copy()
    for col in normalized_df.columns:
        if col == 'Davies-Bouldin Index':  # Lower is better - invert normalization
            normalized_df[col] = 1 - (normalized_df[col] - normalized_df[col].min()) / (normalized_df[col].max() - normalized_df[col].min())
        else:  # Higher is better - standard normalization
            normalized_df[col] = (normalized_df[col] - normalized_df[col].min()) / (normalized_df[col].max() - normalized_df[col].min())

    # Average normalized score across all metrics
    avg_normalized = normalized_df.mean(axis=1)
    avg_normalized.plot(kind='bar', ax=axes[1,1], color=colors[:len(avg_normalized)],
                        edgecolor='black', linewidth=0.5)
    axes[1,1].set_title('Average Normalized Score\n(0-1 scale, higher better)',
                        fontsize=12, fontweight='bold')
    axes[1,1].set_ylabel('Normalized Score', fontsize=10)
    axes[1,1].tick_params(axis='x', rotation=45)
    axes[1,1].grid(True, alpha=0.3)

    # Add value labels on bars for the normalized plot
    for i, v in enumerate(avg_normalized):
        axes[1,1].text(i, v + 0.01, f'{v:.3f}', ha='center', va='bottom', fontsize=9)

    plt.tight_layout()
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    plt.savefig(output_dir / f"metrics_paper_{dataset_name}.png", dpi=300, bbox_inches='tight')
    plt.close()

    print(f"Paper-quality metrics plot saved as: {output_dir / f'metrics_paper_{dataset_name}.png'}")


def plot_elbow_analysis(X, k_range, dataset_name, output_dir="plots"):
    """
    Create Elbow plot for K-selection justification.
    
    Parameters:
    - X: Standardized data
    - k_range: Range of k values to test (e.g., range(2, 11))
    - dataset_name: Name of dataset for filename
    """
    inertias = []
    silhouettes = []
    
    for k in k_range:
        kmeans = KMeans(n_clusters=k, random_state=42, n_init=10)
        kmeans.fit(X)
        inertias.append(kmeans.inertia_)
        
        # Calculate silhouette if k > 1
        if k > 1:
            from sklearn.metrics import silhouette_score
            sil = silhouette_score(X, kmeans.labels_)
            silhouettes.append(sil)
        else:
            silhouettes.append(0)
    
    # Create figure with two subplots
    fig, axes = plt.subplots(1, 2, figsize=(14, 5))
    
    # Elbow plot (Inertia)
    axes[0].plot(k_range, inertias, 'bo-', linewidth=2, markersize=8)
    axes[0].set_xlabel('Number of Clusters (k)', fontsize=12)
    axes[0].set_ylabel('Inertia (Within-Cluster Sum of Squares)', fontsize=12)
    axes[0].set_title('Elbow Method for Optimal k', fontsize=14, fontweight='bold')
    axes[0].grid(True, alpha=0.3)
    axes[0].axvline(x=5, color='red', linestyle='--', linewidth=2, alpha=0.7, label='k=5 (chosen)')
    axes[0].legend()
    
    # Silhouette plot
    axes[1].plot(k_range, silhouettes, 'go-', linewidth=2, markersize=8)
    axes[1].set_xlabel('Number of Clusters (k)', fontsize=12)
    axes[1].set_ylabel('Silhouette Score', fontsize=12)
    axes[1].set_title('Silhouette Score vs Number of Clusters', fontsize=14, fontweight='bold')
    axes[1].grid(True, alpha=0.3)
    axes[1].axvline(x=5, color='red', linestyle='--', linewidth=2, alpha=0.7, label='k=5 (chosen)')
    axes[1].legend()
    
    plt.tight_layout()
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    plt.savefig(output_dir / f"elbow_analysis_{dataset_name}.png", dpi=300, bbox_inches='tight')
    plt.close()
    
    print(f"Elbow analysis plot saved as: {output_dir / f'elbow_analysis_{dataset_name}.png'}")
    
    # Return optimal k based on elbow (simple heuristic: max second derivative)
    if len(inertias) > 2:
        # Calculate second derivative (rate of change of rate of change)
        diffs = np.diff(inertias)
        second_diffs = np.diff(diffs)
        optimal_k_elbow = list(k_range)[np.argmax(np.abs(second_diffs)) + 1]
    else:
        optimal_k_elbow = 3
    
    # Optimal k based on silhouette
    optimal_k_sil = list(k_range)[np.argmax(silhouettes)]
    
    print(f"  Optimal k by Elbow method: {optimal_k_elbow}")
    print(f"  Optimal k by Silhouette: {optimal_k_sil}")
    
    return optimal_k_elbow, optimal_k_sil