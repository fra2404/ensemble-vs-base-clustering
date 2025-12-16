import argparse
import os
import numpy as np
from data_preprocessing import load_and_preprocess_data, tune_dbscan_parameters
from clustering_algorithms import run_all_base_algorithms
from ensemble_methods import run_ensemble_methods
from evaluation import evaluate_all_algorithms, perform_stability_analysis, perform_noise_stability_analysis
from interpretation import print_all_interpretations
from plots import plot_clusters_pca, plot_similarity_heatmap, plot_metrics_comparison, plot_stability_boxplot, plot_noise_stability, plot_metrics_paper_version, plot_elbow_analysis
from utils import cluster_sizes, hgpa_ensemble, mcla_ensemble
from sklearn.cluster import AgglomerativeClustering, DBSCAN, KMeans

# CLI configuration: choose dataset without modifying code
parser = argparse.ArgumentParser(description="Run clustering analysis on a chosen dataset")
parser.add_argument(
    "--dataset",
    "-d",
    choices=["mall_customers", "customer_personality", "wholesale_customers"],
    default="mall_customers",
    help="Dataset to analyze",
)
args = parser.parse_args()
DATASET = args.dataset

# Create directories
os.makedirs('plots', exist_ok=True)
os.makedirs('results', exist_ok=True)

# Load and preprocess data
X_raw, X = load_and_preprocess_data(dataset=DATASET)
print(f"Data loaded and standardized for {DATASET}. Shape:", X.shape)

# Elbow analysis for k-selection justification
print("\n" + "="*50)
print("K-SELECTION ANALYSIS (Elbow Method)")
print("="*50)
optimal_k_elbow, optimal_k_sil = plot_elbow_analysis(X, range(2, 11), DATASET)
print(f"Using k=5 for this analysis (balances quality and interpretability)")
print("="*50 + "\n")

# Tune DBSCAN
eps_db, min_samples_db = tune_dbscan_parameters(X)
print(f"DBSCAN params: eps={eps_db:.3f}, min_samples={min_samples_db}")

# Run base algorithms
base_results = run_all_base_algorithms(X, n_clusters=5, eps=eps_db, min_samples=min_samples_db)
algorithms = {k: v for k, v in base_results.items() if not k.endswith(' Model')}

# Run ensemble methods
ensemble_results = run_ensemble_methods([base_results['K-Means'], base_results['Agglomerative'], base_results['DBSCAN']])
algorithms.update(ensemble_results)

print("All clustering completed.")

# Evaluate
metrics_df = evaluate_all_algorithms(X, algorithms)
metrics_df.to_csv(f'results/metrics_{DATASET}.csv')
print(f"Metrics computed and saved to results/metrics_{DATASET}.csv")
print(metrics_df)

# Interpretations
print_all_interpretations(X_raw, algorithms, dataset=DATASET)

# Cluster sizes
print("\nCluster Sizes:")
for name, labels in algorithms.items():
    sizes = cluster_sizes(labels)
    print(f"{name}: {sizes}")

# K-analysis (simplified)
print("\nK-analysis for K-Means, HGPA, MCLA (k=3-7):")
k_range = range(3, 8)
for k in k_range:
    km_labels = KMeans(n_clusters=k, random_state=42).fit_predict(X)
    agg_labels = AgglomerativeClustering(n_clusters=k, linkage='ward').fit_predict(X)
    db_labels = DBSCAN(eps=eps_db, min_samples=min_samples_db).fit_predict(X)
    hg_labels = hgpa_ensemble([km_labels, agg_labels, db_labels], k)
    mc_labels = mcla_ensemble([km_labels, agg_labels, db_labels], k)
    km_metrics = evaluate_all_algorithms(X, {'K-Means': km_labels}).loc['K-Means']
    hg_metrics = evaluate_all_algorithms(X, {'HGPA': hg_labels}).loc['HGPA']
    mc_metrics = evaluate_all_algorithms(X, {'MCLA': mc_labels}).loc['MCLA']
    print(f"k={k}: K-Means Sil={km_metrics['Silhouette Score']:.3f}, HGPA Sil={hg_metrics['Silhouette Score']:.3f}, MCLA Sil={mc_metrics['Silhouette Score']:.3f}")

# Stability
stability_data = perform_stability_analysis(X, algorithms)
noise_data = perform_noise_stability_analysis(X, algorithms)
print("Stability analysis completed.")

# Plots
for name, labels in algorithms.items():
    plot_clusters_pca(X, labels, f'{name} Clusters (PCA)', f'{name.lower()}_pca_{DATASET}')

# Similarity for CSPA
n_samples = len(X)
similarity = np.zeros((n_samples, n_samples))
base_labels_list = [base_results['K-Means'], base_results['Agglomerative'], base_results['DBSCAN']]
for i in range(n_samples):
    for j in range(n_samples):
        sim = sum(1 for labels in base_labels_list if labels[i] == labels[j])
        similarity[i, j] = sim / len(base_labels_list)
plot_similarity_heatmap(similarity, 'CSPA Similarity Matrix', f'cspa_similarity_{DATASET}')

plot_metrics_comparison(metrics_df[['Silhouette Score', 'Davies-Bouldin Index', 'Calinski-Harabasz Index']], f'metrics_comparison_{DATASET}')
plot_metrics_paper_version(metrics_df[['Silhouette Score', 'Davies-Bouldin Index', 'Calinski-Harabasz Index']], DATASET)
plot_stability_boxplot(stability_data, f'stability_bootstrap_{DATASET}')
plot_noise_stability(noise_data, f'stability_noise_{DATASET}')

print("All plots generated in plots/ directory.")

# Summary
print("\nStability Summary (Bootstrap ARI mean, VI mean):")
for name in stability_data:
    ari_mean = np.mean(stability_data[name]['ARI'])
    vi_mean = np.mean(stability_data[name]['VI'])
    print(f"{name}: ARI={ari_mean:.3f}, VI={vi_mean:.3f}")

print("\nNoise Stability (ARI, VI):")
for name in noise_data:
    ari = noise_data[name]['ARI']
    vi = noise_data[name]['VI']
    print(f"{name}: ARI={ari:.3f}, VI={vi:.3f}")

# Recommendation
best_sil = metrics_df['Silhouette Score'].max()
best_method = metrics_df['Silhouette Score'].idxmax()
print(f"\nBest method by Silhouette: {best_method} ({best_sil:.3f})")

best_stable = max(stability_data, key=lambda x: np.mean(stability_data[x]['ARI']))
print(f"Most stable method: {best_stable}")

print("\nProject completed successfully!")
print("Check plots/ for visualizations and results/metrics.csv for metrics.")