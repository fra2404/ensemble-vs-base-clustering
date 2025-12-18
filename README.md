# Ensemble vs Base Clustering for Customer Segmentation

## Overview

This project explores and compares the effectiveness of ensemble clustering methods versus traditional base clustering algorithms for customer segmentation. The analysis is performed on three real-world datasets, and the results are thoroughly evaluated using multiple clustering metrics and stability analyses. The project is accompanied by a scientific paper that details the methodology, results, and insights.

## Project Structure

```
final_submission/
├── code/         # All Python scripts and modules
├── dataset/      # Datasets used for analysis
├── plots/        # Generated plots and visualizations
├── doc/          # Documentation and the scientific paper (PDF)
├── results/      # Output metrics and results
├── .gitignore    # Excludes cache, plots, and temp files
└── README.md     # Project documentation (this file)
```

## Datasets

- **Mall_Customers.csv**
- **Customer_Personality.csv**
- **Wholesale_Customers.csv**

All datasets are located in the `dataset/` folder.

## Clustering Algorithms

- **Base Algorithms:**
  - K-Means
  - Agglomerative Clustering
  - DBSCAN
  - HDBSCAN
- **Ensemble Methods:**
  - CSPA (Cluster-based Similarity Partitioning Algorithm)
  - HGPA (HyperGraph Partitioning Algorithm)
  - MCLA (Meta-CLustering Algorithm)

## Evaluation Metrics

- Silhouette Score
- Davies-Bouldin Index
- Calinski-Harabasz Index
- Average Normalized Score
- Stability Analysis (Bootstrap ARI, Variation of Information)
- Noise Stability

## How to Run

1. **Set up the environment:**

   - Recommended: Use the provided conda environment (see `doc/` for details).
   - Required packages: `scikit-learn`, `matplotlib`, `seaborn`, `pandas`, `numpy`, `hdbscan`, etc.

2. **Run the main script:**

   - From the `code/` directory, you can select the dataset via CLI:
     ```sh
     python3 main.py --dataset mall_customers
     python3 main.py --dataset customer_personality
     python3 main.py --dataset wholesale_customers
     ```
   - All results, plots, and metrics will be saved in the appropriate folders.

3. **Outputs:**
   - Plots: `plots/`
   - Metrics: `results/metrics_<dataset>.csv`
   - Interpretations and summaries: printed to console and saved in `results/`

## Paper

A detailed scientific paper describing the methodology, experiments, and findings is available in the `doc/` folder (PDF format). The paper includes:

- Literature review
- Methodological details
- Algorithmic explanations
- Experimental results
- Discussion and conclusions

## Reproducibility

- All code is parameterized and can be run on any of the provided datasets without modification.
- The project structure and CLI interface ensure easy testing and extension.
- The `.gitignore` excludes only temporary, cache, and output files (e.g., `plots/`, Python cache, LaTeX build files).

## Authors

Francesco Albano [fra2404].

## License

This project is for academic purposes. For reuse or distribution, please contact the author.
