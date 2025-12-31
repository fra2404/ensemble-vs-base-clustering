# Ensemble vs Base Clustering for Customer Segmentation

## Overview

This project explores and compares the effectiveness of ensemble clustering methods versus traditional base clustering algorithms for customer segmentation. The analysis is performed on three real-world datasets, and the results are thoroughly evaluated using multiple clustering metrics and stability analyses. The project is accompanied by a scientific paper that details the methodology, results, and insights.

## Project Structure

```
final_submission/
├── code/         # All Python scripts and modules
├── dataset/      # Datasets used for analysis
├── plots/        # Generated plots and visualizations (output)
├── results/      # Output metrics and results (generated at runtime)
├── doc/          # Documentation and paper PDF
├── .gitignore
└── README.md
```

## Datasets

- **Mall_Customers.csv**
- **customer_personality.csv**
- **wholesale_customers.csv**

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

  - Recommended: Use the provided conda environment (e.g. `conda activate ai`).
   - Required packages: `scikit-learn`, `matplotlib`, `seaborn`, `pandas`, `numpy`, `hdbscan`, etc.

2. **Install dependencies:**

```sh
python3 -m pip install -r code/requirements.txt
```

3. **Run the main script (from this `final_submission/` folder):**

```sh
python3 code/main.py --dataset mall_customers
python3 code/main.py --dataset customer_personality
python3 code/main.py --dataset wholesale_customers
```

Outputs:

- Plots: `plots/`
- Metrics CSV: `results/metrics_<dataset>.csv`

Interpretations and summaries are printed to console.

## Paper

A detailed scientific paper describing the methodology, experiments, and findings is available in `doc/main.pdf` (PDF).

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
