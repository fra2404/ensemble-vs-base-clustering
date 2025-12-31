from __future__ import annotations

from datetime import date
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.cluster import DBSCAN
from sklearn.preprocessing import RobustScaler, StandardScaler


def _project_root() -> Path:
    return Path(__file__).resolve().parent.parent


def _default_dataset_path(dataset: str) -> Path:
    dataset_dir = _project_root() / "dataset"
    mapping = {
        "mall_customers": "Mall_Customers.csv",
        "customer_personality": "customer_personality.csv",
        "wholesale_customers": "wholesale_customers.csv",
    }
    try:
        filename = mapping[dataset]
    except KeyError as exc:
        raise ValueError(
            "Dataset not supported. Choose 'mall_customers', 'customer_personality', or 'wholesale_customers'"
        ) from exc
    return dataset_dir / filename


def load_and_preprocess_data(dataset='mall_customers', filepath=None):
    """
    Load dataset and preprocess.

    Args:
    - dataset: 'mall_customers' or 'customer_personality'
    - filepath: custom filepath, if None uses default

    Returns:
    - X_raw: Original features
    - X: Standardized features
    """
    if filepath is None:
        filepath = _default_dataset_path(dataset)
    filepath = Path(filepath)

    if dataset == 'mall_customers':
        df = pd.read_csv(filepath)
        features = ['Age', 'Annual Income (k$)', 'Spending Score (1-100)']
        X_raw = df[features].dropna()
        scaler = StandardScaler()
        X = scaler.fit_transform(X_raw)
    elif dataset == 'customer_personality':
        df = pd.read_csv(filepath, sep=None, engine='python')  # Auto-detect separator
        # Select relevant features for customer segmentation
        features = ['Year_Birth', 'Income', 'Recency']
        # Convert Year_Birth to Age
        df['Age'] = date.today().year - df['Year_Birth']
        # Convert Income to thousands (k$)
        df['Income'] = df['Income'] / 1000
        features = ['Age', 'Income', 'Recency']

        X_raw = df[features].dropna()

        # Remove outliers using IQR method
        for col in features:
            Q1 = X_raw[col].quantile(0.25)
            Q3 = X_raw[col].quantile(0.75)
            IQR = Q3 - Q1
            lower_bound = Q1 - 1.5 * IQR
            upper_bound = Q3 + 1.5 * IQR
            X_raw = X_raw[(X_raw[col] >= lower_bound) & (X_raw[col] <= upper_bound)]

        # Transform Recency to a 0-100 score (inverse: lower recency = higher score)
        if 'Recency' in X_raw.columns:
            min_recency = X_raw['Recency'].min()
            max_recency = X_raw['Recency'].max()
            X_raw['Recency'] = 100 * (1 - (X_raw['Recency'] - min_recency) / (max_recency - min_recency))

        # Use RobustScaler for better handling of remaining outliers
        scaler = RobustScaler()
        X = scaler.fit_transform(X_raw)
    elif dataset == 'wholesale_customers':
        df = pd.read_csv(filepath)
        # Select 3 main features for customer segmentation
        features = ['Fresh', 'Milk', 'Grocery']
        X_raw = df[features].dropna()

        # Remove outliers using IQR method
        for col in features:
            Q1 = X_raw[col].quantile(0.25)
            Q3 = X_raw[col].quantile(0.75)
            IQR = Q3 - Q1
            lower_bound = Q1 - 1.5 * IQR
            upper_bound = Q3 + 1.5 * IQR
            X_raw = X_raw[(X_raw[col] >= lower_bound) & (X_raw[col] <= upper_bound)]

        # Use RobustScaler
        scaler = RobustScaler()
        X = scaler.fit_transform(X_raw)
    else:
        raise ValueError(
            "Dataset not supported. Choose 'mall_customers', 'customer_personality', or 'wholesale_customers'"
        )
    return X_raw, X


def tune_dbscan_parameters(X):
    """
    Tune DBSCAN parameters using k-distance.

    Returns:
    - eps, min_samples
    """
    from utils import tune_dbscan_eps, compute_all_metrics
    best_sil = -1
    best_params = None
    for min_samples in range(3, 11):
        eps = tune_dbscan_eps(X, min_samples)
        dbscan_temp = DBSCAN(eps=eps, min_samples=min_samples)
        db_labels_temp = dbscan_temp.fit_predict(X)
        metrics_temp = compute_all_metrics(X, db_labels_temp)
        sil = metrics_temp['Silhouette Score']
        if not np.isnan(sil) and sil > best_sil:
            best_sil = sil
            best_params = (eps, min_samples)

    if best_params is None:
        return 0.5, 5  # default
    return best_params