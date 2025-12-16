import numpy as np


def interpret_clusters(X_raw, labels, algorithm_name, dataset='mall_customers'):
    """
    Interpret clusters based on feature statistics.
    
    Args:
    - X_raw: Original feature DataFrame
    - labels: Cluster labels
    - algorithm_name: Name of the algorithm
    - dataset: Dataset type ('mall_customers', 'customer_personality', 'wholesale_customers')

    Returns:
    - interpretations: list of strings
    """
    interpretations = []
    
    for cluster in np.unique(labels):
        if cluster == -1:  # noise
            continue
        cluster_data = X_raw[labels == cluster]
        
        if dataset == 'mall_customers':
            age_mean = cluster_data['Age'].mean()
            age_std = cluster_data['Age'].std()
            income_mean = cluster_data['Annual Income (k$)'].mean()
            income_std = cluster_data['Annual Income (k$)'].std()
            spending_mean = cluster_data['Spending Score (1-100)'].mean()
            spending_std = cluster_data['Spending Score (1-100)'].std()

            # Age category
            if age_mean < 30:
                age_cat = "Young"
            elif age_mean < 50:
                age_cat = "Middle-aged"
            else:
                age_cat = "Senior"

            # Income category
            if income_mean < 40:
                income_cat = "Low Income"
            elif income_mean < 70:
                income_cat = "Medium Income"
            else:
                income_cat = "High Income"

            # Spending category
            if spending_mean < 40:
                spending_cat = "Low Spending"
            elif spending_mean < 70:
                spending_cat = "Medium Spending"
            else:
                spending_cat = "High Spending"

            interp = f"{algorithm_name} Cluster {cluster}: {age_cat} ({age_mean:.1f}±{age_std:.1f} years), {income_cat} ({income_mean:.1f}k±{income_std:.1f}k), {spending_cat} ({spending_mean:.1f}±{spending_std:.1f})"
            
        elif dataset == 'customer_personality':
            age_mean = cluster_data['Age'].mean()
            age_std = cluster_data['Age'].std()
            income_mean = cluster_data['Income'].mean()  # Already in k$
            income_std = cluster_data['Income'].std()
            recency_mean = cluster_data['Recency'].mean()  # 0-100 score
            recency_std = cluster_data['Recency'].std()

            # Age category
            if age_mean < 35:
                age_cat = "Young"
            elif age_mean < 55:
                age_cat = "Middle-aged"
            else:
                age_cat = "Senior"

            # Income category (already in thousands)
            if income_mean < 35:
                income_cat = "Low Income"
            elif income_mean < 60:
                income_cat = "Medium Income"
            else:
                income_cat = "High Income"

            # Recency/Engagement category (0-100 score, higher = more engaged)
            if recency_mean < 40:
                engagement_cat = "Low Engagement"
            elif recency_mean < 70:
                engagement_cat = "Medium Engagement"
            else:
                engagement_cat = "High Engagement"

            interp = f"{algorithm_name} Cluster {cluster}: {age_cat} ({age_mean:.1f}±{age_std:.1f} years), {income_cat} ({income_mean:.1f}k±{income_std:.1f}k), {engagement_cat} (Score: {recency_mean:.1f}±{recency_std:.1f})"
            
        elif dataset == 'wholesale_customers':
            fresh_mean = cluster_data['Fresh'].mean()
            fresh_std = cluster_data['Fresh'].std()
            milk_mean = cluster_data['Milk'].mean()
            milk_std = cluster_data['Milk'].std()
            grocery_mean = cluster_data['Grocery'].mean()
            grocery_std = cluster_data['Grocery'].std()

            # Fresh products category
            if fresh_mean < 5000:
                fresh_cat = "Low Fresh"
            elif fresh_mean < 15000:
                fresh_cat = "Medium Fresh"
            else:
                fresh_cat = "High Fresh"

            # Milk products category
            if milk_mean < 3000:
                milk_cat = "Low Milk"
            elif milk_mean < 8000:
                milk_cat = "Medium Milk"
            else:
                milk_cat = "High Milk"

            # Grocery category
            if grocery_mean < 4000:
                grocery_cat = "Low Grocery"
            elif grocery_mean < 12000:
                grocery_cat = "Medium Grocery"
            else:
                grocery_cat = "High Grocery"

            interp = f"{algorithm_name} Cluster {cluster}: {fresh_cat} ({fresh_mean:.0f}±{fresh_std:.0f}), {milk_cat} ({milk_mean:.0f}±{milk_std:.0f}), {grocery_cat} ({grocery_mean:.0f}±{grocery_std:.0f})"
        
        else:
            raise ValueError(f"Unknown dataset: {dataset}")
            
        interpretations.append(interp)
    return interpretations


def print_all_interpretations(X_raw, all_labels, dataset='mall_customers'):
    """Print interpretations for all algorithms."""
    for name, labels in all_labels.items():
        print(f"\n{name} Interpretations:")
        interps = interpret_clusters(X_raw, labels, name, dataset)
        for interp in interps:
            print(interp)