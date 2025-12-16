from utils import cspa_ensemble, hgpa_ensemble, mcla_ensemble


def run_ensemble_methods(base_labels):
    """
    Run ensemble clustering methods.

    Parameters:
    - base_labels: list of label arrays from base algorithms

    Returns:
    - ensemble_results: dict with ensemble labels
    """
    ensemble_results = {}
    ensemble_results['CSPA'] = cspa_ensemble(base_labels)
    ensemble_results['HGPA'] = hgpa_ensemble(base_labels)
    ensemble_results['MCLA'] = mcla_ensemble(base_labels)
    return ensemble_results