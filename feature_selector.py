import numpy as np
from sklearn.feature_selection import SelectKBest, f_regression, RFE
from sklearn.linear_model import LinearRegression

def filter_correlation(X, threshold=0.95):
    """Filter highly correlated features.
    
    Args:
        X: Input array or DataFrame
        threshold: Correlation threshold (default: 0.95)
    
    Returns:
        List of indices of features to keep
    """
    # Ensure X is a numpy array
    X = np.asarray(X)
    
    # Handle empty or single-column case
    if X.size == 0 or X.shape[1] <= 1:
        return list(range(X.shape[1])) if X.shape[1] > 0 else []
    
    # Only keep columns that are numeric
    numeric_indices = []
    numeric_cols = []
    for i in range(X.shape[1]):
        try:
            X_col = X[:, i].astype(float)
            numeric_indices.append(i)
            numeric_cols.append(X_col)
        except (ValueError, TypeError):
            continue  # skip non-numeric columns
    
    # If no numeric columns, return all indices
    if not numeric_cols:
        return list(range(X.shape[1]))
    
    # Stack numeric columns
    X_numeric = np.stack(numeric_cols, axis=1)
    
    # Compute correlation matrix
    try:
        corr_matrix = np.corrcoef(X_numeric.T)
    except Exception as e:
        print(f"Error computing correlation matrix: {e}")
        return numeric_indices
    
    # Handle single-column case after correlation
    if corr_matrix.shape[0] < 2 or corr_matrix.shape[1] < 2:
        return numeric_indices
    
    # Get upper triangular matrix
    upper = np.triu(np.ones(corr_matrix.shape), k=1).astype(bool)
    
    # Find highly correlated features
    to_drop = [i for i in range(corr_matrix.shape[0]) if any(corr_matrix[i, upper[i]] > threshold)]
    
    # Map back to original indices
    keep_indices = [numeric_indices[i] for i in range(X_numeric.shape[1]) if i not in to_drop]
    
    return keep_indices

def filter_variance(X, threshold=0.01):
    """Filter features with low variance."""
    X = np.asarray(X)
    if X.size == 0:
        return []
    variances = np.var(X, axis=0)
    return [i for i, var in enumerate(variances) if var > threshold]

def select_k_best(X, y, k='all', score_func=f_regression):
    """Select k best features based on statistical tests."""
    selector = SelectKBest(score_func=score_func, k=k)
    selector.fit(X, y)
    return selector.scores_

def recursive_feature_elimination(X, y, n_features_to_select=5):
    """Perform recursive feature elimination."""
    model = LinearRegression()
    rfe = RFE(estimator=model, n_features_to_select=n_features_to_select)
    rfe.fit(X, y)
    return rfe.support_ 