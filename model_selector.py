import numpy as np

def suggest_tasks(df):
    numeric_cols = df.select_dtypes(include=[np.number]).columns.tolist()
    categorical_cols = df.select_dtypes(include=['object']).columns.tolist()
    return {
        'regression': len(numeric_cols) > 1,
        'classification': len(categorical_cols) > 0,
        'clustering': len(numeric_cols) > 2,
        'feature_importance': True,
        'correlation_analysis': len(numeric_cols) > 1
    } 