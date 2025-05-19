import os
import plotly.graph_objects as go
import json
from plotly.utils import PlotlyJSONEncoder

class Visualizer:
    def __init__(self, static_path):
        self.static_path = static_path

    def create_feature_importance_plot(self, feature_importance):
        """Create a bar plot of feature importances using Plotly."""
        if not feature_importance:
            return None
            
        # Sort features by importance
        sorted_features = sorted(feature_importance.items(), key=lambda x: x[1], reverse=True)
        features = [x[0] for x in sorted_features]
        importances = [x[1] for x in sorted_features]
        
        # Create the bar plot
        fig = go.Figure(data=[
            go.Bar(
                x=features,
                y=importances,
                text=[f'{imp:.3f}' for imp in importances],
                textposition='auto',
            )
        ])
        
        # Update layout
        fig.update_layout(
            title='Feature Importance',
            xaxis_title='Features',
            yaxis_title='Importance',
            showlegend=False,
            xaxis={'tickangle': 45},
            height=400,
            margin=dict(b=100)  # Add bottom margin for rotated labels
        )
        
        # Convert figure to JSON-serializable format
        return {
            'data': json.loads(json.dumps(fig.data, cls=PlotlyJSONEncoder)),
            'layout': json.loads(json.dumps(fig.layout, cls=PlotlyJSONEncoder))
        }

    def create_regression_results_plot(self, y_true, y_pred):
        """Create a scatter plot of true vs predicted values."""
        fig = go.Figure(data=[
            go.Scatter(
                x=y_true,
                y=y_pred,
                mode='markers',
                name='Predictions'
            )
        ])
        
        # Add perfect prediction line
        min_val = min(min(y_true), min(y_pred))
        max_val = max(max(y_true), max(y_pred))
        fig.add_trace(go.Scatter(
            x=[min_val, max_val],
            y=[min_val, max_val],
            mode='lines',
            name='Perfect Prediction',
            line=dict(dash='dash', color='red')
        ))
        
        # Update layout
        fig.update_layout(
            title='True vs Predicted Values',
            xaxis_title='True Values',
            yaxis_title='Predicted Values',
            showlegend=True,
            height=400
        )
        
        # Convert figure to JSON-serializable format
        return {
            'data': json.loads(json.dumps(fig.data, cls=PlotlyJSONEncoder)),
            'layout': json.loads(json.dumps(fig.layout, cls=PlotlyJSONEncoder))
        }

    def create_clustering_plot(self, df, feature_cols, clusters, cluster_centers):
        """Create a scatter plot of clusters."""
        if len(feature_cols) < 2:
            return None
            
        # Create scatter plot
        fig = go.Figure()
        
        # Add data points
        for cluster_id in clusters.unique():
            mask = clusters == cluster_id
            fig.add_trace(go.Scatter(
                x=df[feature_cols[0]][mask],
                y=df[feature_cols[1]][mask],
                mode='markers',
                name=f'Cluster {cluster_id}'
            ))
        
        # Add cluster centers if available
        if cluster_centers:
            fig.add_trace(go.Scatter(
                x=[center[feature_cols[0]] for center in cluster_centers],
                y=[center[feature_cols[1]] for center in cluster_centers],
                mode='markers',
                marker=dict(
                    symbol='star',
                    size=15,
                    color='black'
                ),
                name='Cluster Centers'
            ))
        
        # Update layout
        fig.update_layout(
            title='Cluster Visualization',
            xaxis_title=feature_cols[0],
            yaxis_title=feature_cols[1],
            showlegend=True,
            height=400
        )
        
        # Convert figure to JSON-serializable format
        return {
            'data': json.loads(json.dumps(fig.data, cls=PlotlyJSONEncoder)),
            'layout': json.loads(json.dumps(fig.layout, cls=PlotlyJSONEncoder))
        }

    def create_correlation_matrix(self, df, feature_cols):
        """Create a correlation matrix heatmap."""
        # Calculate correlation matrix
        corr_matrix = df[feature_cols].corr()
        
        # Create heatmap
        fig = go.Figure(data=go.Heatmap(
            z=corr_matrix.values,
            x=corr_matrix.columns,
            y=corr_matrix.columns,
            colorscale='RdBu',
            zmin=-1,
            zmax=1
        ))
        
        # Update layout
        fig.update_layout(
            title='Correlation Matrix',
            height=400,
            margin=dict(b=100)  # Add bottom margin for rotated labels
        )
        
        # Convert figure to JSON-serializable format
        return {
            'data': json.loads(json.dumps(fig.data, cls=PlotlyJSONEncoder)),
            'layout': json.loads(json.dumps(fig.layout, cls=PlotlyJSONEncoder))
        } 