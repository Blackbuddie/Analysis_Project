from pyspark.ml.classification import (
    RandomForestClassifier, LogisticRegression, 
    GBTClassifier, DecisionTreeClassifier,
    MultilayerPerceptronClassifier, NaiveBayes
)
from pyspark.ml.regression import (
    LinearRegression, RandomForestRegressor,
    GBTRegressor, DecisionTreeRegressor
)
from pyspark.ml.clustering import (
    KMeans, BisectingKMeans, GaussianMixture
)
from pyspark.ml.feature import VectorAssembler, StringIndexer
from pyspark.ml.evaluation import (
    MulticlassClassificationEvaluator,
    RegressionEvaluator,
    ClusteringEvaluator
)
from pyspark.ml.tuning import CrossValidator, ParamGridBuilder
import logging

logger = logging.getLogger(__name__)

class ModelTrainer:
    def __init__(self, spark_session):
        self.spark = spark_session
        
    def prepare_features(self, df, feature_cols, target_col=None):
        """Prepare features for model training"""
        try:
            # Create vector assembler for features
            assembler = VectorAssembler(inputCols=feature_cols, outputCol="features")
            assembled_df = assembler.transform(df)
            
            if target_col:
                # Handle categorical target variable
                if not self._is_numeric(df, target_col):
                    indexer = StringIndexer(inputCol=target_col, outputCol="label")
                    assembled_df = indexer.fit(assembled_df).transform(assembled_df)
                else:
                    assembled_df = assembled_df.withColumnRenamed(target_col, "label")
                
            return assembled_df
        except Exception as e:
            logger.error(f"Error preparing features: {str(e)}")
            raise
    
    def _is_numeric(self, df, column_name):
        """Check if a column is numeric"""
        return df.schema[column_name].dataType.typeName() in ['double', 'float', 'int', 'long']
    
    def train_classifier(self, df, feature_cols, target_col, algorithm='random_forest', params=None):
        """Train a classification model with cross-validation"""
        try:
            # Prepare data
            prepared_df = self.prepare_features(df, feature_cols, target_col)
            train_data, test_data = prepared_df.randomSplit([0.8, 0.2])
            
            # Select base model
            if algorithm == 'random_forest':
                model = RandomForestClassifier(
                    featuresCol="features",
                    labelCol="label",
                    numTrees=10
                )
                param_grid = ParamGridBuilder() \
                    .addGrid(model.numTrees, [10, 20, 30]) \
                    .addGrid(model.maxDepth, [5, 10, 15]) \
                    .build()
            
            elif algorithm == 'gbt':
                model = GBTClassifier(
                    featuresCol="features",
                    labelCol="label"
                )
                param_grid = ParamGridBuilder() \
                    .addGrid(model.maxDepth, [5, 10, 15]) \
                    .addGrid(model.maxIter, [10, 20, 30]) \
                    .build()
            
            elif algorithm == 'logistic':
                model = LogisticRegression(
                    featuresCol="features",
                    labelCol="label"
                )
                param_grid = ParamGridBuilder() \
                    .addGrid(model.regParam, [0.01, 0.1, 0.3]) \
                    .addGrid(model.elasticNetParam, [0.0, 0.5, 1.0]) \
                    .build()
            
            elif algorithm == 'decision_tree':
                model = DecisionTreeClassifier(
                    featuresCol="features",
                    labelCol="label"
                )
                param_grid = ParamGridBuilder() \
                    .addGrid(model.maxDepth, [5, 10, 15]) \
                    .addGrid(model.impurity, ['gini', 'entropy']) \
                    .build()
            
            elif algorithm == 'neural_network':
                # Define the layers
                layers = [len(feature_cols), 10, 5, 2]  # Adjust based on your needs
                model = MultilayerPerceptronClassifier(
                    featuresCol="features",
                    labelCol="label",
                    layers=layers
                )
                param_grid = ParamGridBuilder() \
                    .addGrid(model.maxIter, [100, 200]) \
                    .addGrid(model.blockSize, [64, 128]) \
                    .build()
            
            else:
                raise ValueError(f"Unsupported classification algorithm: {algorithm}")
            
            # Create evaluator
            evaluator = MulticlassClassificationEvaluator(
                labelCol="label",
                predictionCol="prediction",
                metricName="accuracy"
            )
            
            # Create CrossValidator
            cv = CrossValidator(
                estimator=model,
                estimatorParamMaps=param_grid,
                evaluator=evaluator,
                numFolds=3
            )
            
            # Train model with cross-validation
            cv_model = cv.fit(train_data)
            best_model = cv_model.bestModel
            
            # Evaluate on test set
            predictions = best_model.transform(test_data)
            accuracy = evaluator.evaluate(predictions)
            
            # Get feature importance if available
            feature_importance = self._get_feature_importance(best_model, feature_cols)
            
            return {
                'model': best_model,
                'accuracy': accuracy,
                'feature_importance': feature_importance,
                'best_params': best_model.extractParamMap(),
                'cv_metrics': cv_model.avgMetrics
            }
        except Exception as e:
            logger.error(f"Error training classifier: {str(e)}")
            raise
    
    def train_regressor(self, df, feature_cols, target_col, algorithm='random_forest', params=None):
        """Train a regression model with cross-validation"""
        try:
            # Prepare data
            prepared_df = self.prepare_features(df, feature_cols, target_col)
            train_data, test_data = prepared_df.randomSplit([0.8, 0.2])
            
            # Select base model
            if algorithm == 'random_forest':
                model = RandomForestRegressor(
                    featuresCol="features",
                    labelCol="label"
                )
                param_grid = ParamGridBuilder() \
                    .addGrid(model.numTrees, [10, 20, 30]) \
                    .addGrid(model.maxDepth, [5, 10, 15]) \
                    .build()
            
            elif algorithm == 'gbt':
                model = GBTRegressor(
                    featuresCol="features",
                    labelCol="label"
                )
                param_grid = ParamGridBuilder() \
                    .addGrid(model.maxDepth, [5, 10, 15]) \
                    .addGrid(model.maxIter, [10, 20, 30]) \
                    .build()
            
            elif algorithm == 'linear':
                model = LinearRegression(
                    featuresCol="features",
                    labelCol="label"
                )
                param_grid = ParamGridBuilder() \
                    .addGrid(model.regParam, [0.01, 0.1, 0.3]) \
                    .addGrid(model.elasticNetParam, [0.0, 0.5, 1.0]) \
                    .build()
            
            elif algorithm == 'decision_tree':
                model = DecisionTreeRegressor(
                    featuresCol="features",
                    labelCol="label"
                )
                param_grid = ParamGridBuilder() \
                    .addGrid(model.maxDepth, [5, 10, 15]) \
                    .build()
            
            else:
                raise ValueError(f"Unsupported regression algorithm: {algorithm}")
            
            # Create evaluator
            evaluator = RegressionEvaluator(
                labelCol="label",
                predictionCol="prediction",
                metricName="rmse"
            )
            
            # Create CrossValidator
            cv = CrossValidator(
                estimator=model,
                estimatorParamMaps=param_grid,
                evaluator=evaluator,
                numFolds=3
            )
            
            # Train model with cross-validation
            cv_model = cv.fit(train_data)
            best_model = cv_model.bestModel
            
            # Evaluate on test set
            predictions = best_model.transform(test_data)
            rmse = evaluator.evaluate(predictions)
            
            # Calculate additional metrics
            r2_evaluator = RegressionEvaluator(
                labelCol="label",
                predictionCol="prediction",
                metricName="r2"
            )
            r2 = r2_evaluator.evaluate(predictions)
            
            return {
                'model': best_model,
                'rmse': rmse,
                'r2': r2,
                'feature_importance': self._get_feature_importance(best_model, feature_cols),
                'best_params': best_model.extractParamMap(),
                'cv_metrics': cv_model.avgMetrics
            }
        except Exception as e:
            logger.error(f"Error training regressor: {str(e)}")
            raise
    
    def train_clustering(self, df, feature_cols, algorithm='kmeans', n_clusters=3):
        """Train a clustering model"""
        try:
            # Prepare data
            prepared_df = self.prepare_features(df, feature_cols)
            
            # Select base model
            if algorithm == 'kmeans':
                model = KMeans(
                    featuresCol="features",
                    k=n_clusters,
                    seed=42
                )
            elif algorithm == 'bisecting_kmeans':
                model = BisectingKMeans(
                    featuresCol="features",
                    k=n_clusters,
                    seed=42
                )
            elif algorithm == 'gaussian_mixture':
                model = GaussianMixture(
                    featuresCol="features",
                    k=n_clusters,
                    seed=42
                )
            else:
                raise ValueError(f"Unsupported clustering algorithm: {algorithm}")
            
            # Train model
            model = model.fit(prepared_df)
            
            # Make predictions
            predictions = model.transform(prepared_df)
            
            # Calculate silhouette score
            evaluator = ClusteringEvaluator()
            silhouette = evaluator.evaluate(predictions)
            
            # Get cluster sizes
            cluster_sizes = predictions.groupBy("prediction").count().collect()
            cluster_size_dict = {f"Cluster {int(row['prediction'])}": int(row['count']) for row in cluster_sizes}
            
            # Format cluster centers if available
            cluster_centers_raw = model.clusterCenters() if hasattr(model, 'clusterCenters') else None
            formatted_centers = []
            if cluster_centers_raw:
                for i, center in enumerate(cluster_centers_raw):
                    # Create a dictionary with cluster ID and feature values
                    center_dict = {
                        "Cluster ID": i,
                        **{feature_cols[j]: round(float(center[j]), 2) for j in range(len(feature_cols))}
                    }
                    formatted_centers.append(center_dict)
            
            # Return formatted results
            return {
                'model': model,
                'silhouette_score': round(silhouette, 4),
                'cluster_sizes': cluster_size_dict,
                'cluster_centers': formatted_centers,
                'algorithm': algorithm,
                'n_clusters': n_clusters
            }
            
        except Exception as e:
            logger.error(f"Error training clustering model: {str(e)}")
            raise
    
    def _get_feature_importance(self, model, feature_cols):
        """Extract feature importance if available"""
        try:
            if hasattr(model, 'featureImportances'):
                return dict(zip(feature_cols, model.featureImportances.toArray()))
            return None
        except Exception as e:
            logger.warning(f"Could not extract feature importance: {str(e)}")
            return None 