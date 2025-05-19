from pyspark.ml.feature import VectorAssembler
from pyspark.ml.regression import LinearRegression, LinearRegressionModel
from pyspark.ml.classification import LogisticRegression
from pyspark.ml.clustering import KMeans
from pyspark.sql.functions import count, isnan, when
from pyspark.ml.feature import StringIndexer
from pyspark.ml import Pipeline
from pyspark.sql.types import IntegerType
import json
import numpy as np
import pandas as pd
import logging
import traceback
from pyspark.ml import PipelineModel
from pyspark.sql.types import StringType, DoubleType
import os
import uuid
from config import Config

logger = logging.getLogger(__name__)

class DataAnalyzer:
    def __init__(self, spark_manager, file_content=None, file_extension=None):
        self.spark_manager = spark_manager
        self.file_content = file_content
        self.file_extension = file_extension
        self.df = None
        if file_content is not None and file_extension is not None:
            self.load_data()
    
    def load_data(self):
        """Load data directly from content into Spark DataFrame"""
        if self.file_content is not None and self.file_extension is not None:
            self.df = self.spark_manager.read_file_content(self.file_content, self.file_extension)
    
    def set_dataframe(self, df):
        """Set an existing Spark DataFrame"""
        self.df = df
    
    def _convert_to_serializable(self, obj):
        """Convert objects to JSON serializable format."""
        try:
            if hasattr(obj, 'toPandas'):  # Spark DataFrame
                return json.loads(obj.toPandas().to_json(orient='records'))
            if isinstance(obj, pd.DataFrame):
                return json.loads(obj.to_json(orient='records'))
            if isinstance(obj, pd.Series):
                return obj.tolist()
            if isinstance(obj, np.ndarray):
                return obj.tolist()
            if isinstance(obj, (np.generic, np.int64, np.int32, np.float64, np.float32)):
                return obj.item()
            if isinstance(obj, dict):
                return {self._convert_to_serializable(k): self._convert_to_serializable(v) for k, v in obj.items()}
            if isinstance(obj, list):
                return [self._convert_to_serializable(item) for item in obj]
            if hasattr(obj, '__dict__'):  # Fallback for custom objects
                return self._convert_to_serializable(obj.__dict__)
            return obj
        except Exception as e:
            logger.error(f"Serialization error: {e}")
            return str(obj)

    def initial_analysis(self):
        """Perform initial analysis of the data"""
        # Convert Spark DataFrame to Pandas for analysis
        pandas_df = self.df.toPandas()
        
        # Get basic information about the dataset
        columns = pandas_df.columns.tolist()
        row_count = len(pandas_df)
        
        # Analyze data types and null counts for each column
        column_analysis = []
        for column in columns:
            column_info = {
                'name': column,
                'type': str(pandas_df[column].dtype),
                'null_count': int(pandas_df[column].isnull().sum()),
                'distinct_count': int(pandas_df[column].nunique()),
                'sample_values': pandas_df[column].dropna().head(5).tolist()
            }
            
            # Add numeric statistics if applicable
            if np.issubdtype(pandas_df[column].dtype, np.number):
                column_info.update({
                    'min': float(pandas_df[column].min()),
                    'max': float(pandas_df[column].max()),
                    'mean': float(pandas_df[column].mean()),
                    'std': float(pandas_df[column].std())
                })
            
            column_analysis.append(column_info)
        
        result = {
            'row_count': row_count,
            'column_count': len(columns),
            'columns': column_analysis,
            'correlation_matrix': self._convert_to_serializable(
                pandas_df.select_dtypes(include=[np.number]).corr()
            ) if len(pandas_df.select_dtypes(include=[np.number]).columns) > 1 else None
        }
        
        return self._convert_to_serializable(result)
    
    def _prepare_ml_dataframe(self, df_in, feature_col_names, target_col_name=None, \
                              features_vec_col="features", label_col="label"):
        """
        Prepares a DataFrame for ML by indexing string features/target and assembling features.
        Returns the transformed DataFrame and a list of the column names that were input to VectorAssembler.
        """
        current_df = df_in
        
        assembler_input_feature_names = []
        pipeline_stages = []

        # Process feature columns for indexing
        for col_name in feature_col_names:
            # Ensure column exists before checking its type
            if col_name not in current_df.columns:
                raise ValueError(f"Feature column '{col_name}' not found in DataFrame.")
            
            if dict(current_df.dtypes)[col_name] == 'string':
                indexed_name = col_name + "_indexed"
                # Avoid adding duplicate stages if a feature is listed multiple times (though unlikely)
                if not any(isinstance(stage, StringIndexer) and stage.getInputCol() == col_name and stage.getOutputCol() == indexed_name for stage in pipeline_stages):
                    pipeline_stages.append(StringIndexer(inputCol=col_name, outputCol=indexed_name, handleInvalid="keep"))
                assembler_input_feature_names.append(indexed_name)
            else: # Numeric or other type suitable for assembler
                assembler_input_feature_names.append(col_name)
        
        final_label_col_name_in_df = None
        # Process target column for indexing or renaming
        if target_col_name:
            if target_col_name not in current_df.columns:
                raise ValueError(f"Target column '{target_col_name}' not found in DataFrame.")

            final_label_col_name_in_df = label_col
            if dict(current_df.dtypes)[target_col_name] == 'string':
                # Avoid adding duplicate stage for target if it was also a feature (though features/target are usually distinct)
                 if not any(isinstance(stage, StringIndexer) and stage.getInputCol() == target_col_name and stage.getOutputCol() == label_col for stage in pipeline_stages):
                    pipeline_stages.append(StringIndexer(inputCol=target_col_name, outputCol=label_col, handleInvalid="keep"))
            # Numeric target renaming will be handled after pipeline transformation to avoid issues if target is also a feature processed by pipeline
            
        # Apply StringIndexer stages if any
        if pipeline_stages:
            transform_pipeline = Pipeline(stages=pipeline_stages)
            current_df = transform_pipeline.fit(current_df).transform(current_df)

        # Rename numeric target to 'label' if necessary (must be done after pipeline)
        if target_col_name and final_label_col_name_in_df:
            if dict(df_in.dtypes)[target_col_name] != 'string': # Original type was not string
                if target_col_name != label_col: # And it's not already named 'label'
                    current_df = current_df.withColumnRenamed(target_col_name, label_col)
                # Cast the numeric label column to IntegerType
                logger.info(f"Casting numeric target column '{label_col}' to IntegerType.")
                current_df = current_df.withColumn(label_col, current_df[label_col].cast(IntegerType()))
            # else: if original target was string, StringIndexer already produced numeric (double) labels
            # which are generally fine. If specific models require integer, further casting might be needed here too.
            # For now, let's assume StringIndexer output (0.0, 1.0, ..) is acceptable or implicitly handled.
            # The error specifically mentioned non-integer float from a non-string source.
        
        # --- Filter out rows with nulls in the label column ---
        if final_label_col_name_in_df: # This is true if target_col_name was provided
            logger.info(f"Filtering out rows where the label column '{label_col}' is null.")
            before_filter_count = current_df.count()
            current_df = current_df.na.drop(subset=[label_col])
            after_filter_count = current_df.count()
            logger.info(f"Rows before label null filter: {before_filter_count}, after: {after_filter_count}. Removed: {before_filter_count - after_filter_count}")
            if after_filter_count == 0:
                raise ValueError("All rows were removed after filtering for null labels. Check your target column.")

        # Assemble features
        # Ensure assembler_input_feature_names are unique and valid columns in current_df
        valid_assembler_inputs = []
        seen_cols = set()
        for col in assembler_input_feature_names:
            if col in current_df.columns and col not in seen_cols:
                valid_assembler_inputs.append(col)
                seen_cols.add(col)
            elif col not in current_df.columns:
                logger.warning(f"Column '{col}' intended for assembler not found after pipeline. Skipping.")

        if not valid_assembler_inputs:
            raise ValueError("No valid feature columns available for VectorAssembler after processing.")

        assembler = VectorAssembler(inputCols=valid_assembler_inputs, outputCol="features", handleInvalid="skip")
        ml_ready_df = assembler.transform(current_df)
        
        return ml_ready_df, valid_assembler_inputs

    def suggest_analysis_type(self, target_feature=None):
        """Suggest appropriate analysis types based on data characteristics"""
        if not target_feature:
            return ['clustering', 'exploratory']
        
        col_type = str(self.df.schema[target_feature].dataType)
        distinct_count = self.df.select(target_feature).distinct().count()
        
        if 'int' in col_type.lower() or 'double' in col_type.lower():
            if distinct_count < 10:
                return ['classification']
            return ['regression']
        elif 'string' in col_type.lower():
            return ['classification']
        
        return ['exploratory']
    
    def analyze(self, analysis_type, selected_features=None, target_feature=None):
        """Perform the specified type of analysis"""
        if not selected_features:
            selected_features = self.df.columns
        
        if analysis_type == 'exploratory':
            return self._exploratory_analysis(selected_features)
        elif analysis_type == 'regression':
            return self._regression_analysis(selected_features, target_feature)
        elif analysis_type == 'classification':
            return self._classification_analysis(selected_features, target_feature)
        elif analysis_type == 'clustering':
            return self._clustering_analysis(selected_features)
        else:
            raise ValueError(f"Unsupported analysis type: {analysis_type}")
    
    def _exploratory_analysis(self, selected_features):
        """Perform exploratory data analysis"""
        pandas_df = self.df.select(selected_features).toPandas()
        results = {}
        
        # Add overall dataset information
        results['dataset_info'] = {
            'total_rows': len(pandas_df),
            'total_columns': len(selected_features),
            'missing_values_percentage': {
                col: float(pandas_df[col].isnull().sum() / len(pandas_df) * 100)
                for col in selected_features
            }
        }
        
        for feature in selected_features:
            if np.issubdtype(pandas_df[feature].dtype, np.number):
                # Numeric column analysis
                stats = pandas_df[feature].describe()
                results[feature] = {
                    'type': 'numeric',
                    'stats': {
                        'min': float(stats['min']),
                        'max': float(stats['max']),
                        'mean': float(stats['mean']),
                        'std': float(stats['std']),
                        'quartiles': {
                            '25%': float(stats['25%']),
                            '50%': float(stats['50%']),
                            '75%': float(stats['75%'])
                        }
                    },
                    'histogram': self._create_histogram(pandas_df[feature]),
                    'missing_values': int(pandas_df[feature].isnull().sum()),
                    'missing_percentage': float(pandas_df[feature].isnull().sum() / len(pandas_df) * 100),
                    'unique_values': int(pandas_df[feature].nunique())
                }
                
                # Add correlation with other numeric columns
                numeric_cols = pandas_df.select_dtypes(include=[np.number]).columns
                if len(numeric_cols) > 1:
                    correlations = pandas_df[numeric_cols].corr()[feature].to_dict()
                    results[feature]['correlations'] = {
                        k: float(v) for k, v in correlations.items() if k != feature
                    }
            else:
                # Categorical column analysis
                # Convert dictionary objects to strings before analysis
                pandas_df[feature] = pandas_df[feature].apply(lambda x: str(x) if isinstance(x, dict) else x)
                value_counts = pandas_df[feature].value_counts()
                top_values = value_counts.head(10)
                results[feature] = {
                    'type': 'categorical',
                    'top_values': {str(k): int(v) for k, v in zip(top_values.index, top_values.values)},
                    'missing_values': int(pandas_df[feature].isnull().sum()),
                    'missing_percentage': float(pandas_df[feature].isnull().sum() / len(pandas_df) * 100),
                    'unique_values': int(pandas_df[feature].nunique()),
                    'most_common_value': str(value_counts.index[0]) if len(value_counts) > 0 else None,
                    'least_common_value': str(value_counts.index[-1]) if len(value_counts) > 0 else None
                }
        
        return self._convert_to_serializable(results)
    
    def _create_histogram(self, series):
        """Create histogram data for numeric columns"""
        try:
            # Convert series to numpy array if it's not already
            if isinstance(series, list):
                series = np.array(series)
            elif isinstance(series, pd.Series):
                series = series.values
            
            # Remove NaN values
            series = series[~np.isnan(series)]
            
            if len(series) == 0:
                return {
                    'counts': [],
                    'bin_edges': []
                }
            
            hist, bin_edges = np.histogram(series, bins='auto')
            return {
                'counts': hist.tolist(),
                'bin_edges': bin_edges.tolist()
            }
        except Exception as e:
            logger.error(f"Error creating histogram: {str(e)}")
            return {
                'counts': [],
                'bin_edges': []
            }
    
    def _regression_analysis(self, selected_features, target_feature):
        """Perform regression analysis, save the model, and return metrics & model info."""
        try:
            logger.info(f"Starting regression analysis. Features: {selected_features}, Target: {target_feature}")
            original_feature_cols = [f for f in selected_features if f != target_feature] # Features user selected for model input
            
            if not original_feature_cols:
                raise ValueError("No feature columns selected for regression.")
            if not target_feature or target_feature not in self.df.columns:
                raise ValueError(f"Target column '{target_feature}' not found or not specified.")

            # --- Define Pipeline Stages ---
            stages = []
            
            # Store information about how features are transformed for the prediction endpoint
            # This will map original feature names to their names after potential processing (e.g., indexing)
            # and also store the dtypes the model expects for raw input.
            model_input_schema_info = [] 

            current_df_for_pipeline = self.df # Start with the original dataframe

            # 1. StringIndexer stages for categorical features
            processed_feature_cols_for_assembler = [] # Columns that will go into VectorAssembler
            for col_name in original_feature_cols:
                col_type = dict(current_df_for_pipeline.dtypes)[col_name]
                model_input_schema_info.append({'name': col_name, 'original_type': col_type, 'processed_name': col_name})
                
                if col_type == 'string':
                    indexed_col_name = col_name + "_indexed_reg_pipeline"
                    stages.append(StringIndexer(inputCol=col_name, outputCol=indexed_col_name, handleInvalid="keep")) # keep or error or skip
                    processed_feature_cols_for_assembler.append(indexed_col_name)
                    # Update processed_name for schema info
                    for item in model_input_schema_info:
                        if item['name'] == col_name:
                            item['processed_name'] = indexed_col_name
                            item['is_indexed'] = True # Mark that it was indexed
                            break
                else: # Numeric column (or other type VectorAssembler can handle)
                    processed_feature_cols_for_assembler.append(col_name)
                    # Mark that it was not indexed (or handle type conversion if needed)
                    for item in model_input_schema_info:
                        if item['name'] == col_name:
                            item['is_indexed'] = False
                            # We might need to ensure it's double for assembler if it's int
                            # current_df_for_pipeline = current_df_for_pipeline.withColumn(col_name, current_df_for_pipeline[col_name].cast(DoubleType()))
                            break


            if not processed_feature_cols_for_assembler:
                 raise ValueError("No feature columns available for VectorAssembler after processing strings.")

            # 2. VectorAssembler stage
            assembler = VectorAssembler(inputCols=processed_feature_cols_for_assembler, outputCol="features", handleInvalid="skip") # Standardized to 'features'
            stages.append(assembler)

            # 3. Regression model stage
            # The label column for the pipeline needs to be numeric.
            # If target_feature is string, it must be indexed. If numeric, it should be cast to Double.
            pipeline_label_col = "label"  # Standardize label column as well
            
            target_col_original_type = dict(current_df_for_pipeline.dtypes)[target_feature]
            model_input_schema_info.append({'name': target_feature, 'original_type': target_col_original_type, 'is_target': True})


            if target_col_original_type == 'string':
                target_indexer = StringIndexer(inputCol=target_feature, outputCol=pipeline_label_col, handleInvalid="error") # Error on unseen labels for target
                stages.insert(0, target_indexer) # Target indexing should happen on the original data
            else: # Numeric target, ensure it's double and named as pipeline_label_col
                # This renaming/casting must happen on the DataFrame fed to pipeline.fit
                # To avoid modifying stages list after definition, we do it on the df.
                 current_df_for_pipeline = current_df_for_pipeline.withColumn(pipeline_label_col, current_df_for_pipeline[target_feature].cast(DoubleType()))


            lr = LinearRegression(featuresCol="features", labelCol=pipeline_label_col)
            stages.append(lr)

            pipeline = Pipeline(stages=stages)
            
            logger.info(f"Fitting regression pipeline with stages: {stages}")
            logger.info(f"Schema for pipeline fit: ")
            current_df_for_pipeline.printSchema() # Log schema used for fitting

            # Filter out nulls from the pipeline_label_col before fitting
            rows_before_label_filter = current_df_for_pipeline.count()
            current_df_for_pipeline = current_df_for_pipeline.na.drop(subset=[pipeline_label_col])
            rows_after_label_filter = current_df_for_pipeline.count()
            logger.info(f"Rows for pipeline fit: before label null filter={rows_before_label_filter}, after={rows_after_label_filter}")

            if rows_after_label_filter == 0:
                raise ValueError("Target column contains all nulls or becomes all nulls after processing. Cannot train regression model.")

            pipeline_model = pipeline.fit(current_df_for_pipeline)
            logger.info("Regression pipeline fitted successfully.")

            # --- Save the PipelineModel ---
            # Ensure MODELS_FOLDER is defined in Config
            if not hasattr(Config, 'MODELS_FOLDER'):
                # Define a default if not in Config
                Config.MODELS_FOLDER = os.path.join(Config.RESULTS_FOLDER if hasattr(Config, 'RESULTS_FOLDER') else 'results', "models")
            
            if not os.path.exists(Config.MODELS_FOLDER):
                os.makedirs(Config.MODELS_FOLDER)
            
            model_id = f"regression_pipeline_{str(uuid.uuid4())[:12]}" # Longer ID
            model_save_path = os.path.join(Config.MODELS_FOLDER, model_id)
            
            # Ensure the path is a proper URI for Spark on local FS, especially for Windows
            if os.name == 'nt': # For Windows
                # Convert C:\\path\\to\\model to file:///C:/path/to/model
                # os.path.abspath will resolve, then replace backslashes
                abs_model_save_path = os.path.abspath(model_save_path)
                uri_model_save_path = "file:///" + abs_model_save_path.replace("\\\\", "/")
            else: # For Unix-like systems
                uri_model_save_path = "file://" + os.path.abspath(model_save_path)

            logger.info(f"Attempting to save Regression PipelineModel to: {uri_model_save_path}")
            pipeline_model.save(uri_model_save_path)
            logger.info(f"Regression PipelineModel saved successfully to: {uri_model_save_path}")

            # --- Metrics Calculation ---
            # For metrics, we need to transform the original data (or a split) using the pipeline model
            # This ensures metrics are based on the same preprocessing the saved model uses.
            
            # Create a DataFrame for evaluation that includes the target column processed as pipeline_label_col
            eval_df = self.df # Start with original
            if target_col_original_type == 'string':
                 eval_df = target_indexer.fit(eval_df).transform(eval_df) # Apply fitted target indexer
            else:
                 eval_df = eval_df.withColumn("label", eval_df[target_feature].cast(DoubleType()))
            eval_df = eval_df.na.drop(subset=["label"]) # Drop null labels

            # Split this evaluation DataFrame
            train_data_for_eval, test_data_for_eval = eval_df.randomSplit([0.8, 0.2], seed=42)

            if test_data_for_eval.count() == 0:
                logger.warning("Test data for evaluation is empty. Metrics will be based on training data or N/A.")
                # Fallback to training data for prediction or handle appropriately
                predictions_for_metrics = pipeline_model.transform(train_data_for_eval) if train_data_for_eval.count() > 0 else None
            else:
                predictions_for_metrics = pipeline_model.transform(test_data_for_eval)
            
            metrics_calc_results = {}
            if predictions_for_metrics and predictions_for_metrics.count() > 0:
                from pyspark.ml.evaluation import RegressionEvaluator
                r2_eval = RegressionEvaluator(labelCol="label", predictionCol="prediction", metricName="r2")
                metrics_calc_results['r2'] = r2_eval.evaluate(predictions_for_metrics)
                
                rmse_eval = RegressionEvaluator(labelCol="label", predictionCol="prediction", metricName="rmse")
                metrics_calc_results['rmse'] = rmse_eval.evaluate(predictions_for_metrics)
                
                mae_eval = RegressionEvaluator(labelCol="label", predictionCol="prediction", metricName="mae")
                metrics_calc_results['mae'] = mae_eval.evaluate(predictions_for_metrics)
                logger.info(f"Metrics calculated: R2={metrics_calc_results.get('r2')}, RMSE={metrics_calc_results.get('rmse')}, MAE={metrics_calc_results.get('mae')}")
            else:
                logger.warning("No predictions generated for metrics calculation.")
                metrics_calc_results['r2'] = 'N/A'
                metrics_calc_results['rmse'] = 'N/A'
                metrics_calc_results['mae'] = 'N/A'

            # Extracting coefficients and intercept from the LinearRegressionModel within the PipelineModel
            lr_model_in_pipeline = pipeline_model.stages[-1] # Assumes LR is the last stage
            coefficients_dict = {}
            intercept_val = 0.0
            if isinstance(lr_model_in_pipeline, LinearRegressionModel): # Check type
                intercept_val = lr_model_in_pipeline.intercept
                # The coefficients correspond to 'processed_feature_cols_for_assembler'
                # Map them back to original feature names
                for i, name_in_assembler in enumerate(processed_feature_cols_for_assembler):
                    # Find original name
                    original_name = name_in_assembler
                    if name_in_assembler.endswith("_indexed_reg_pipeline"):
                        original_name = name_in_assembler.replace("_indexed_reg_pipeline", "")
                    coefficients_dict[original_name] = lr_model_in_pipeline.coefficients[i]
                logger.info(f"Model intercept: {intercept_val}, Coefficients: {coefficients_dict}")

            final_metrics = {
                'r2': metrics_calc_results.get('r2'),
                'rmse': metrics_calc_results.get('rmse'),
                'mae': metrics_calc_results.get('mae'),
                'coefficients': coefficients_dict,
                'intercept': intercept_val,
                'model_id': model_id, # For API to load the model
                'model_save_path': uri_model_save_path, # Return the URI path
                'model_input_schema': model_input_schema_info, # Crucial for frontend form and prediction endpoint
                'target_column_original': target_feature,
                'pipeline_label_column': "label",
                'message': 'Regression model trained and saved successfully.'
            }
            
            # --- BEGIN DATA INSPECTION LOGS ---
            logger.info(f"[DEBUG] Selected features: {original_feature_cols}")
            logger.info(f"[DEBUG] Target feature: {target_feature}")
            logger.info(f"[DEBUG] DataFrame row count: {current_df_for_pipeline.count()}, columns: {len(current_df_for_pipeline.columns)}")
            if target_feature:
                unique_targets = current_df_for_pipeline.select(target_feature).distinct().count()
                logger.info(f"[DEBUG] Unique values in target: {unique_targets}")
            for col in original_feature_cols + ([target_feature] if target_feature else []):
                try:
                    missing = current_df_for_pipeline.filter(current_df_for_pipeline[col].isNull()).count()
                    logger.info(f"[DEBUG] Missing values in {col}: {missing}")
                except Exception as e:
                    logger.info(f"[DEBUG] Could not compute missing values for {col}: {e}")
            # Class distribution (for regression, just value counts)
            if target_feature:
                try:
                    class_dist = current_df_for_pipeline.groupBy(target_feature).count().toPandas()
                    logger.info(f"[DEBUG] Value distribution (target):\n{class_dist}")
                except Exception as e:
                    logger.info(f"[DEBUG] Could not compute value distribution: {e}")
            # Feature variance (for numeric features)
            for col in original_feature_cols:
                try:
                    var = current_df_for_pipeline.agg({col: "variance"}).collect()[0][0]
                    logger.info(f"[DEBUG] Variance of {col}: {var}")
                except Exception as e:
                    logger.info(f"[DEBUG] Could not compute variance for {col}: {e}")
            # --- END DATA INSPECTION LOGS ---

            return self._convert_to_serializable(final_metrics)

        except Exception as e:
            logger.error(f"Error in regression analysis: {str(e)}")
            logger.error(traceback.format_exc()) # Ensure traceback is logged
            # Return a more structured error
            return self._convert_to_serializable({
                'error': f"Error in regression analysis: {str(e)}",
                'trace': traceback.format_exc()
            })
    
    def _classification_analysis(self, selected_features, target_feature):
        """Perform classification analysis"""
        # Prepare features
        feature_cols = [f for f in selected_features if f != target_feature]
        
        logger.info(f"Original selected features for classification: {selected_features}")
        logger.info(f"Feature columns for VectorAssembler: {feature_cols}")
        logger.info(f"Target column: {target_feature}")
        logger.info(f"Schema of self.df before preparation:")
        self.df.printSchema()
        logger.info(f"Sample of self.df before preparation (showing target_feature '{target_feature}'):")
        self.df.select(feature_cols + [target_feature]).show(5, truncate=False)

        ml_ready_df, actual_assembler_inputs = self._prepare_ml_dataframe(
            self.df,
            feature_col_names=feature_cols,
            target_col_name=target_feature,
            features_vec_col="features",
            label_col="label"
        )
        
        logger.info("Schema of ml_ready_df (after _prepare_ml_dataframe):")
        ml_ready_df.printSchema()
        logger.info("Sample of ml_ready_df (showing features and label):")
        ml_ready_df.select("features", "label").show(5, truncate=False)
        
        # --- BEGIN DATA INSPECTION LOGS ---
        logger.info(f"[DEBUG] Selected features: {feature_cols}")
        logger.info(f"[DEBUG] Target feature: {target_feature}")
        logger.info(f"[DEBUG] DataFrame row count: {ml_ready_df.count()}, columns: {len(ml_ready_df.columns)}")
        if target_feature:
            unique_targets = ml_ready_df.select('label').distinct().count()
            logger.info(f"[DEBUG] Unique values in target (label): {unique_targets}")
        for col in actual_assembler_inputs + (["label"] if target_feature else []):
            try:
                missing = ml_ready_df.filter(ml_ready_df[col].isNull()).count()
                logger.info(f"[DEBUG] Missing values in {col}: {missing}")
            except Exception as e:
                logger.info(f"[DEBUG] Could not compute missing values for {col}: {e}")
        # Class distribution
        if target_feature:
            try:
                class_dist = ml_ready_df.groupBy('label').count().toPandas()
                logger.info(f"[DEBUG] Class distribution (label):\n{class_dist}")
            except Exception as e:
                logger.info(f"[DEBUG] Could not compute class distribution: {e}")
        # Feature variance (for numeric features)
        for col in actual_assembler_inputs:
            try:
                var = ml_ready_df.agg({col: "variance"}).collect()[0][0]
                logger.info(f"[DEBUG] Variance of {col}: {var}")
            except Exception as e:
                logger.info(f"[DEBUG] Could not compute variance for {col}: {e}")
        # --- END DATA INSPECTION LOGS ---

        # Split data
        train_data, test_data = ml_ready_df.randomSplit([0.8, 0.2], seed=42)
        
        # --- Check test_data size and diversity ---
        min_test_samples = 10 # Define a minimum number of samples for a reliable test set
        min_distinct_labels_in_test = 2 # Define minimum number of distinct labels needed

        test_data_count = test_data.count()
        if test_data_count == 0:
            logger.warning("Test data is empty after split. Cannot evaluate model.")
            # Return a message or default metrics indicating no evaluation was possible
            return self._convert_to_serializable({
                'accuracy': 'N/A - Test set was empty',
                'correct_predictions': 0,
                'total_test_samples': 0,
                'message': 'Test set was empty after splitting. Not enough data for evaluation.'
            })
        
        distinct_labels_count_in_test = test_data.select("label").distinct().count()
        
        if test_data_count < min_test_samples or distinct_labels_count_in_test < min_distinct_labels_in_test:
            logger.warning(f"Test set is too small (count: {test_data_count}) or lacks label diversity (distinct labels: {distinct_labels_count_in_test}). Accuracy metric might be unreliable.")
            # Optionally, still proceed but the frontend should display this warning prominently with the accuracy.
            # For now, we will proceed but this log is important.

        # Train model - ensure model uses "features" and "label"
        lr = LogisticRegression(featuresCol="features", labelCol="label")
        model = lr.fit(train_data)
        
        # Make predictions
        predictions = model.transform(test_data)
        
        logger.info("Sample of predictions DataFrame:")
        try:
            # Construct list of columns to select for predictions log, ensuring target_feature is present if it exists
            # It might have been renamed to 'label' or indexed into 'label'.
            # 'label' is the definitive ground truth column after _prepare_ml_dataframe.
            # We also want to see the original target_feature if it's still in test_data (it might not be if it was string and got indexed without keeping original).
            # And the actual_assembler_inputs to see what went into 'features'.
            
            prediction_log_cols = ["label", "rawPrediction", "probability", "prediction"] 
            # Add original feature columns that went into the assembler if they exist in test_data
            for col_name in actual_assembler_inputs:
                if col_name in test_data.columns and col_name not in prediction_log_cols:
                    prediction_log_cols.append(col_name)
            # Add the original target_feature if it exists in test_data and is not already 'label'
            if target_feature in test_data.columns and target_feature != "label" and target_feature not in prediction_log_cols:
                 prediction_log_cols.append(target_feature)

            predictions.select(prediction_log_cols).show(20, truncate=False)
        except Exception as e_log_pred:
            logger.error(f"Error logging predictions sample: {str(e_log_pred)}")
            logger.info("Minimal predictions log due to error:")
            predictions.select("label", "prediction").show(20, truncate=False)

        logger.info(f"Value counts for original target '{target_feature}' in test_data (if column still exists):")
        if target_feature in test_data.columns:
            test_data.groupBy(target_feature).count().show()
        else:
            logger.info(f"Original target column '{target_feature}' not found in test_data (likely indexed to 'label').")

        logger.info("Value counts for 'label' in test_data:")
        test_data.groupBy("label").count().show()
        logger.info("Value counts for 'prediction' column in predictions DF:")
        predictions.groupBy("prediction").count().show()
        
        # Calculate metrics
        accuracy = 0.0
        correct_predictions = 0
        # total_predictions_in_test_set should be test_data_count as this is the data fed to model.transform
        # after null label filtering.
        total_test_samples_evaluated = test_data_count 

        if predictions.count() > 0: # Ensure predictions DF is not empty (model actually outputted something)
            correct_predictions = predictions.filter(predictions["prediction"] == predictions["label"]).count()
            # Base accuracy on the number of items the model made a prediction for.
            # Normally predictions.count() should equal test_data_count.
            accuracy = float(correct_predictions / predictions.count()) if predictions.count() > 0 else 0.0 
        else:
            logger.warning("Predictions DataFrame is empty, though test_data was not. Accuracy set to 0.")
        
        logger.info(f"Calculated accuracy: {accuracy}")
        logger.info(f"Correct predictions: {correct_predictions}, Total test samples evaluated: {total_test_samples_evaluated}")

        metrics = {
            'accuracy': accuracy,
            'correct_predictions': correct_predictions, 
            'total_test_samples': total_test_samples_evaluated, 
            'message': 'Analysis successful' # Default message
        }
        
        # Add a sample of predictions to the metrics for frontend display
        try:
            # Select relevant columns for the prediction sample table
            prediction_log_cols = ["label", "prediction", "probability"]
            if "rawPrediction" in predictions.columns: # rawPrediction can be very wide
                # Taking only first few elements if it's a vector, or skip if too complex
                pass # For now, let's keep it simple and rely on probability and prediction

            # Add original feature columns that went into the assembler if they exist
            # This helps in understanding the prediction against actual feature values
            for col_name in actual_assembler_inputs:
                if col_name in test_data.columns and col_name not in prediction_log_cols:
                    prediction_log_cols.append(col_name)
            
            # Add the original target_feature if it exists in test_data and is not already 'label'
            if target_feature in test_data.columns and target_feature != "label" and target_feature not in prediction_log_cols:
                 prediction_log_cols.append(target_feature)

            # Take a sample of predictions
            prediction_sample_df = predictions.select(prediction_log_cols).limit(20).toPandas()
            # Clean up column names in pandas_df if they were derived (e.g. feature_indexed)
            # For simplicity, we'll use them as is for now.
            metrics['prediction_sample'] = prediction_sample_df.to_dict(orient='records')
            metrics['prediction_sample_headers'] = prediction_sample_df.columns.tolist()

        except Exception as e_pred_sample:
            logger.error(f"Error preparing prediction sample for frontend: {str(e_pred_sample)}")
            metrics['prediction_sample'] = []
            metrics['prediction_sample_headers'] = []
            metrics['message'] = 'Analysis successful, but error preparing prediction sample for display.'


        # Update message if test set was small or lacked diversity
        if test_data_count < min_test_samples or distinct_labels_count_in_test < min_distinct_labels_in_test:
            metrics['message'] = f"Warning: Test set size ({test_data_count}) or label diversity ({distinct_labels_count_in_test}) is low. Metrics might be unreliable."
            if test_data_count == 0:
                 metrics['message'] = 'Test set was empty after splitting. Not enough data for evaluation.'
                 metrics['accuracy'] = 'N/A - Test set was empty' # Ensure accuracy reflects this state

        try:
            if hasattr(model, 'summary') and hasattr(model.summary, 'areaUnderROC'):
                 metrics['area_under_roc'] = float(model.summary.areaUnderROC)
        except Exception as roc_e:
            logger.warning(f"Could not retrieve areaUnderROC: {str(roc_e)}")

        return self._convert_to_serializable(metrics)
    
    def _clustering_analysis(self, selected_features, k=3):
        """Perform clustering analysis"""
        try:
            # Verify selected features exist in dataframe
            missing_features = [f for f in selected_features if f not in self.df.columns]
            if missing_features:
                raise ValueError(f"Features not found in dataset: {', '.join(missing_features)}")

            # Prepare data using the new helper
            # For clustering, selected_features are the features, no target column.
            ml_ready_df, actual_assembler_inputs = self._prepare_ml_dataframe(
                self.df, 
                feature_col_names=selected_features, 
                target_col_name=None,
                features_vec_col="features" 
            )
            
            kmeans = KMeans(k=k, featuresCol='features', predictionCol='prediction')
            model = kmeans.fit(ml_ready_df) 
            
            predictions = model.transform(ml_ready_df)
            
            cluster_sizes = predictions.groupBy('prediction').count().collect()
            spark_centers = model.clusterCenters() # List of np.array (vectors)

            # Map cluster centers back to original selected_feature names
            # The order of elements in spark_centers vectors corresponds to actual_assembler_inputs
            mapped_centers_list_of_lists = []
            for center_vector in spark_centers:
                current_center_ordered_by_selected_features = []
                for original_feature_name in selected_features:
                    # Determine the name this original feature would have in actual_assembler_inputs
                    name_in_assembler_inputs = ""
                    is_string_type = dict(self.df.dtypes).get(original_feature_name) == 'string'
                    if is_string_type:
                        name_in_assembler_inputs = original_feature_name + "_indexed"
                    else:
                        name_in_assembler_inputs = original_feature_name
                    
                    try:
                        # Find the index of this name in actual_assembler_inputs to get the correct value from center_vector
                        idx_in_vector = actual_assembler_inputs.index(name_in_assembler_inputs)
                        current_center_ordered_by_selected_features.append(center_vector[idx_in_vector])
                    except ValueError:
                        logger.warning(f"Could not map cluster center value for '{original_feature_name}' (expected as '{name_in_assembler_inputs}' in assembler inputs). Appending None.")
                        current_center_ordered_by_selected_features.append(None) # Or some placeholder
                mapped_centers_list_of_lists.append(current_center_ordered_by_selected_features)
            
            metrics = {
                'k': k,
                'cluster_sizes': {int(row['prediction']): int(row['count']) for row in cluster_sizes},
                'cluster_centers': mapped_centers_list_of_lists, 
                'cluster_center_feature_names': selected_features 
            }
            
            # --- BEGIN DATA INSPECTION LOGS ---
            logger.info(f"[DEBUG] Selected features: {selected_features}")
            logger.info(f"[DEBUG] DataFrame row count: {ml_ready_df.count()}, columns: {len(ml_ready_df.columns)}")
            if selected_features:
                unique_targets = ml_ready_df.select(selected_features[0]).distinct().count()
                logger.info(f"[DEBUG] Unique values in target: {unique_targets}")
            for col in actual_assembler_inputs + (selected_features if selected_features else []):
                try:
                    missing = ml_ready_df.filter(ml_ready_df[col].isNull()).count()
                    logger.info(f"[DEBUG] Missing values in {col}: {missing}")
                except Exception as e:
                    logger.info(f"[DEBUG] Could not compute missing values for {col}: {e}")
            # Class distribution (for clustering, just value counts)
            if selected_features:
                try:
                    class_dist = ml_ready_df.groupBy(selected_features[0]).count().toPandas()
                    logger.info(f"[DEBUG] Value distribution (target):\n{class_dist}")
                except Exception as e:
                    logger.info(f"[DEBUG] Could not compute value distribution: {e}")
            # Feature variance (for numeric features)
            for col in actual_assembler_inputs:
                try:
                    var = ml_ready_df.agg({col: "variance"}).collect()[0][0]
                    logger.info(f"[DEBUG] Variance of {col}: {var}")
                except Exception as e:
                    logger.info(f"[DEBUG] Could not compute variance for {col}: {e}")
            # --- END DATA INSPECTION LOGS ---

            return self._convert_to_serializable(metrics)
        except Exception as e:
            logger.error(f"Error in clustering analysis: {str(e)}")
            logger.error(traceback.format_exc()) # Add traceback for better debugging
            raise 