import os
os.environ["HADOOP_HOME"] = "C:/hadoop-3.3.6"

from flask import Flask, request, jsonify, render_template, send_from_directory
from flask_cors import CORS
from config import Config
from database import MongoDB
from spark_manager import SparkManager
from data_analyzer import DataAnalyzer
from models.model_training import ModelTrainer
from visualization import Visualizer
import uuid
import json
import traceback
import logging
import datetime
import pandas as pd
import numpy as np
from pyspark.ml import PipelineModel
from pyspark.sql.types import StructType, StructField, StringType, DoubleType, IntegerType, BooleanType
from model_selector import suggest_tasks
from logging_config import configure_logging
from feature_selector import filter_correlation, filter_variance, select_k_best, recursive_feature_elimination
from io import StringIO
import webbrowser
from threading import Timer
from pyspark.ml.feature import VectorAssembler

# Configure logging
configure_logging()

logger = logging.getLogger(__name__)

# Create a StringIO handler to capture log messages
log_capture = StringIO()
log_handler = logging.StreamHandler(log_capture)
log_handler.setLevel(logging.INFO)
logger.addHandler(log_handler)

class JSONEncoder(json.JSONEncoder):
    def default(self, obj):
        try:
            if hasattr(obj, 'toPandas'):
                return json.loads(obj.toPandas().to_json(orient='records'))
            if isinstance(obj, pd.DataFrame):
                return obj.to_dict(orient='records')
            if isinstance(obj, pd.Series):
                return obj.tolist()
            if isinstance(obj, np.ndarray):
                return obj.tolist()
            if isinstance(obj, (np.generic, np.int64, np.int32, np.float64, np.float32)):
                return obj.item()
            if isinstance(obj, dict):
                return {k: self.default(v) for k, v in obj.items()}
            if isinstance(obj, list):
                return [self.default(v) for v in obj]
            return super().default(obj)
        except Exception as e:
            logger.error(f"JSON serialization error: {str(e)}")
            return str(obj)

app = Flask(__name__)
# Configure CORS to allow all origins for development
CORS(app, resources={
    r"/*": {
        "origins": "*",
        "methods": ["GET", "POST", "OPTIONS"],
        "allow_headers": ["Content-Type"]
    }
})
app.config.from_object(Config)

# Define allowed file extensions
app.config['ALLOWED_EXTENSIONS'] = {'csv', 'json', 'xlsx'}
app.json_encoder = JSONEncoder

# Explicitly set maximum content length
app.config['MAX_CONTENT_LENGTH'] = Config.MAX_CONTENT_LENGTH
# Enable chunked encoding
app.config['MAX_CONTENT_LENGTH'] = None

# Initialize folders
try:
    Config.init_folders()
    logger.info("Folders initialized successfully")
except Exception as e:
    logger.error(f"Error initializing folders: {str(e)}")
    raise

# Initialize MongoDB connection
try:
    mongo = MongoDB()
    # Test MongoDB connection
    mongo.test_connection()
    logger.info("MongoDB connection established successfully")
except Exception as e:
    logger.error(f"Error connecting to MongoDB: {str(e)}")
    raise

# Initialize Spark
try:
    spark_manager = SparkManager()
    logger.info("Spark initialized successfully")
except Exception as e:
    logger.error(f"Error initializing Spark: {str(e)}")
    raise

# Initialize visualizer
visualizer = Visualizer(os.path.join(app.root_path, 'static'))

@app.route('/')
def index():
    return render_template('index.html')

@app.route('/upload', methods=['POST'])
def upload_file():
    try:
        logger.info("Starting file upload process")
        if 'file' not in request.files:
            logger.warning("No file part in request")
            return jsonify({'error': 'No file part'}), 400
        
        file = request.files['file']
        if file.filename == '':
            logger.warning("No selected file")
            return jsonify({'error': 'No selected file'}), 400
        
        file_extension = file.filename.split('.')[-1].lower()
        if file_extension not in app.config['ALLOWED_EXTENSIONS']:
            logger.warning(f"File type not allowed: {file.filename}")
            return jsonify({'error': 'File type not allowed'}), 400
        
        # Read file content
        file_content = file.read()
        
        # Initialize data analyzer with content
        logger.info("Initializing data analyzer")
        analyzer = DataAnalyzer(spark_manager, file_content, file_extension)
        
        # Get initial analysis
        logger.info("Performing initial analysis")
        analysis_results = analyzer.initial_analysis()
        logger.info("Initial analysis completed")
        
        # Get columns from the DataFrame
        columns = analyzer.df.columns
        logger.info(f"Columns found: {columns}")
        
        # Generate unique collection name for this dataset
        collection_name = f"dataset_{str(uuid.uuid4()).replace('-', '_')}"
        
        # Store data in MongoDB
        logger.info("Storing data in MongoDB")
        spark_manager.write_to_mongodb(analyzer.df, collection_name)
        
        # Store metadata in MongoDB
        logger.info("Storing metadata in MongoDB")
        metadata = {
            'original_filename': file.filename,
            'collection_name': collection_name,
            'status': 'uploaded',
            'analysis': analysis_results,
            'columns': columns,  # Store columns in metadata
            'created_at': datetime.datetime.utcnow()
        }
        file_id = mongo.insert_file_metadata(metadata)
        logger.info(f"Metadata stored with ID: {file_id}")

        # --- AUTO-TRAIN MODELS ON UPLOAD ---
        try:
            logger.info("Auto-training models on upload...")
            df = analyzer.df
            all_columns = df.columns
            if len(all_columns) < 2:
                logger.warning("Not enough columns to train models.")
            else:
                # Use last column as default target for regression/classification
                default_target = all_columns[-1]
                feature_cols = [col for col in all_columns if col != default_target]
                from pyspark.ml import Pipeline
                from pyspark.ml.feature import VectorAssembler, StringIndexer
                from pyspark.ml.regression import LinearRegression
                from pyspark.ml.classification import LogisticRegression
                from pyspark.ml.clustering import KMeans
                import os

                # Prepare folders
                MODELS_FOLDER = os.path.join(app.root_path, 'auto_models')
                if not os.path.exists(MODELS_FOLDER):
                    os.makedirs(MODELS_FOLDER)

                # --- REGRESSION PIPELINE ---
                reg_assembler = VectorAssembler(inputCols=feature_cols, outputCol="features")
                reg_lr = LinearRegression(featuresCol="features", labelCol="label")
                reg_pipeline = Pipeline(stages=[reg_assembler, reg_lr])
                reg_df = df.withColumnRenamed(default_target, "label")
                reg_model = reg_pipeline.fit(reg_df)
                reg_model_path = os.path.join(MODELS_FOLDER, f"regression_{file_id}")
                reg_model.save(reg_model_path)
                metadata['regression_model_path'] = reg_model_path

                # --- CLASSIFICATION PIPELINE (if target is categorical with <20 unique values) ---
                n_unique = df.select(default_target).distinct().count()
                if n_unique > 1 and n_unique <= 20 and str(df.schema[default_target].dataType) == 'StringType':
                    class_indexer = StringIndexer(inputCol=default_target, outputCol="label")
                    class_assembler = VectorAssembler(inputCols=feature_cols, outputCol="features")
                    class_lr = LogisticRegression(featuresCol="features", labelCol="label")
                    class_pipeline = Pipeline(stages=[class_indexer, class_assembler, class_lr])
                    class_model = class_pipeline.fit(df)
                    class_model_path = os.path.join(MODELS_FOLDER, f"classification_{file_id}")
                    class_model.save(class_model_path)
                    metadata['classification_model_path'] = class_model_path
                else:
                    logger.info("Classification model not trained: target not categorical or too many unique values.")

                # --- CLUSTERING PIPELINE ---
                cluster_assembler = VectorAssembler(inputCols=all_columns, outputCol="features")
                kmeans = KMeans(featuresCol="features", k=3, seed=42)
                cluster_pipeline = Pipeline(stages=[cluster_assembler, kmeans])
                cluster_model = cluster_pipeline.fit(df)
                cluster_model_path = os.path.join(MODELS_FOLDER, f"clustering_{file_id}")
                cluster_model.save(cluster_model_path)
                metadata['clustering_model_path'] = cluster_model_path

                # Update metadata with model paths
                metadata['model_training_progress'] = 0
                metadata['model_training_message'] = 'Model is being trained. Please wait...'
                mongo.update_file_metadata(file_id, metadata)

                # At the end of training (after model.save()):
                metadata['model_training_progress'] = 100
                metadata['model_training_message'] = 'Model training complete.'
                mongo.update_file_metadata(file_id, metadata)

                logger.info("Auto-trained models saved and metadata updated.")
        except Exception as e:
            logger.error(f"Error auto-training models on upload: {str(e)}")
            logger.error(traceback.format_exc())

        response_data = {
            'message': 'File processed successfully',
            'file_id': str(file_id),
            'collection_name': collection_name,
            'columns': columns,  # Include columns in response
            'analysis': analysis_results
        }
        
        logger.info("Upload process completed successfully")
        return jsonify(response_data)
    except Exception as e:
        logger.error(f"Error in upload: {str(e)}")
        logger.error(traceback.format_exc())
        return jsonify({'error': f'Error processing file: {str(e)}'}), 500

@app.route('/analyze/<file_id>', methods=['POST'])
def analyze_data(file_id):
    try:
        logger.info(f"Starting analysis for file_id: {file_id}")
        analysis_type = request.json.get('analysis_type')
        selected_features = request.json.get('selected_features', [])
        target_feature = request.json.get('target_feature', None)
        
        logger.info(f"Analysis parameters - type: {analysis_type}, features: {selected_features}, target: {target_feature}")
        
        # Get file metadata from MongoDB
        metadata = mongo.get_file_metadata(file_id)
        if not metadata:
            logger.warning(f"File not found: {file_id}")
            return jsonify({'error': 'File not found'}), 404
        
        # Get data from MongoDB collection
        collection_name = metadata.get('collection_name')
        if not collection_name:
            return jsonify({'error': 'Collection name not found in metadata'}), 400
            
        # Initialize analyzer with data from MongoDB
        logger.info(f"Reading data from collection: {collection_name}")
        df = spark_manager.read_from_mongodb(collection_name)
        analyzer = DataAnalyzer(spark_manager)
        analyzer.set_dataframe(df)
        
        # Perform analysis
        logger.info("Starting analysis")
        # Check if the selected features/target match any existing model
        model_key = f"model_{'_'.join(selected_features)}_{target_feature}"
        metadata = mongo.get_file_metadata(file_id)  # Refresh metadata
        if 'trained_models' not in metadata:
            metadata['trained_models'] = {}
        trained_models = metadata['trained_models']
        model_path = trained_models.get(model_key)
        if not model_path:
            # Train a new model with the selected features/target
            logger.info("No pre-trained model for selected features/target. Training new model...")
            from pyspark.ml import Pipeline
            from pyspark.ml.feature import VectorAssembler, StringIndexer
            from pyspark.ml.regression import LinearRegression
            from pyspark.ml.classification import LogisticRegression
            import os
            MODELS_FOLDER = os.path.join(app.root_path, 'auto_models')
            if not os.path.exists(MODELS_FOLDER):
                os.makedirs(MODELS_FOLDER)
            # Determine task type
            if analysis_type == 'regression':
                assembler = VectorAssembler(inputCols=selected_features, outputCol="features")
                lr = LinearRegression(featuresCol="features", labelCol="label")
                pipeline = Pipeline(stages=[assembler, lr])
                df_train = df.withColumnRenamed(target_feature, "label")
                model = pipeline.fit(df_train)
            elif analysis_type == 'classification':
                indexer = StringIndexer(inputCol=target_feature, outputCol="label")
                assembler = VectorAssembler(inputCols=selected_features, outputCol="features")
                lr = LogisticRegression(featuresCol="features", labelCol="label")
                pipeline = Pipeline(stages=[indexer, assembler, lr])
                model = pipeline.fit(df)
            else:
                model = None  # For clustering, handled elsewhere
            if model:
                model_path = os.path.join(MODELS_FOLDER, f"{analysis_type}_{file_id}_{'_'.join(selected_features)}_{target_feature}")
                model.save(model_path)
                trained_models[model_key] = model_path
                metadata['trained_models'] = trained_models
                mongo.update_file_metadata(file_id, metadata)
                logger.info(f"Trained and saved new {analysis_type} model for selected features/target.")
                # Inform the user (frontend should display this message)
                return jsonify({'message': 'Model was not pre-trained for these columns. Training now. Please retry your analysis in a moment.'}), 202
        # Always load and use the PipelineModel for analysis
        from pyspark.ml import PipelineModel
        if not model_path:
            return jsonify({'message': 'Model is still being trained. Please wait and try again.'}), 202
        pipeline_model = PipelineModel.load(model_path)
        predictions = pipeline_model.transform(df)
        # You can now use 'predictions' for further analysis or metrics
        # (If analyzer.analyze uses predictions, pass them as needed)
        results = analyzer.analyze(
            analysis_type=analysis_type,
            selected_features=selected_features,
            target_feature=target_feature
        )
        logger.info("Analysis completed")
        # Update metadata with results
        logger.info("Updating metadata with results")
        mongo.update_file_metadata(file_id, {
            'status': 'analyzed',
            'analysis_type': analysis_type,
            'results': results
        })
        logger.info("Analysis process completed successfully")
        return jsonify(results)
    except Exception as e:
        logger.error(f"Error in analysis: {str(e)}")
        logger.error(traceback.format_exc())
        return jsonify({'error': f'Analysis error: {str(e)}'}), 500

@app.route('/favicon.ico')
def favicon():
    return send_from_directory(os.path.join(app.root_path, 'static'),
                             'favicon.ico', mimetype='image/vnd.microsoft.icon')

@app.route('/tasks/<file_id>', methods=['GET', 'POST'])
def handle_tasks(file_id):
    try:
        if request.method == 'GET':
            # Get file metadata and suggest tasks
            metadata = mongo.get_file_metadata(file_id)
            if not metadata:
                return jsonify({'error': 'File not found'}), 404
            
            # Get data sample for analysis
            collection_name = metadata.get('collection_name')
            df = spark_manager.read_from_mongodb(collection_name).limit(1000).toPandas()
            
            # Use the new model_selector module
            suggested_tasks = suggest_tasks(df)
            
            return jsonify({
                'available_features': df.columns.tolist(),
                'numeric_features': df.select_dtypes(include=[np.number]).columns.tolist(),
                'categorical_features': df.select_dtypes(include=['object']).columns.tolist(),
                'suggested_tasks': suggested_tasks,
                'data_shape': {'rows': len(df), 'columns': len(df.columns)}
            })
        
        elif request.method == 'POST':
            # Get task parameters
            task_params = request.json
            task_type = task_params.get('task_type')
            feature_cols = task_params.get('features', [])
            target_col = task_params.get('target')
            algorithm = task_params.get('algorithm', 'random_forest')
            
            # Get file metadata
            metadata = mongo.get_file_metadata(file_id)
            if not metadata:
                return jsonify({'error': 'File not found'}), 404
            
            # Get data from MongoDB collection
            collection_name = metadata.get('collection_name')
            if not collection_name:
                return jsonify({'error': 'Collection name not found in metadata'}), 400
            
            logger.info(f"Reading data from collection: {collection_name} for task: {task_type}")
            df = spark_manager.read_from_mongodb(collection_name)
            logger.info(f"DataFrame columns from MongoDB: {df.columns}")
            logger.info("DataFrame schema from MongoDB:")
            df.printSchema()
            
            # Convert to pandas for feature selection
            df_pandas = df.select(feature_cols + [target_col]).toPandas()
            
            # Feature selection
            try:
                # Convert to numpy array and handle non-numeric columns
                X = df_pandas[feature_cols].values
                y = df_pandas[target_col].values
                
                # Apply feature selection
                selected_indices = filter_correlation(X)
                if not selected_indices:
                    print("Warning: No features selected after correlation filtering. Using all features.")
                    selected_indices = list(range(len(feature_cols)))
                
                selected_features = [feature_cols[i] for i in selected_indices]
                print(f"Selected features after correlation filtering: {selected_features}")
                
                # Update feature columns
                feature_cols = selected_features
                
            except Exception as e:
                print(f"Error during feature selection: {e}")
                print("Continuing with all features...")
            
            # --- BEGIN DATA INSPECTION LOGS ---
            logger.info(f"[DEBUG] Selected features: {feature_cols}")
            logger.info(f"[DEBUG] Target feature: {target_col}")
            logger.info(f"[DEBUG] DataFrame row count: {df.count()}, columns: {len(df.columns)}")
            if target_col:
                unique_targets = df.select(target_col).distinct().count()
                logger.info(f"[DEBUG] Unique values in target: {unique_targets}")
            for col in feature_cols + ([target_col] if target_col else []):
                try:
                    missing = df.filter(df[col].isNull()).count()
                    logger.info(f"[DEBUG] Missing values in {col}: {missing}")
                except Exception as e:
                    logger.info(f"[DEBUG] Could not compute missing values for {col}: {e}")
            # Class distribution
            if target_col:
                try:
                    class_dist = df.groupBy(target_col).count().toPandas()
                    logger.info(f"[DEBUG] Class distribution:\n{class_dist}")
                except Exception as e:
                    logger.info(f"[DEBUG] Could not compute class distribution: {e}")
            # Feature variance (for numeric features)
            for col in feature_cols:
                try:
                    var = df.agg({col: "variance"}).collect()[0][0]
                    logger.info(f"[DEBUG] Variance of {col}: {var}")
                except Exception as e:
                    logger.info(f"[DEBUG] Could not compute variance for {col}: {e}")
            # --- END DATA INSPECTION LOGS ---
            
            # Initialize model trainer
            model_trainer = ModelTrainer(spark_manager.get_spark_session())
            
            # Perform task
            results = {}
            if task_type == 'regression':
                model_results = model_trainer.train_regressor(df, feature_cols, target_col, algorithm)
                results['model_performance'] = {'rmse': model_results['rmse']}
                if model_results['feature_importance']:
                    results['feature_importance_plot'] = visualizer.create_feature_importance_plot(
                        model_results['feature_importance']
                    )
                
                # Create regression results plot
                predictions = model_results['model'].transform(df)
                results['prediction_plot'] = visualizer.create_regression_results_plot(
                    predictions.select('label').toPandas(),
                    predictions.select('prediction').toPandas()
                )
            
            elif task_type == 'classification':
                model_results = model_trainer.train_classifier(df, feature_cols, target_col, algorithm)
                results['model_performance'] = {'accuracy': model_results['accuracy']}
                if model_results['feature_importance']:
                    results['feature_importance_plot'] = visualizer.create_feature_importance_plot(
                        model_results['feature_importance']
                    )
            
            elif task_type == 'clustering':
                n_clusters = task_params.get('n_clusters', 3)
                logger.info(f"Feature_cols for clustering from request: {feature_cols}")
                model_results = model_trainer.train_clustering(df, feature_cols, n_clusters)
                results['cluster_sizes'] = model_results['cluster_sizes']
                
                # Create clustering visualization if 2-3 features
                if 2 <= len(feature_cols) <= 3:
                    predictions = model_results['model'].transform(df)
                    results['clustering_plot'] = visualizer.create_clustering_plot(
                        predictions.toPandas(),
                        feature_cols,
                        predictions.select('cluster').toPandas(),
                        model_results['cluster_centers']
                    )
            
            elif task_type == 'correlation_analysis':
                df_pandas = df.select(feature_cols).toPandas()
                results['correlation_plot'] = visualizer.create_correlation_matrix(
                    df_pandas, feature_cols
                )
            
            # Store results in MongoDB
            mongo.update_file_metadata(file_id, {
                'tasks': {
                    task_type: {
                        'timestamp': datetime.datetime.utcnow(),
                        'parameters': task_params,
                        'results': results
                    }
                }
            })
            
            return jsonify(results)
            
    except Exception as e:
        logger.error(f"Error in task handling: {str(e)}")
        logger.error(traceback.format_exc())
        return jsonify({'error': str(e)}), 500

@app.route('/results/<file_id>', methods=['GET'])
def get_results(file_id):
    try:
        # Get file metadata with results
        metadata = mongo.get_file_metadata(file_id)
        if not metadata:
            return jsonify({'error': 'File not found'}), 404
        
        # Return all tasks and their results
        return jsonify({
            'file_info': {
                'original_filename': metadata.get('original_filename'),
                'upload_date': metadata.get('created_at'),
                'status': metadata.get('status')
            },
            'tasks': metadata.get('tasks', {})
        })
        
    except Exception as e:
        logger.error(f"Error getting results: {str(e)}")
        logger.error(traceback.format_exc())
        return jsonify({'error': str(e)}), 500

@app.route('/predict', methods=['POST'])
def predict_from_model():
    try:
        request_data = request.get_json()
        model_uri_path = request_data.get('model_uri_path')
        user_input_features = request_data.get('features')
        model_input_schema_desc = request_data.get('model_input_schema')

        if not all([model_uri_path, user_input_features, model_input_schema_desc]):
            return jsonify({'error': 'Missing model_uri_path, features, or model_input_schema in request'}), 400

        logger.info(f"Received prediction request for model: {model_uri_path}")
        logger.info(f"Input features: {user_input_features}")

        # Load the PipelineModel
        try:
            loaded_model = PipelineModel.load(model_uri_path)
            logger.info(f"Successfully loaded model from {model_uri_path}")
        except Exception as e:
            logger.error(f"Error loading model from {model_uri_path}: {str(e)}")
            logger.error(traceback.format_exc())
            return jsonify({'error': f"Error loading model: {str(e)}"}), 500

        # Prepare Spark DataFrame from user input features based on model_input_schema_desc
        data_row_values = []
        struct_fields = []
        
        # Filter schema for input features only (not target)
        schema_for_input_df = [info for info in model_input_schema_desc if not info.get('is_target', False)]

        for col_info in schema_for_input_df:
            col_name = col_info.get('name')
            original_type_str = col_info.get('original_type')
            
            if not col_name or not original_type_str:
                logger.error(f"Invalid column info in model_input_schema: {col_info}")
                return jsonify({'error': 'Invalid model_input_schema provided'}), 400

            raw_value = user_input_features.get(col_name)
            if raw_value is None:
                # This check assumes all features in the schema are required.
                # For optional features, the model should be trained to handle nulls/missing values,
                # and this logic might need adjustment (e.g., pass None or a default).
                logger.error(f"Missing feature in input: {col_name}")
                return jsonify({'error': f'Missing required feature: {col_name}'}), 400

            casted_value = None
            spark_type = None
            try:
                if original_type_str.lower() in ['integer', 'int', 'long']:
                    casted_value = int(raw_value)
                    spark_type = IntegerType()
                elif original_type_str.lower() in ['double', 'float', 'decimal']: # Added decimal
                    casted_value = float(raw_value)
                    spark_type = DoubleType()
                elif original_type_str.lower() == 'string':
                    casted_value = str(raw_value)
                    spark_type = StringType()
                elif original_type_str.lower() == 'boolean': # Added boolean
                    casted_value = bool(raw_value)
                    spark_type = BooleanType()
                else:
                    logger.warning(f"Unknown type '{original_type_str}' for feature '{col_name}'. Attempting to cast to string.")
                    casted_value = str(raw_value)
                    spark_type = StringType()
            except ValueError as ve:
                logger.error(f"Type casting error for feature '{col_name}' (expected {original_type_str}, got '{raw_value}'): {str(ve)}")
                return jsonify({'error': f"Invalid value for feature '{col_name}'. Expected type {original_type_str}."}), 400
            
            data_row_values.append(casted_value)
            struct_fields.append(StructField(col_name, spark_type, True)) # True for nullable

        if not data_row_values or not struct_fields:
             return jsonify({'error': 'No valid features processed from input schema for prediction.'}), 400

        data_for_df = [tuple(data_row_values)]
        schema_for_df = StructType(struct_fields)
        
        spark_session = spark_manager.get_spark_session()
        input_spark_df = spark_session.createDataFrame(data_for_df, schema=schema_for_df)
        logger.info("Input DataFrame for prediction schema:")
        input_spark_df.printSchema()
        logger.info("Input DataFrame for prediction data:")
        input_spark_df.show(truncate=False)

        # Make prediction
        predictions_df = loaded_model.transform(input_spark_df)
        logger.info("Predictions DataFrame schema:")
        predictions_df.printSchema()
        logger.info("Predictions DataFrame data:")
        predictions_df.show(truncate=False)

        # Extract the prediction
        # The prediction column is typically named "prediction" by Spark ML models
        if "prediction" not in predictions_df.columns:
            logger.error("'prediction' column not found in predictions output.")
            return jsonify({'error': "Failed to get 'prediction' column from model output."}), 500
            
        prediction_value = predictions_df.select("prediction").first()[0]
        
        logger.info(f"Prediction successful. Value: {prediction_value}")
        return jsonify({'prediction': prediction_value})

    except Exception as e:
        logger.error(f"Error in /predict endpoint: {str(e)}")
        logger.error(traceback.format_exc())
        return jsonify({'error': f'Prediction error: {str(e)}'}), 500

# New endpoint to download the trained model
@app.route('/download_model/<file_id>')
def download_model(file_id):
    metadata = mongo.get_file_metadata(file_id)
    model_path = metadata.get('model_path')
    if not model_path or not os.path.exists(model_path):
        return jsonify({'error': 'Model not found'}), 404
    return send_from_directory(os.path.dirname(model_path), os.path.basename(model_path), as_attachment=True)

@app.route('/logs', methods=['GET'])
def get_logs():
    logs = log_capture.getvalue()
    return jsonify({'logs': logs})

@app.route('/model_status/<file_id>')
def model_status(file_id):
    metadata = mongo.get_file_metadata(file_id)
    progress = metadata.get('model_training_progress', 0)
    message = metadata.get('model_training_message', '')
    return jsonify({'progress': progress, 'message': message})

def open_browser():
    try:
        webbrowser.open('http://127.0.0.1:8080/')
    except Exception as e:
        logger.error(f"Failed to open browser: {str(e)}")

if __name__ == '__main__':
    try:
        Timer(2, open_browser).start()  # Increased delay to 2 seconds
        logger.info("Starting Flask application on port 8080...")
        app.run(host='127.0.0.1', port=8080, debug=True)
    except Exception as e:
        logger.error(f"Failed to start Flask application: {str(e)}")
        logger.error(traceback.format_exc()) 