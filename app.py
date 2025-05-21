import os

# Set Hadoop home and update PATH for winutils.exe
os.environ["HADOOP_HOME"] = r"C:\hadoop-3.3.6"
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

# --- Add file logging for production ---
file_handler = logging.FileHandler('app.log')
file_handler.setLevel(logging.INFO)
file_formatter = logging.Formatter('%(asctime)s %(levelname)s %(name)s %(threadName)s : %(message)s')
file_handler.setFormatter(file_formatter)
logger.addHandler(file_handler)

class JSONEncoder(json.JSONEncoder):
    def default(self, obj):
        try:
            from datetime import datetime
            if isinstance(obj, datetime):
                return obj.isoformat()  # Convert datetime to ISO format string
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
from config.cors_config import CORSConfig

# Configure CORS with secure settings
cors_config = CORSConfig()
CORS(app, resources=cors_config.resources)
app.config.from_object(Config)

# Define allowed file extensions
app.config['ALLOWED_EXTENSIONS'] = Config.ALLOWED_EXTENSIONS
app.json_encoder = JSONEncoder

# Set maximum content length from config
app.config['MAX_CONTENT_LENGTH'] = Config.MAX_CONTENT_LENGTH

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

from middleware.security import SecurityMiddleware

@app.route('/upload', methods=['POST'])
@SecurityMiddleware.validate_content_length
@SecurityMiddleware.validate_file_extension
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

        # --- ASYNC AUTO-TRAIN MODELS ON UPLOAD ---
        # (Removed: No auto-training after upload)

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
@SecurityMiddleware.validate_content_length
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
        
        # --- IMMEDIATELY HANDLE EXPLORATORY ANALYSIS ---
        if analysis_type == 'exploratory':
            results = analyzer.analyze(
                analysis_type=analysis_type,
                selected_features=selected_features,
                target_feature=target_feature
            )
            logger.info("Exploratory analysis completed")
            # Store results under 'tasks' for frontend compatibility
            mongo.update_file_metadata(file_id, {
                'status': 'analyzed',
                'tasks': {
                    'exploratory': {
                        'timestamp': datetime.datetime.utcnow(),
                        'parameters': {
                            'selected_features': selected_features,
                            'target_feature': target_feature
                        },
                        'results': results
                    }
                }
            })
            return jsonify({'results': results}), 200
        # --- END EXPLORATORY BLOCK ---

        # Validate target for regression/classification
        if analysis_type in ['regression', 'classification']:
            if not target_feature or target_feature not in df.columns:
                return jsonify({'error': 'You must select a valid target column for this analysis type.'}), 400
        # For exploratory analysis, skip model training
        if analysis_type == 'exploratory':
            results = analyzer.analyze(
                analysis_type=analysis_type,
                selected_features=selected_features,
                target_feature=target_feature
            )
            logger.info("Exploratory analysis completed")
            mongo.update_file_metadata(file_id, {
                'status': 'analyzed',
                'analysis_type': analysis_type,
                'results': results
            })
            logger.info("Analysis process completed successfully")
            return jsonify(results)

        # Perform analysis
        logger.info("Starting analysis")
        # Check if the selected features/target match any existing model
        model_key = f"model_{'_'.join(selected_features)}_{target_feature}"
        metadata = mongo.get_file_metadata(file_id)  # Refresh metadata
        if 'trained_models' not in metadata:
            metadata['trained_models'] = {}
        trained_models = metadata['trained_models']
        model_path = trained_models.get(model_key)

        model = None
        just_trained = False
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
                # Store the feature columns used for training in metadata
                if 'trained_feature_cols' not in metadata:
                    metadata['trained_feature_cols'] = {}
                metadata['trained_feature_cols'][model_key] = selected_features
                metadata['trained_models'] = trained_models
                mongo.update_file_metadata(file_id, metadata)
                logger.info(f"Trained and saved new {analysis_type} model for selected features/target.")
                just_trained = True
            else:
                return jsonify({'message': 'Model training failed.'}), 500

        # Use the trained model directly if just trained, otherwise load from disk
        if just_trained and model is not None:
            pipeline_model = model
        else:
            from pyspark.ml import PipelineModel
            if model_path:
                pipeline_model = PipelineModel.load(model_path)
            else:
                return jsonify({'message': 'Model is still being trained. Please wait and try again.'}), 202

        predictions = pipeline_model.transform(df)
        results = analyzer.analyze(
            analysis_type=analysis_type,
            selected_features=selected_features,
            target_feature=target_feature,
            predictions=predictions  # Pass predictions if needed
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
            
            # --- Suggest only suitable target features ---
            total_rows = len(df)
            possible_targets = []
            for col in df.columns:
                try:
                    # Always allow numeric columns as regression targets
                    if str(df[col].dtype).startswith(('int', 'float')):
                        possible_targets.append(col)
                        continue
                    unique_vals = df[col].nunique(dropna=True)
                    # For classification: less than 50% unique (categorical)
                    if unique_vals < 0.5 * total_rows and col not in possible_targets:
                        possible_targets.append(col)
                except TypeError:
                    # Skip columns with unhashable types (like dicts)
                    continue

            return jsonify({
                'available_features': df.columns.tolist(),
                'numeric_features': df.select_dtypes(include=[np.number]).columns.tolist(),
                'categorical_features': df.select_dtypes(include=['object']).columns.tolist(),
                'suggested_tasks': suggested_tasks,
                'data_shape': {'rows': len(df), 'columns': len(df.columns)},
                'possible_targets': possible_targets
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
            
            # Check that all selected columns exist in the DataFrame
            required_cols = feature_cols + ([target_col] if target_col else [])
            missing_cols = [col for col in required_cols if col not in df.columns]
            if missing_cols:
                return jsonify({'error': f"Selected columns not found in data: {missing_cols}"}), 400
            
            # --- Check if target is suitable for modeling ---
            if target_col:
                total_rows = df.count()
                unique_targets = df.select(target_col).distinct().count()
                # For regression: warn if target is almost unique
                if task_type == 'regression' and unique_targets / total_rows > 0.9:
                    return jsonify({'error': 'Choose a more appropriate target and features, and your model should train successfully.'}), 400
                # For classification: warn if too many classes
                if task_type == 'classification' and unique_targets > 0.5 * total_rows:
                    return jsonify({'error': 'Choose a more appropriate target and features, and your model should train successfully.'}), 400
            
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
            # --- Detect categorical and numeric features ---
            # Get Spark dtypes as a dict
            spark_dtypes = dict(df.dtypes)
            categorical_cols = [col for col in feature_cols if spark_dtypes[col] == 'string']
            numeric_cols = [col for col in feature_cols if spark_dtypes[col] != 'string']
            # Prepare indexers for categorical features
            from pyspark.ml.feature import StringIndexer
            indexers = [
                StringIndexer(inputCol=col, outputCol=f"{col}_indexed", handleInvalid="keep")
                for col in categorical_cols
            ]
            assembler_inputs = [f"{col}_indexed" for col in categorical_cols] + numeric_cols
            # --- Regression ---
            if task_type == 'regression':
                from pyspark.ml import Pipeline
                from pyspark.ml.feature import VectorAssembler
                from pyspark.ml.regression import LinearRegression
                stages = []
                # Index target if string
                if spark_dtypes[target_col] == 'string':
                    label_indexer = StringIndexer(inputCol=target_col, outputCol="label", handleInvalid="keep")
                    stages.append(label_indexer)
                    df = label_indexer.fit(df).transform(df)
                else:
                    df = df.withColumnRenamed(target_col, "label")
                stages += indexers
                assembler = VectorAssembler(inputCols=assembler_inputs, outputCol="features")
                lr = LinearRegression(featuresCol="features", labelCol="label")
                stages += [assembler, lr]
                pipeline = Pipeline(stages=stages)
                pipeline_model = pipeline.fit(df)
                predictions = pipeline_model.transform(df)
                results['model_performance'] = {'rmse': None}  # You can add metrics if needed
            # --- Classification ---
            elif task_type == 'classification':
                from pyspark.ml import Pipeline
                from pyspark.ml.feature import VectorAssembler, StringIndexer
                from pyspark.ml.classification import LogisticRegression
                stages = []
                # Index target if string
                if spark_dtypes[target_col] == 'string':
                    label_indexer = StringIndexer(inputCol=target_col, outputCol="label", handleInvalid="keep")
                    stages.append(label_indexer)
                    df = label_indexer.fit(df).transform(df)
                else:
                    df = df.withColumnRenamed(target_col, "label")
                stages += indexers
                assembler = VectorAssembler(inputCols=assembler_inputs, outputCol="features")
                lr = LogisticRegression(featuresCol="features", labelCol="label")
                stages += [assembler, lr]
                pipeline = Pipeline(stages=stages)
                pipeline_model = pipeline.fit(df)
                model_path = os.path.join(app.root_path, 'auto_models', f"classification_{file_id}_{'_'.join(feature_cols)}_{target_col}")
                pipeline_model.save(model_path)
                if 'trained_feature_cols' not in metadata:
                    metadata['trained_feature_cols'] = {}
                model_key = f"classification_{'_'.join(feature_cols)}_{target_col}"
                metadata['trained_feature_cols'][model_key] = feature_cols
                if 'trained_models' not in metadata:
                    metadata['trained_models'] = {}
                metadata['trained_models'][model_key] = model_path
                mongo.update_file_metadata(file_id, metadata)
                predictions = pipeline_model.transform(df)
                results['model_performance'] = {'accuracy': None}  # You can add metrics if needed
            # --- Clustering ---
            elif task_type == 'clustering':
                from pyspark.ml import Pipeline
                from pyspark.ml.feature import VectorAssembler
                from pyspark.ml.clustering import KMeans
                n_clusters = task_params.get('n_clusters', 3)
                stages = indexers
                assembler = VectorAssembler(inputCols=assembler_inputs, outputCol="features")
                kmeans = KMeans(featuresCol="features", k=n_clusters, seed=42)
                stages += [assembler, kmeans]
                pipeline = Pipeline(stages=stages)
                pipeline_model = pipeline.fit(df)
                model_path = os.path.join(app.root_path, 'auto_models', f"clustering_{file_id}_{'_'.join(feature_cols)}")
                pipeline_model.save(model_path)
                if 'trained_feature_cols' not in metadata:
                    metadata['trained_feature_cols'] = {}
                model_key = f"clustering_{'_'.join(feature_cols)}"
                metadata['trained_feature_cols'][model_key] = feature_cols
                if 'trained_models' not in metadata:
                    metadata['trained_models'] = {}
                metadata['trained_models'][model_key] = model_path
                mongo.update_file_metadata(file_id, metadata)
                predictions = pipeline_model.transform(df)
                results['cluster_sizes'] = None  # You can add cluster info if needed
            # --- Correlation Analysis ---
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

# --- Health check endpoint for monitoring ---
@app.route('/health', methods=['GET'])
def health_check():
    return jsonify({'status': 'ok'}), 200

# --- Flask error handlers for graceful error responses ---
@app.errorhandler(500)
def internal_error(error):
    logger.error(f"Internal server error: {error}")
    return jsonify({'error': 'Internal server error'}), 500

@app.errorhandler(404)
def not_found_error(error):
    logger.warning(f"Not found: {error}")
    return jsonify({'error': 'Not found'}), 404

if __name__ == '__main__':
    try:
        Timer(2, open_browser).start()  # Increased delay to 2 seconds
        logger.info("Starting Flask application on port 8080...")
        import os
        debug_mode = os.environ.get('FLASK_DEBUG', '0') == '1'
        app.run(host='0.0.0.0', port=8080, debug=debug_mode)
    except Exception as e:
        logger.error(f"Failed to start Flask application: {str(e)}")
        logger.error(traceback.format_exc()) 