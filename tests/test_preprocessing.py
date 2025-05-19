import pytest
from pyspark.sql import SparkSession
from pyspark.sql.types import StructType, StructField, StringType, IntegerType, FloatType
import pandas as pd
import numpy as np
from preprocessing import DataPreprocessor
import os
import logging

# Set up logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

@pytest.fixture(scope="session")
def spark():
    """Create a Spark session for testing"""
    try:
        # Set environment variables for Windows
        os.environ['HADOOP_HOME'] = 'C:\\hadoop-3.3.6'
        os.environ['JAVA_HOME'] = 'C:\\Program Files\\Java\\jdk-11'
        os.environ['SPARK_LOCAL_IP'] = '127.0.0.1'
        
        # Create Spark session with Windows-specific configurations
        spark = SparkSession.builder \
            .appName("TestPreprocessing") \
            .master("local[*]") \
            .config("spark.driver.memory", "1g") \
            .config("spark.executor.memory", "1g") \
            .config("spark.sql.execution.arrow.pyspark.enabled", "false") \
            .config("spark.python.worker.reuse", "false") \
            .config("spark.driver.maxResultSize", "512m") \
            .config("spark.sql.shuffle.partitions", "2") \
            .config("spark.default.parallelism", "2") \
            .getOrCreate()
        
        logger.info("Spark session created successfully")
        logger.info(f"Spark version: {spark.version}")
        logger.info(f"Spark config: {spark.sparkContext.getConf().getAll()}")
        
        yield spark
        
        logger.info("Stopping Spark session")
        spark.stop()
    except Exception as e:
        logger.error(f"Error creating Spark session: {str(e)}")
        raise

@pytest.fixture(scope="session")
def sample_df(spark):
    """Create a sample DataFrame for testing"""
    # Create schema
    schema = StructType([
        StructField("id", IntegerType(), True),
        StructField("name", StringType(), True),
        StructField("age", IntegerType(), True),
        StructField("salary", FloatType(), True),
        StructField("department", StringType(), True)
    ])
    
    # Create data
    data = [
        (1, "John", 30, 50000.0, "IT"),
        (2, "Alice", None, 60000.0, "HR"),
        (3, "Bob", 35, None, "IT"),
        (4, None, 40, 55000.0, None),
        (5, "Eve", 32, 65000.0, "HR")
    ]
    
    return spark.createDataFrame(data, schema)

@pytest.fixture
def preprocessor(spark):
    """Create DataPreprocessor instance"""
    return DataPreprocessor()

def test_get_data_summary(preprocessor, sample_df):
    """Test data summary generation"""
    try:
        logger.info("Testing get_data_summary...")
        summary = preprocessor.get_data_summary(sample_df)
        # summary is now a DataFrame, so just check it's a DataFrame and has expected columns
        assert hasattr(summary, 'columns')
        assert 'summary' in summary.columns
    except Exception as e:
        logger.error(f"Error in test_get_data_summary: {str(e)}")
        raise

def test_handle_missing_values(preprocessor, sample_df):
    """Test missing value handling"""
    try:
        logger.info("Testing handle_missing_values...")
        processed_df = preprocessor.handle_missing_values(sample_df)
        # Check if no nulls remain in numeric columns
        for column in ['age', 'salary']:
            null_count = processed_df.filter(processed_df[column].isNull()).count()
            logger.info(f"Column {column} has {null_count} null values")
            assert null_count == 0
    except Exception as e:
        logger.error(f"Error in test_handle_missing_values: {str(e)}")
        raise

def test_scale_features(preprocessor, sample_df):
    """Test feature scaling"""
    try:
        logger.info("Testing scale_features...")
        numeric_cols = ['age', 'salary']
        scaled_df = preprocessor.scale_features(sample_df, numeric_cols)
        logger.info(f"Scaled columns: {scaled_df.columns}")
        assert 'features' in scaled_df.columns
    except Exception as e:
        logger.error(f"Error in test_scale_features: {str(e)}")
        raise

def test_encode_categorical(preprocessor, sample_df):
    """Test categorical encoding"""
    try:
        logger.info("Testing encode_categorical...")
        categorical_cols = ['department']
        encoded_df = preprocessor.encode_categorical(sample_df, categorical_cols)
        logger.info(f"Encoded columns: {encoded_df.columns}")
        # Should have 'department_index' column and not 'department'
        assert 'department_index' in encoded_df.columns
        assert 'department' not in encoded_df.columns
    except Exception as e:
        logger.error(f"Error in test_encode_categorical: {str(e)}")
        raise

def test_reduce_dimensionality(preprocessor, sample_df):
    """Test PCA dimensionality reduction"""
    try:
        logger.info("Testing reduce_dimensionality...")
        # First scale features to get a 'features' column
        numeric_cols = ['age', 'salary']
        scaled_df = preprocessor.scale_features(sample_df, numeric_cols)
        reduced_df = preprocessor.reduce_dimensionality(scaled_df, n_components=1)
        logger.info(f"Transformed columns: {reduced_df.columns}")
        assert 'features' in reduced_df.columns
    except Exception as e:
        logger.error(f"Error in test_reduce_dimensionality: {str(e)}")
        raise

def test_detect_outliers(preprocessor, sample_df):
    """Test outlier detection"""
    try:
        logger.info("Testing detect_outliers...")
        numeric_cols = ['age', 'salary']
        outliers = preprocessor.detect_outliers(sample_df, numeric_cols, method='zscore', threshold=3)
        logger.info(f"Outliers: {outliers}")
        assert 'age' in outliers
        assert 'salary' in outliers
    except Exception as e:
        logger.error(f"Error in test_detect_outliers: {str(e)}")
        raise 