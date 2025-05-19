from pyspark.ml.feature import (
    StandardScaler, MinMaxScaler, Imputer, StringIndexer, 
    OneHotEncoder, VectorAssembler, PCA
)
from pyspark.sql.functions import col, when, isnan, isnull, count
from pyspark.sql.types import NumericType, DoubleType
import logging
from pyspark.sql import functions as F
from pyspark.ml import Pipeline

# Set up logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

class DataPreprocessor:
    def __init__(self):
        """Initialize the data preprocessor"""
        self.scalers = {}
        self.indexers = {}
        self.pca_model = None
    
    def get_data_summary(self, df):
        """Get summary statistics for the dataframe"""
        try:
            logger.info("Getting data summary...")
            summary = df.summary()
            logger.info(f"Summary statistics:\n{summary.show()}")
            return summary
        except Exception as e:
            logger.error(f"Error in get_data_summary: {str(e)}")
            raise
    
    def handle_missing_values(self, df, strategy='mean'):
        """Handle missing values in the dataframe"""
        try:
            logger.info(f"Handling missing values using {strategy} strategy...")
            numeric_cols = [f.name for f in df.schema.fields if f.dataType == DoubleType()]
            
            if strategy == 'mean':
                for col in numeric_cols:
                    mean_val = df.select(F.mean(col)).collect()[0][0]
                    df = df.fillna(mean_val, subset=[col])
            elif strategy == 'median':
                for col in numeric_cols:
                    median_val = df.approxQuantile(col, [0.5], 0.01)[0]
                    df = df.fillna(median_val, subset=[col])
                    
            logger.info("Missing values handled successfully")
            return df
        except Exception as e:
            logger.error(f"Error in handle_missing_values: {str(e)}")
            raise
    
    def scale_features(self, df, columns):
        """Scale numeric features using StandardScaler"""
        try:
            logger.info(f"Scaling features: {columns}")
            assembler = VectorAssembler(inputCols=columns, outputCol="features")
            scaler = StandardScaler(inputCol="features", outputCol="scaled_features")
            
            pipeline = Pipeline(stages=[assembler, scaler])
            model = pipeline.fit(df)
            df_scaled = model.transform(df)
            
            # Drop original features and keep scaled ones
            for col in columns:
                df_scaled = df_scaled.drop(col)
            df_scaled = df_scaled.withColumnRenamed("scaled_features", "features")
            
            logger.info("Features scaled successfully")
            return df_scaled
        except Exception as e:
            logger.error(f"Error in scale_features: {str(e)}")
            raise
    
    def encode_categorical(self, df, columns):
        """Encode categorical features using StringIndexer"""
        try:
            logger.info(f"Encoding categorical features: {columns}")
            indexers = [StringIndexer(inputCol=col, outputCol=f"{col}_index") for col in columns]
            pipeline = Pipeline(stages=indexers)
            model = pipeline.fit(df)
            df_encoded = model.transform(df)
            
            # Drop original categorical columns
            for col in columns:
                df_encoded = df_encoded.drop(col)
                
            logger.info("Categorical features encoded successfully")
            return df_encoded
        except Exception as e:
            logger.error(f"Error in encode_categorical: {str(e)}")
            raise
    
    def reduce_dimensionality(self, df, n_components=2):
        """Reduce dimensionality using PCA"""
        try:
            logger.info(f"Reducing dimensionality to {n_components} components")
            pca = PCA(k=n_components, inputCol="features", outputCol="pca_features")
            model = pca.fit(df)
            
            # Get explained variance
            explained_variance = model.explainedVariance.toArray()
            logger.info(f"Explained variance: {explained_variance}")
            
            df_reduced = model.transform(df)
            df_reduced = df_reduced.drop("features")
            df_reduced = df_reduced.withColumnRenamed("pca_features", "features")
            
            logger.info("Dimensionality reduction completed successfully")
            return df_reduced
        except Exception as e:
            logger.error(f"Error in reduce_dimensionality: {str(e)}")
            raise
    
    def detect_outliers(self, df, columns, method='zscore', threshold=3):
        """Detect outliers in numeric columns"""
        try:
            logger.info(f"Detecting outliers using {method} method")
            outliers = {}
            
            for col in columns:
                if method == 'zscore':
                    stats = df.select(col).summary().collect()
                    mean = float(stats[1][1])
                    std = float(stats[2][1])
                    
                    df = df.withColumn(f"{col}_zscore", 
                                     (F.col(col) - mean) / std)
                    outliers[col] = df.filter(F.abs(F.col(f"{col}_zscore")) > threshold)
                    df = df.drop(f"{col}_zscore")
            
            logger.info("Outlier detection completed successfully")
            return outliers
        except Exception as e:
            logger.error(f"Error in detect_outliers: {str(e)}")
            raise 