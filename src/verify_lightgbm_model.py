import lightgbm as lgb
import pandas as pd
import numpy as np
import os
from pyspark.sql import SparkSession

# Paths (Relative to src/ directory)
# Paths
DATA_PATH = "/home/naraka/Documents/DevProjects/M5-Sales-Forecasting/data/train.parquet"
MODEL_PATH = "/home/naraka/Documents/DevProjects/M5-Sales-Forecasting/models/LightGBM/lgb_gpu_model.txt"

def evaluate():
    print("Initializing Spark for data loading...")
    spark = SparkSession.builder \
        .appName("Model_Evaluation") \
        .config("spark.driver.memory", "2g") \
        .getOrCreate()
        
    print(f"Loading validation sample from {DATA_PATH}...")
    # Load a random 5% sample for verification (keep it fast and low memory)
    df_spark = spark.read.parquet(DATA_PATH).sample(fraction=0.05, seed=123)
    
    # Convert to Pandas
    print("Converting to Pandas...")
    df = df_spark.toPandas()
    spark.stop()
    
    # Prepare Features
    TARGET = 'sales'
    drop_cols = ['d', 'id', 'date', 'wm_yr_wk', 'item_id', TARGET]
    
    # Ensure columns match model features
    # Load model to check expected features
    print(f"Loading model from {MODEL_PATH}...")
    try:
        model = lgb.Booster(model_file=MODEL_PATH)
    except Exception as e:
        print(f"Error loading model: {e}")
        return

    features = model.feature_name()
    print(f"Model expects features: {features}")
    
    # Filter DataFrame to expected features
    X = df[features]
    y_true = df[TARGET]
    
    # Preprocess inputs (Category Handling)
    # The model expects integer codes for categories.
    # IN A REAL SCENARIO, we should use the same encoders.
    # For this verification on raw training data, the integer ids (store_id, cat_id) 
    # should already be compatible if they came from the same ETL pipeline.
    for col in X.columns:
        if X[col].dtype == 'object':
             X[col] = X[col].astype('category').cat.codes

    print("Running predictions...")
    # Use raw numpy array to avoid pandas categorical metadata mismatch
    y_pred = model.predict(X.to_numpy())
    
    # Calculate RMSE
    rmse = np.sqrt(np.mean((y_true - y_pred) ** 2))
    
    # Calculate Percentage Metrics
    mean_sales = np.mean(y_true)
    
    # 1. Item-Level Accuracy (Harder due to zeros)
    # WMAPE (Weighted Mean Absolute Percentage Error)
    wmape = np.sum(np.abs(y_true - y_pred)) / np.sum(y_true)
    
    # 2. Business Volume Accuracy (What managers care about)
    # How close is the TOTAL predicted sum to the TOTAL actual sum?
    total_actual = np.sum(y_true)
    total_pred = np.sum(y_pred)
    volume_accuracy = 1.0 - abs(total_actual - total_pred) / total_actual
    
    print(f"\nVALIDATION RESULTS:")
    print(f"-------------------")
    print(f"RMSE Score: {rmse:.4f}")
    print(f"Mean Sales: {mean_sales:.4f}")
    print(f"Item-Level WMAPE: {wmape:.2%}")
    print(f"-------------------")
    print(f"Total Actual Volume: {total_actual:,.0f}")
    print(f"Total Predicted Vol: {total_pred:,.0f}")
    print(f"Business Volume Accuracy: {volume_accuracy:.2%}")

if __name__ == "__main__":
    evaluate()
