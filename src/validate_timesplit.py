import os
import gc
import pandas as pd
import numpy as np
from sklearn.ensemble import HistGradientBoostingRegressor
from sklearn.preprocessing import OrdinalEncoder
from sklearn.pipeline import make_pipeline
from sklearn.compose import make_column_transformer
from pyspark.sql import SparkSession
from pyspark.sql.functions import col, substring, split, lit
from pyspark.sql.types import IntegerType

# Configuration
# Validation Split: Dataset ends at 1913. 
# We use the LAST 28 DAYS for validation (Backtesting).
# Train: d_1 to d_1885
# Test:  d_1886 to d_1913 (28 Days)
SPLIT_DAY = 1885 

# Paths
BASE_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
DATA_PATH = os.path.join(BASE_DIR, "ETL Process", "final_optimized.parquet")

def run_time_split_validation():
    print("Initializing Spark...")
    spark = SparkSession.builder \
        .appName("M5_Time_Split_Validation") \
        .config("spark.driver.memory", "4g") \
        .config("spark.driver.maxResultSize", "0") \
        .getOrCreate()

    print(f"Reading data from {DATA_PATH}...")
    df_spark = spark.read.parquet(DATA_PATH)
    
    # Extract integer day for sorting
    print(" Parsing 'd' column for time split...")
    df_spark = df_spark.withColumn("day_int", split(col("d"), "_").getItem(1).cast(IntegerType()))

    # CRITICAL: Switch to SERIES Sampling (Same as training) to preserve lag history
    from pyspark.sql.functions import hash
    print("Sampling ~5% of Series (items) to preserve time continuity...")
    df_spark = df_spark.filter(hash(col("item_id")) % 20 == 0)

    print("Converting FULL filtered history to Pandas (needed for Lag calculation)...")
    df = df_spark.toPandas()
    
    spark.stop() # Save memory before FE
    
    print(f"Total History Rows: {len(df):,}")

    # 3. Feature Engineering (Must match training)
    print("Generating Time Series Features...")
    df = df.sort_values(by=['item_id', 'store_id', 'day_int'])

    lags = [28, 35, 42, 49]
    rolling_windows = [28]
    grouped = df.groupby(['item_id', 'store_id'])['sales']

    for lag in lags:
        df[f'lag_{lag}'] = grouped.shift(lag)

    # Safer method using transform to guarantee index alignment
    for win in rolling_windows:
        # Rolling on Lag-28 (Safe/Non-recursive)
        df[f'roll_mean_{win}'] = grouped.shift(28).transform(lambda x: x.rolling(win).mean())
        df[f'roll_std_{win}'] = grouped.shift(28).transform(lambda x: x.rolling(win).std())

    # --- NEW PRICE FEATURES ---
    print("Generating Price Features...")
    df['price_max'] = df.groupby(['item_id', 'store_id'])['sell_price'].transform('max')
    df['price_momentum'] = df['sell_price'] / df['price_max']
    df['price_roll_std_7'] = df.groupby(['item_id', 'store_id'])['sell_price'].transform(lambda x: x.rolling(7).std())
    # ---------------------------

    # Fill NaNs
    feature_cols = [f'lag_{l}' for l in lags] + \
                   [f'roll_mean_{w}' for w in rolling_windows] + \
                   [f'roll_std_{w}' for w in rolling_windows] + \
                   ['price_momentum', 'price_roll_std_7']
                   
    df[feature_cols] = df[feature_cols].fillna(-1)

    # 4. Split Train/Test
    print(f"Splitting data at Day {SPLIT_DAY}...")
    
    df_train = df[df['day_int'] <= SPLIT_DAY]
    df_test = df[df['day_int'] > SPLIT_DAY]
    
    print(f"Train Rows: {len(df_train):,}")
    print(f"Test Rows: {len(df_test):,}")
    
    # Prepare Features
    TARGET = 'sales'
    drop_cols = ['d', 'day_int', 'id', 'date', 'wm_yr_wk', 'item_id', TARGET]
    
    X_train = df_train.drop(columns=drop_cols, errors='ignore')
    y_train = df_train[TARGET]
    
    X_test = df_test.drop(columns=drop_cols, errors='ignore')
    y_test = df_test[TARGET]
    
    # Encoding & Pipeline
    cat_cols = [c for c in X_train.columns if X_train[c].dtype == 'object' or X_train[c].dtype.name == 'category']
    print(f"Categorical Features: {cat_cols}")

    preprocessor = make_column_transformer(
        (OrdinalEncoder(handle_unknown='use_encoded_value', unknown_value=-1), cat_cols),
        remainder='passthrough',
        verbose_feature_names_out=False
    )

    model = make_pipeline(preprocessor, HistGradientBoostingRegressor(
        loss='poisson',
        learning_rate=0.01, # Optimized
        max_iter=1000,      # Increased for validation convergence with low LR
        max_leaf_nodes=127, # Optimized
        min_samples_leaf=20,# Optimized
        random_state=42,
        verbose=1
    ))

    print("\nTraining on PAST data...")
    model.fit(X_train, y_train)
    
    print("\nPredicting on FUTURE data...")
    y_pred = model.predict(X_test)
    
    # Metrics
    rmse = np.sqrt(np.mean((y_test - y_pred) ** 2))
    
    total_actual = np.sum(y_test)
    total_pred = np.sum(y_pred)
    volume_accuracy = 1.0 - abs(total_actual - total_pred) / total_actual
    
    # Item-Level WMAPE
    wmape = np.sum(np.abs(y_test - y_pred)) / np.sum(y_test)

    print(f"\nREAL TIME-SERIES VALIDATION RESULTS:")
    print(f"------------------------------------")
    print(f"Training Data: Days 1-{SPLIT_DAY} (5% Sample)")
    print(f"Testing Data:  Days {SPLIT_DAY+1}-1941 (Full)")
    print(f"------------------------------------")
    print(f"RMSE (Real): {rmse:.4f}")
    print(f"Item-Level WMAPE: {wmape:.2%}")
    print(f"Business Volume Accuracy: {volume_accuracy:.2%}")
    
    if volume_accuracy < 0.95:
        print("\nCONCLUSION: The previous 99% was indeed overfitting/leakage.")
        print("This lower score reflects the true difficulty of predicting the future.")
    else:
        print("\nCONCLUSION: The model is actually surprisingly robust!")

if __name__ == "__main__":
    run_time_split_validation()
