import os
import gc
import pandas as pd
import numpy as np
import lightgbm as lgb
from pyspark.sql import SparkSession
from pyspark.sql.functions import col, hash, split
from pyspark.sql.types import IntegerType, DoubleType
from onnxmltools import convert_lightgbm
from onnxconverter_common.data_types import FloatTensorType

# Paths
BASE_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
DATA_PATH = os.path.join(BASE_DIR, "ETL Process", "final_optimized.parquet")

def train():
    print("Initializing Spark...")
    spark = SparkSession.builder \
        .appName("M5_LGBM_Training") \
        .config("spark.driver.memory", "4g") \
        .getOrCreate()

    print(f"Reading data from {DATA_PATH}...")
    df_spark = spark.read.parquet(DATA_PATH)
    
    # 1. Series Sampling (Critical for Lags)
    print("Sampling ~5% of Series (items) to preserve time continuity...")
    df_spark_sample = df_spark.filter(hash(col("item_id")) % 20 == 0)
    
    print("Optimizing types...")
    # Quick cast to float/int to save RAM
    select_exprs = []
    for field in df_spark_sample.schema.fields:
        if isinstance(field.dataType, DoubleType):
            select_exprs.append(col(field.name).cast("float").alias(field.name))
        else:
            select_exprs.append(col(field.name))
    df_spark_sample = df_spark_sample.select(*select_exprs)

    print("Converting to Pandas...")
    df = df_spark_sample.toPandas()
    spark.stop()

    # 2. Feature Engineering (Grandmaster Strategy)
    print("Generating Features...")
    df['d_int'] = df['d'].apply(lambda x: int(x.split('_')[1]))
    df = df.sort_values(by=['item_id', 'store_id', 'd_int'])

    grouped = df.groupby(['item_id', 'store_id'])['sales']
    
    # Lags
    lags = [28, 35, 42, 49]
    for lag in lags:
        df[f'lag_{lag}'] = grouped.shift(lag)
        
    # Rolling (Transform for safety)
    rolling_windows = [28]
    for win in rolling_windows:
        df[f'roll_mean_{win}'] = grouped.shift(28).transform(lambda x: x.rolling(win).mean())
        df[f'roll_std_{win}'] = grouped.shift(28).transform(lambda x: x.rolling(win).std())

    # Price Features
    print("Generating Price Features...")
    df['price_max'] = df.groupby(['item_id', 'store_id'])['sell_price'].transform('max')
    df['price_momentum'] = df['sell_price'] / df['price_max']
    df['price_roll_std_7'] = df.groupby(['item_id', 'store_id'])['sell_price'].transform(lambda x: x.rolling(7).std())

    # Fill NaNs
    feature_cols = [f'lag_{l}' for l in lags] + \
                   [f'roll_mean_{w}' for w in rolling_windows] + \
                   [f'roll_std_{w}' for w in rolling_windows] + \
                   ['price_momentum', 'price_roll_std_7']
    df[feature_cols] = df[feature_cols].fillna(-1)

    # Prepare Data
    TARGET = 'sales'
    drop_cols = ['d', 'd_int', 'id', 'date', 'wm_yr_wk', 'item_id', TARGET]
    
    # Categoricals
    cat_cols = ['store_id', 'dept_id', 'cat_id', 'state_id', 'event_name_1', 'event_type_1']
    for c in cat_cols:
        if c in df.columns:
            df[c] = df[c].astype('category')

    X = df.drop(columns=drop_cols, errors='ignore')
    y = df[TARGET]

    # Create Dataset
    train_data = lgb.Dataset(
        X, label=y, categorical_feature=[c for c in cat_cols if c in X.columns], free_raw_data=False
    )
    
    # Parameters (Matching Sklearn Optimization)
    params = {
        'objective': 'tweedie', # Better than Poisson for Zero-Inflated
        'metric': 'rmse',
        'learning_rate': 0.01,
        'num_leaves': 127,
        'min_data_in_leaf': 20,
        'tweedie_variance_power': 1.1,
        'verbose': -1,
        'seed': 42
    }
    
    print("Training LightGBM...")
    model = lgb.train(
        params,
        train_data,
        num_boost_round=200
    )
    
    # Save .txt
    model_path = os.path.join(BASE_DIR, "models", "LightGBM", "lightgbm_model.txt")
    model.save_model(model_path)
    print(f"Model saved to {model_path}")

    # Export to ONNX
    print("Exporting to ONNX...")
    try:
        # Define Input Type: FloatTensorType
        # ONNX usually expects Floats. LightGBM handles Categoricals internally, 
        # but ONNX converters often prefer one numerical input tensor if we don't define complex schemas.
        # Simple approach: Treat all as Float (Categoricals are often ints under the hood), 
        # BUT this loses category info.
        # Better: onnxmltools can handle passing a dataframe-like structure if defined correctly.
        # Standard approach for simple LightGBM: 1 Input Variable (FloatTensor) of shape [None, n_features]
        
        initial_types = [('input', FloatTensorType([None, X.shape[1]]))]
        onnx_model = convert_lightgbm(model, initial_types=initial_types, target_opset=14)
        
        onnx_path = os.path.join(BASE_DIR, "models", "LightGBM", "m5_lightgbm.onnx")
        with open(onnx_path, "wb") as f:
            f.write(onnx_model.SerializeToString())
        print(f"ONNX Model saved to {onnx_path}")
        
    except Exception as e:
        print(f"ONNX Error: {e}")
        import traceback
        traceback.print_exc()

if __name__ == "__main__":
    train()
