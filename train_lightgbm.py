import os
import gc
import pandas as pd
import numpy as np
import lightgbm as lgb
from pyspark.sql import SparkSession

# 1. Initialize Spark Session with Memory Limits
# Disabling Arrow optimization ('spark.sql.execution.arrow.pyspark.enabled': 'false') 
# because it caused JVM crashes/memory leaks on this specific setup.
# Reduced memory to 2g to be safer on laptops.
print("Initializing Spark Session...")
# Increasing memory to 8g and enabling Off-Heap memory to handle the 58M row transfer.
# Arrow uses off-heap memory, so enabling it explicitly helps avoid Heap Space OOM.
# 1. Initialize Spark Session with Memory Limits
# Reducing memory to 4g to leave space for the OS and Python process (Pandas DF).
# Enabling Arrow for faster transfer, but keeping off-heap moderate.
print("Initializing Spark Session...")
spark = SparkSession.builder \
    .appName("M5_LGBM_Training") \
    .config("spark.driver.memory", "4g") \
    .config("spark.driver.maxResultSize", "0") \
    .config("spark.sql.execution.arrow.pyspark.enabled", "true") \
    .config("spark.memory.offHeap.enabled", "true") \
    .config("spark.memory.offHeap.size", "2g") \
    .getOrCreate()

print(f"Spark Version: {spark.version}")

# 2. Load Data from Parquet
# Adjust this path if 'final_optimized.parquet' is in a different subdirectory
DATA_PATH = "ETL Process/final_optimized.parquet"
MODEL_DIR = "models"
os.makedirs(MODEL_DIR, exist_ok=True)

# Read Parquet file
print(f"Reading data from {DATA_PATH}...")
df_spark = spark.read.parquet(DATA_PATH)

# Sampling to fit in memory
# 58M rows * 20 cols * 4 bytes approx 4.5Gb raw, but Pandas overhead is 2-3x.
# Full dataset requires ~16GB+ RAM just for the DataFrame.
SAMPLING_RATIO = 0.1  # Train on 10% of data
if SAMPLING_RATIO < 1.0:
    print(f"Sampling {SAMPLING_RATIO*100}% of data to fit in memory...")
    df_spark = df_spark.sample(withReplacement=False, fraction=SAMPLING_RATIO, seed=42)

print(f"Total Rows (after sampling): {df_spark.count():,}")
# df_spark.printSchema()ration
# Optimize types WITHIN Spark to reduce transfer size
print("Optimizing data types in Spark before transfer...")
from pyspark.sql.types import DoubleType, LongType
from pyspark.sql.functions import col

# Select columns with cast
select_exprs = []
for field in df_spark.schema.fields:
    if isinstance(field.dataType, DoubleType):
        # Cast double (8 bytes) to float (4 bytes)
        select_exprs.append(col(field.name).cast("float").alias(field.name))
    elif isinstance(field.dataType, LongType):
        # Cast long (8 bytes) to int (4 bytes) - assuming values fit in int32
        select_exprs.append(col(field.name).cast("int").alias(field.name))
    else:
        select_exprs.append(col(field.name))

df_spark_optimized = df_spark.select(*select_exprs)

# Convert to Pandas (Using Arrow optimization)
print("Converting to Pandas (this might take a minute)...")
df = df_spark_optimized.toPandas()

print("Data loaded into Pandas via Spark.")
print(df.info())

print("Data loaded into Pandas via Spark.")
print(df.info())

# Free up Spark memory
spark.stop()
print("Spark Session stopped.")

# 4. LightGBM Training (GPU Optimized)
# Define features and target
TARGET = 'sales'
X = df.drop(columns=[TARGET])
y = df[TARGET]

# Drop non-numeric/unnecessary columns
# 'd' is day index (d_1, ...), 'id' is string id, 'date' is datetime (if present), 'wm_yr_wk' is id
# 'item_id' removed because high cardinality (~3000) breaks GPU training (bin size limit)
drop_cols = ['d', 'id', 'date', 'wm_yr_wk', 'item_id']
X = X.drop(columns=drop_cols, errors='ignore')
print(f"Dropped columns: {drop_cols}")

# Identify categorical features automatically or manually
# Common M5 columns: item_id, dept_id, cat_id, store_id, state_id
cat_feats = [c for c in X.columns if c in [
    'item_id', 'dept_id', 'cat_id', 'store_id', 'state_id', 
    'year', 'month', 'wday', 'event_name_1', 'event_type_1', 
    'event_name_2', 'event_type_2'
]]

print(f"Categorical Features: {cat_feats}")

# Convert object columns to category type for LGBM
# Anything remaining that is object must be converted or dropped
for c in X.columns:
    if X[c].dtype == 'object':
        X[c] = X[c].astype('category')
        if c not in cat_feats:
            cat_feats.append(c)

train_data = lgb.Dataset(X, label=y, categorical_feature=cat_feats)

# GPU Configuration
params = {
    'objective': 'tweedie',
    'metric': 'rmse',
    'boosting_type': 'gbdt',
    'learning_rate': 0.05,
    'num_leaves': 63,
    'feature_fraction': 0.8,
    'bagging_fraction': 0.8,
    'bagging_freq': 5,
    'verbose': 1,
    'n_jobs': -1,
    'seed': 42,
    
    # GPU Parameters for RTX 3050
    'device': 'gpu',
    'gpu_platform_id': 0,
    'gpu_device_id': 0,
    'force_col_wise': True, # Optimized for column-wise parallelism
    'max_bin': 63 # Constrain bins for GPU compatibility
}

print("Starting training with GPU...")
model = lgb.train(
    params,
    train_data,
    num_boost_round=1000,
    valid_sets=[train_data],
    valid_names=['train'],
    callbacks=[
        lgb.early_stopping(stopping_rounds=50),
        lgb.log_evaluation(period=50)
    ]
)

print("Training completed!")

# 5. Save Model and Feature Importance
# Save model
model_path = os.path.join(MODEL_DIR, 'lgb_gpu_model.txt')
model.save_model(model_path)
print(f"Model saved to {model_path}")

# Feature Importance
importance = pd.DataFrame({
    'Feature': model.feature_name(),
    'Importance': model.feature_importance(importance_type='gain')
}).sort_values(by='Importance', ascending=False)

print("\nTop 10 Features:")
print(importance.head(10))
