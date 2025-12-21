import pandas as pd
import numpy as np
import lightgbm as lgb
import gc
from tqdm import tqdm
from sklearn.metrics import mean_squared_error
from statsmodels.tsa.statespace.sarimax import SARIMAX

# --- CONFIG ---
STORES = ['CA_1', 'CA_2', 'CA_3', 'TX_1', 'TX_2', 'TX_3', 'WI_1', 'WI_2', 'WI_3']
FILE_PATH = '/content/drive/MyDrive/final_optimized.parquet'

# Same Params as Production
PARAMS = {
    'objective': 'tweedie',
    'tweedie_variance_power': 1.1,
    'metric': 'rmse',
    'learning_rate': 0.1,
    'num_leaves': 31,
    'feature_fraction': 0.8,
    'bagging_fraction': 0.8,
    'bagging_freq': 1,
    'n_jobs': -1,
    'verbose': -1,
    'device': 'gpu',
    'gpu_platform_id': 0,
    'gpu_device_id': 0,
    'max_bin': 63,
    'gpu_use_dp': False
}

validation_results = []

print(f"STARTING BACKTEST EVALUATION (VALIDATION: DAYS 1886-1913)")

for store_id in tqdm(STORES):
    print(f"\nEvaluating {store_id}...")

    # 1. LOAD DATA
    cols = [
        'id', 'item_id', 'dept_id', 'cat_id', 'store_id',
        'd', 'sales', 'sell_price',
        'event_name_1', 'snap_' + store_id[:2], 'event_type_1'
    ]
    df = pd.read_parquet(FILE_PATH, filters=[('store_id', '==', store_id)], columns=cols)
    df['day_num'] = df['d'].astype(str).str.extract(r'(\d+)').astype(int)

    # 2. DEFINE SPLIT (HOLD OUT LAST 28 DAYS)
    # Train: 1 - 1885
    # Valid: 1886 - 1913
    SPLIT_DAY = 1913 - 28  # 1885

    # 3. PHASE 1: SARIMAX (ON TRAIN SET ONLY)
    print("   -> Training SARIMAX (Train Set Only)...")
    train_agg = df[df['day_num'] <= SPLIT_DAY].groupby('day_num').agg({
        'sales': 'sum',
        f'snap_{store_id[:2]}': 'first',
        'event_name_1': 'first'
    }).reset_index()

    train_agg['is_event'] = train_agg['event_name_1'].fillna('NoEvent').apply(lambda x: 0 if x == 'NoEvent' else 1)

    model_sarima = SARIMAX(
        np.log1p(train_agg['sales']),
        exog=train_agg[[f'snap_{store_id[:2]}', 'is_event']],
        order=(1, 0, 1),
        seasonal_order=(1, 0, 1, 7),
        enforce_stationarity=False,
        enforce_invertibility=False
    )
    res_sarima = model_sarima.fit(disp=False, method='lbfgs', maxiter=50)

    # Forecast the Validation Period (1886-1913)
    # We use the ACTUAL exog features from the valid set to test the model fairly
    valid_exog_agg = df[(df['day_num'] > SPLIT_DAY) & (df['day_num'] <= 1913)].groupby('day_num').agg({
        f'snap_{store_id[:2]}': 'first',
        'event_name_1': 'first'
    }).reset_index()
    valid_exog_agg['is_event'] = valid_exog_agg['event_name_1'].fillna('NoEvent').apply(lambda x: 0 if x == 'NoEvent' else 1)

    pred_log = res_sarima.get_forecast(steps=len(valid_exog_agg), exog=valid_exog_agg[[f'snap_{store_id[:2]}', 'is_event']])
    pred_real = np.expm1(pred_log.predicted_mean)

    # Create Trend Features
    trend_lookup = pd.DataFrame({
        'day_num': valid_exog_agg['day_num'].values,
        'store_trend': pred_real.values
    })
    history_trend = train_agg[['day_num', 'sales']].rename(columns={'sales': 'store_trend'})
    full_trend = pd.concat([history_trend, trend_lookup])

    del train_agg, model_sarima, res_sarima
    gc.collect()

    # 4. PHASE 2: LIGHTGBM FEATURES
    print("   -> Generating Features...")
    df = df.merge(full_trend, on='day_num', how='left')

    dt_col = pd.to_datetime(df['day_num'] - 1, unit='D', origin='2011-01-29')
    df['wday'] = dt_col.dt.weekday.astype('int8')
    df['month'] = dt_col.dt.month.astype('int8')
    df['day'] = dt_col.dt.day.astype('int8')

    cat_cols = ['dept_id', 'cat_id', 'event_name_1', 'event_type_1'] # No item_id
    for col in cat_cols:
        df[col] = df[col].astype('category')

    df = df.sort_values(['id', 'day_num'])
    df['lag_28'] = df.groupby('id')['sales'].shift(28)
    df['lag_35'] = df.groupby('id')['sales'].shift(35)
    df['rolling_mean_28_7'] = df.groupby('id')['lag_28'].transform(lambda x: x.rolling(7).mean())
    df['rolling_std_28_7'] = df.groupby('id')['lag_28'].transform(lambda x: x.rolling(7).std())

    df = df.dropna(subset=['rolling_mean_28_7'])

    # 5. TRAIN & VALIDATE
    print("   -> Training LightGBM & Scoring...")

    features = [
        'store_trend',
        'lag_28', 'lag_35',
        'rolling_mean_28_7', 'rolling_std_28_7',
        'sell_price', f'snap_{store_id[:2]}',
        'dept_id', 'cat_id', 'event_name_1', # item_id removed
        'wday', 'month', 'day'
    ]

    # Split
    train_mask = df['day_num'] <= SPLIT_DAY
    valid_mask = (df['day_num'] > SPLIT_DAY) & (df['day_num'] <= 1913)

    X_train = df.loc[train_mask, features]
    y_train = df.loc[train_mask, 'sales']
    X_valid = df.loc[valid_mask, features]
    y_valid = df.loc[valid_mask, 'sales']

    dtrain = lgb.Dataset(X_train, label=y_train)
    dvalid = lgb.Dataset(X_valid, label=y_valid, reference=dtrain)

    # Train
    model = lgb.train(
        PARAMS,
        dtrain,
        num_boost_round=1000, # Allow early stopping to find optimal
        valid_sets=[dtrain, dvalid],
        callbacks=[lgb.early_stopping(stopping_rounds=30), lgb.log_evaluation(0)] # 0 = Silent
    )

    # Score
    preds = model.predict(X_valid)
    rmse = np.sqrt(mean_squared_error(y_valid, preds))

    print(f"   => {store_id} RMSE: {rmse:.4f}")

    validation_results.append({
        'Store': store_id,
        'RMSE': rmse,
        'Best_Iter': model.best_iteration
    })

    del df, dtrain, dvalid, model, X_train, X_valid
    gc.collect()

# --- REPORT
print("\n=== FINAL VALIDATION REPORT ===")
res_df = pd.DataFrame(validation_results)
print(res_df)
print(f"\nAverage RMSE: {res_df['RMSE'].mean():.4f}")