import pandas as pd
import numpy as np
import lightgbm as lgb
import gc
import onnxmltools
from onnxmltools.convert.common.data_types import FloatTensorType
from tqdm import tqdm
from statsmodels.tsa.statespace.sarimax import SARIMAX

# --- CONFIG ---
STORES = ['CA_1', 'CA_2', 'CA_3', 'TX_1', 'TX_2', 'TX_3', 'WI_1', 'WI_2', 'WI_3']
FILE_PATH = '/content/drive/MyDrive/final_optimized.parquet'
SUBMISSION_PATH = 'submission.csv'

# GPU Parameters
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

all_preds = []

print(f"--- STARTING PRODUCTION LOOP FOR {len(STORES)} STORES ---")

for store_id in tqdm(STORES):
    print(f"\nProcessing {store_id}...")

    # ============================================
    # 1. LOAD DATA
    # ============================================
    cols = [
        'id', 'item_id', 'dept_id', 'cat_id', 'store_id',
        'd', 'sales', 'sell_price',
        'event_name_1', 'snap_' + store_id[:2], 'event_type_1'
    ]

    df = pd.read_parquet(FILE_PATH, filters=[('store_id', '==', store_id)], columns=cols)
    df['day_num'] = df['d'].astype(str).str.extract(r'(\d+)').astype(int)

    # ============================================
    # 2. EXTEND DATA FOR FUTURE (DAYS 1914-1941)
    # ============================================
    max_day = df['day_num'].max()

    if max_day < 1941:
        print("   -> Extending dataframe to Day 1941...")
        future_days = np.arange(1914, 1942)
        unique_items = df['id'].unique()

        future_df = pd.DataFrame(
            index=pd.MultiIndex.from_product([unique_items, future_days], names=['id', 'day_num'])
        ).reset_index()

        last_day_info = df[df['day_num'] == 1913][['id', 'item_id', 'dept_id', 'cat_id', 'store_id', 'sell_price', 'snap_' + store_id[:2]]].drop_duplicates(subset=['id'])
        future_df = future_df.merge(last_day_info, on='id', how='left')

        future_df['sales'] = np.nan
        future_df['d'] = 'd_' + future_df['day_num'].astype(str)
        future_df['event_name_1'] = 'NoEvent'
        future_df['event_type_1'] = 'NoEvent'

        df = pd.concat([df, future_df], axis=0, ignore_index=True)
        del future_df, last_day_info
        gc.collect()

    # ============================================
    # 3. PHASE 1: SARIMAX (MACRO TREND)
    # ============================================
    print("   -> Training SARIMAX...")

    train_agg = df[df['day_num'] <= 1913].groupby('day_num').agg({
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

    future_exog = pd.DataFrame({
        f'snap_{store_id[:2]}': [train_agg[f'snap_{store_id[:2]}'].iloc[-1]] * 28,
        'is_event': [0] * 28
    })

    pred_log = res_sarima.get_forecast(steps=28, exog=future_exog)
    pred_real = np.expm1(pred_log.predicted_mean)

    store_trend_lookup = pd.DataFrame({
        'day_num': np.arange(1914, 1942),
        'store_trend': pred_real.values
    })
    history_trend = train_agg[['day_num', 'sales']].rename(columns={'sales': 'store_trend'})
    full_trend = pd.concat([history_trend, store_trend_lookup])

    del train_agg, model_sarima, res_sarima
    gc.collect()

    # ============================================
    # 4. PHASE 2: LIGHTGBM FEATURE ENG
    # ============================================
    print("   -> Generating Features...")

    df = df.merge(full_trend, on='day_num', how='left')

    dt_col = pd.to_datetime(df['day_num'] - 1, unit='D', origin='2011-01-29')
    df['wday'] = dt_col.dt.weekday.astype('int8')
    df['month'] = dt_col.dt.month.astype('int8')
    df['day'] = dt_col.dt.day.astype('int8')

    cat_cols = ['item_id', 'dept_id', 'cat_id', 'event_name_1', 'event_type_1']
    for col in cat_cols:
        df[col] = df[col].astype('category')

    df = df.sort_values(['id', 'day_num'])
    df['lag_28'] = df.groupby('id')['sales'].shift(28)
    df['lag_35'] = df.groupby('id')['sales'].shift(35)
    df['rolling_mean_28_7'] = df.groupby('id')['lag_28'].transform(lambda x: x.rolling(7).mean())
    df['rolling_std_28_7'] = df.groupby('id')['lag_28'].transform(lambda x: x.rolling(7).std())

    df = df.dropna(subset=['rolling_mean_28_7'])

    # ============================================
    # 5. TRAIN & PREDICT
    # ============================================
    print("   -> Training LightGBM (GPU)...")

    features = [
        'store_trend',
        'lag_28', 'lag_35',
        'rolling_mean_28_7', 'rolling_std_28_7',
        'sell_price', f'snap_{store_id[:2]}',
        'dept_id', 'cat_id', 'event_name_1',
        'wday', 'month', 'day'
    ]

    train_mask = df['day_num'] <= 1913
    pred_mask = df['day_num'] >= 1914

    dtrain = lgb.Dataset(df.loc[train_mask, features], label=df.loc[train_mask, 'sales'])

    # Training
    model = lgb.train(PARAMS, dtrain, num_boost_round=90)

    # --- ONNX EXPORT ---
    print(f"   -> Exporting {store_id} to ONNX...")
    initial_types = [('input', FloatTensorType([None, len(features)]))]
    try:
        onnx_model = onnxmltools.convert_lightgbm(model, initial_types=initial_types)
        onnxmltools.utils.save_model(onnx_model, f'model_{store_id}.onnx')
    except Exception as e:
        print(f"ONNX Export Warning for {store_id}: {e}")

    # Predict Future
    preds = model.predict(df.loc[pred_mask, features])

    # Format
    submit_df = df.loc[pred_mask, ['id', 'day_num']].copy()
    submit_df['sales'] = preds
    submit_df['F'] = 'F' + (submit_df['day_num'] - 1913).astype(str)

    submit_df = submit_df.pivot(index='id', columns='F', values='sales').reset_index()
    all_preds.append(submit_df)

    del df, dtrain, model, submit_df, full_trend
    if 'onnx_model' in locals(): del onnx_model
    gc.collect()

## 6. SAVE
print("\nSaving Submission...")
final_submission = pd.concat(all_preds)
final_submission.to_csv(SUBMISSION_PATH, index=False)
print("Submission Ready & 10 ONNX Models Saved!")