# M5 Sales Forecasting API

REST API for serving M5 Sales Forecasting models (LightGBM and Sklearn GBM).

## Quick Start

```bash
# Install dependencies
pip install -r requirements.txt

# Start the server
uvicorn app:app --reload --port 8000
```

Visit `http://localhost:8000/docs` for interactive API documentation.

## Endpoints

| Method | Endpoint | Description |
|--------|----------|-------------|
| GET | `/` | API info |
| GET | `/health` | Health check |
| GET | `/models` | List available models |
| POST | `/predict` | Single prediction |
| POST | `/predict/batch` | Batch predictions |

## Example Request

```bash
curl -X POST http://localhost:8000/predict \
  -H "Content-Type: application/json" \
  -d '{
    "model_type": "lightgbm",
    "data": {
      "store_id": "CA_1",
      "dept_id": "FOODS_1",
      "cat_id": "FOODS",
      "state_id": "CA",
      "wday": 1,
      "month": 3,
      "year": 2016,
      "sell_price": 2.99,
      "snap_CA": 1,
      "snap_TX": 0,
      "snap_WI": 0,
      "lag_28": 5.0,
      "lag_35": 4.0,
      "lag_42": 6.0,
      "lag_49": 3.0,
      "roll_mean_28": 4.5,
      "roll_std_28": 1.2,
      "price_momentum": 0.95,
      "price_roll_std_7": 0.1
    }
  }'
```

## Docker

```bash
docker build -t m5-api .
docker run -p 8000:8000 m5-api
```
