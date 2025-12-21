# Azure Batch Endpoint Integration Guide

## Overview
We have deployed a **Batch Endpoint** for Sales Forecasting.
*   **Architecture:** Asynchronous Batch Processing.
*   **Cost Model:** Zero-cost idle (Scale-to-Zero). Cluster wakes up on demand.
*   **Latency:** Cold start ~3-5 mins. Optimized for large volume, not real-time.

## 1. Endpoint Details
*   **Name:** `sales-forecast-endpoint-v2`
*   **Scoring URI:** `https://sales-forecast-endpoint-v2.germanywestcentral.inference.ml.azure.com/jobs`
*   **Region:** Germany West Central

## 2. Authentication (AAD)
**CRITICAL:** Do NOT use a static API Key.
You must generate an **Azure Active Directory (AAD) Token** programmatically. Tokens expire in **1 hour**.

### Python Code Example
```python
from azure.identity import DefaultAzureCredential
import requests

# 1. Get Token (Auto-refreshes, valid for 1 hour)
# Ensure your environment is logged in or has Service Principal env vars set
credential = DefaultAzureCredential()
token = credential.get_token("https://ml.azure.com/.default").token

# 2. Prepare Headers
headers = {
    "Authorization": f"Bearer {token}",
    "Content-Type": "application/json"
}
```

## 3. Workflow
The process is **Asynchronous**:

1.  **Prep Data:** Save your input CSV/JSON files to a Blob Storage container accessible by the workspace.
2.  **Trigger:** Send a POST request to the Scoring URI with the blob URL.
3.  **Poll:** You get a Job ID. Poll it until `status` is "Completed".
4.  **Download:** The result `predictions.csv` is written to the default Blob Store in `artifacts/output/`.

## 4. Request Payload
**Method:** `POST`
**URL:** `https://sales-forecast-endpoint-v2.germanywestcentral.inference.ml.azure.com/jobs`

**Body:**
```json
{
    "input_data": {
        "uri": "https://<your_storage_account>.blob.core.windows.net/<container_name>/<folder_path>"
    }
}
```
*Note: The URI must point to the folder or file containing the input data.*

## 5. Input Data Format
Files can be JSON or CSV.
**Required Columns:**
*   `store_id` (e.g., "CA_1", "TX_2")
*   `features` (Numeric list/array matching model input size)

**Example (JSON):**
```json
[
  {
    "store_id": "CA_1",
    "features": [0.5, 0.5, 0.5, 0.5, 0.5, 0.5, 0.5, 0.5, 0.5, 0.5, 0.5, 0.5, 0.5]
  },
  {
    "store_id": "TX_1",
    "features": [0.2, 0.1, 0.1, 0.1, 0.1, 0.1, 0.1, 0.1, 0.1, 0.1, 0.1, 0.1, 0.1]
  }
]
```

## 6. Output Format
Azure aggregates all processed files into one `predictions.csv`.

**Columns:**
*   `file_name`: Source file name.
*   `store_id`: The store the prediction belongs to.
*   `prediction`: The forecasted sales value.

**Example Content:**
```csv
test_data.json CA_1 2.33394885
test_data.json TX_1 0.97295016
```

---
**Permissions Note:** The identity calling the API (Service Principal or User) needs the `Azure Machine Learning Data Scientist` or `Contributor` role on the `SARIMAX_LightGBM` resource group to trigger jobs.
