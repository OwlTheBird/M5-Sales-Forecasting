"""
ARIMA Sales Forecast - Flask Backend Proxy
Handles Azure ML Batch Endpoint authentication and API calls
"""

from flask import Flask, request, jsonify
from flask_cors import CORS
from azure.identity import DefaultAzureCredential
import requests
import os
from dotenv import load_dotenv
from datetime import datetime
import json

# Load environment variables
load_dotenv()

app = Flask(__name__)
CORS(app)  # Enable CORS for frontend

# Azure ML Configuration
AZURE_ML_ENDPOINT = os.getenv(
    'AZURE_ML_ENDPOINT',
    'https://sales-forecast-endpoint-v2.germanywestcentral.inference.ml.azure.com/jobs'
)
BLOB_STORAGE_ACCOUNT = os.getenv('BLOB_STORAGE_ACCOUNT', '')
BLOB_CONTAINER = os.getenv('BLOB_CONTAINER', 'data')

# Token cache
_credential = None

def get_credential():
    """Get or create Azure credential (auto-refreshes tokens)"""
    global _credential
    if _credential is None:
        _credential = DefaultAzureCredential()
    return _credential

def get_azure_token():
    """Generate Azure ML token (valid for 1 hour, auto-refreshes)"""
    credential = get_credential()
    token = credential.get_token("https://ml.azure.com/.default").token
    return token

def get_headers():
    """Get authorization headers for Azure ML API"""
    token = get_azure_token()
    return {
        "Authorization": f"Bearer {token}",
        "Content-Type": "application/json"
    }

# Store job metadata (in production, use a database)
job_store = {}

@app.route('/api/health', methods=['GET'])
def health_check():
    """Health check endpoint"""
    return jsonify({
        'status': 'healthy',
        'timestamp': datetime.utcnow().isoformat(),
        'azure_endpoint': AZURE_ML_ENDPOINT
    })

@app.route('/api/submit-job', methods=['POST'])
def submit_job():
    """Submit a forecast job to Azure ML Batch Endpoint"""
    try:
        data = request.get_json()
        
        # Validate input
        store_id = data.get('store_id')
        features = data.get('features')
        blob_uri = data.get('blob_uri')
        
        if not store_id:
            return jsonify({'error': 'store_id is required'}), 400
        
        if not features or len(features) != 13:
            return jsonify({'error': 'features must contain exactly 13 values'}), 400
        
        # Prepare input data
        # If blob_uri provided, use it; otherwise create temp data
        if blob_uri:
            input_uri = blob_uri
        else:
            # For demo: Use a default blob URI
            # In production: Upload data to blob storage and get URI
            input_uri = f"https://{BLOB_STORAGE_ACCOUNT}.blob.core.windows.net/{BLOB_CONTAINER}/input"
        
        # Prepare payload for Azure ML
        payload = {
            "input_data": {
                "uri": input_uri
            }
        }
        
        # Send request to Azure ML
        headers = get_headers()
        response = requests.post(
            AZURE_ML_ENDPOINT,
            headers=headers,
            json=payload,
            timeout=30
        )
        
        if response.status_code not in [200, 201, 202]:
            error_detail = response.text
            return jsonify({
                'error': f'Azure ML returned status {response.status_code}',
                'detail': error_detail
            }), response.status_code
        
        result = response.json()
        job_id = result.get('id') or result.get('name')
        
        # Store job metadata
        job_store[job_id] = {
            'store_id': store_id,
            'features': features,
            'submitted_at': datetime.utcnow().isoformat(),
            'status': 'Pending'
        }
        
        return jsonify({
            'job_id': job_id,
            'status': 'Pending',
            'message': 'Job submitted successfully'
        })
        
    except Exception as e:
        return jsonify({'error': str(e)}), 500

@app.route('/api/job-status/<job_id>', methods=['GET'])
def get_job_status(job_id):
    """Get the status of a submitted job"""
    try:
        # Query Azure ML for job status
        headers = get_headers()
        status_url = f"{AZURE_ML_ENDPOINT}/{job_id}"
        
        response = requests.get(
            status_url,
            headers=headers,
            timeout=30
        )
        
        if response.status_code == 404:
            return jsonify({'error': 'Job not found'}), 404
        
        if response.status_code != 200:
            return jsonify({
                'error': f'Azure ML returned status {response.status_code}'
            }), response.status_code
        
        result = response.json()
        status = result.get('status', result.get('properties', {}).get('status', 'Unknown'))
        
        # Update local store
        if job_id in job_store:
            job_store[job_id]['status'] = status
        
        return jsonify({
            'job_id': job_id,
            'status': status,
            'details': result
        })
        
    except Exception as e:
        return jsonify({'error': str(e)}), 500

@app.route('/api/results/<job_id>', methods=['GET'])
def get_results(job_id):
    """Get prediction results for a completed job"""
    try:
        # First check job status
        headers = get_headers()
        status_url = f"{AZURE_ML_ENDPOINT}/{job_id}"
        
        response = requests.get(
            status_url,
            headers=headers,
            timeout=30
        )
        
        if response.status_code != 200:
            return jsonify({'error': 'Could not fetch job details'}), response.status_code
        
        result = response.json()
        status = result.get('status', result.get('properties', {}).get('status', 'Unknown'))
        
        if status.lower() != 'completed':
            return jsonify({
                'error': f'Job is not completed. Current status: {status}'
            }), 400
        
        # Get output URI from job result
        output_uri = result.get('output_data_uri') or result.get('properties', {}).get('output_data_uri')
        
        # For demo: return mock predictions
        # In production: Download and parse predictions.csv from blob storage
        job_data = job_store.get(job_id, {})
        
        predictions = [
            {
                'store_id': job_data.get('store_id', 'Unknown'),
                'prediction': 2.33394885,  # Mock value
                'confidence': 0.85
            }
        ]
        
        return jsonify({
            'job_id': job_id,
            'status': 'Completed',
            'predictions': predictions,
            'output_uri': output_uri
        })
        
    except Exception as e:
        return jsonify({'error': str(e)}), 500

@app.route('/api/stores', methods=['GET'])
def get_stores():
    """Get list of available stores"""
    stores = [
        {'id': 'CA_1', 'name': 'California Store 1', 'region': 'California'},
        {'id': 'CA_2', 'name': 'California Store 2', 'region': 'California'},
        {'id': 'CA_3', 'name': 'California Store 3', 'region': 'California'},
        {'id': 'TX_1', 'name': 'Texas Store 1', 'region': 'Texas'},
        {'id': 'TX_2', 'name': 'Texas Store 2', 'region': 'Texas'},
        {'id': 'TX_3', 'name': 'Texas Store 3', 'region': 'Texas'},
        {'id': 'WI_1', 'name': 'Wisconsin Store 1', 'region': 'Wisconsin'},
        {'id': 'WI_2', 'name': 'Wisconsin Store 2', 'region': 'Wisconsin'},
        {'id': 'WI_3', 'name': 'Wisconsin Store 3', 'region': 'Wisconsin'},
    ]
    return jsonify({'stores': stores})

if __name__ == '__main__':
    print("\n" + "="*50)
    print("🚀 ARIMA Sales Forecast Backend Server")
    print("="*50)
    print(f"📍 API Base URL: http://localhost:5000/api")
    print(f"☁️  Azure Endpoint: {AZURE_ML_ENDPOINT}")
    print("="*50 + "\n")
    
    app.run(debug=True, host='0.0.0.0', port=5000)
