# ARIMA Sales Forecasting Web Application

A modern web application for sales forecasting using the Azure ML Batch Endpoint.

![Dashboard Preview](frontend/preview.png)

## 🚀 Features

- **Modern Dashboard**: Dark-themed UI with glassmorphism design
- **Azure ML Integration**: Connects to your deployed ARIMA batch endpoint
- **Real-time Status**: Job progress tracking with visual timeline
- **Results Visualization**: Clean table display for predictions

## 📁 Project Structure

```
arima Model/
├── frontend/
│   ├── index.html    # Main dashboard
│   ├── styles.css    # Modern styling
│   └── app.js        # Application logic
├── backend/
│   ├── server.py     # Flask API proxy
│   ├── requirements.txt
│   └── .env.example  # Environment template
├── BACKEND_HANDOFF.md
└── README.md
```

## 🛠️ Setup

### 1. Backend Setup

```bash
cd backend

# Create virtual environment
python -m venv venv
venv\Scripts\activate  # Windows

# Install dependencies
pip install -r requirements.txt

# Configure Azure credentials
copy .env.example .env
# Edit .env with your Azure details
```

### 2. Azure Authentication

The backend uses `DefaultAzureCredential` which supports multiple auth methods:

**Option A: Azure CLI (Development)**
```bash
az login
```

**Option B: Service Principal (Production)**
Set these in your `.env` file:
```
AZURE_CLIENT_ID=your_client_id
AZURE_CLIENT_SECRET=your_client_secret
AZURE_TENANT_ID=your_tenant_id
```

### 3. Run the Application

**Terminal 1 - Start Backend:**
```bash
cd backend
python server.py
```

**Terminal 2 - Serve Frontend:**
```bash
cd frontend
# Use any static server, e.g.:
python -m http.server 8080
```

Open `http://localhost:8080` in your browser.

## 📡 API Endpoints

| Endpoint | Method | Description |
|----------|--------|-------------|
| `/api/health` | GET | Health check |
| `/api/submit-job` | POST | Submit forecast job |
| `/api/job-status/<job_id>` | GET | Get job status |
| `/api/results/<job_id>` | GET | Get prediction results |
| `/api/stores` | GET | Get available stores |

## 📝 Usage

1. Select a store from the dropdown
2. Enter 13 feature values (comma-separated)
3. Click "Submit Forecast Job"
4. Watch the status timeline as the job processes
5. View results when complete

## ⚠️ Requirements

- Python 3.8+
- Azure subscription with ML workspace
- `Azure Machine Learning Data Scientist` role
- Access to the deployed batch endpoint

## 📄 License

MIT
