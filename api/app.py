"""
FastAPI Application for M5 Sales Forecasting
Serves LightGBM and Sklearn GBM models via REST API
"""
from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from contextlib import asynccontextmanager
from typing import List

from schemas import (
    PredictionRequest,
    BatchPredictionRequest,
    PredictionResponse,
    BatchPredictionResponse,
    HealthResponse,
    ModelInfo
)
from models import model_manager


# API Version
VERSION = "1.0.0"


@asynccontextmanager
async def lifespan(app: FastAPI):
    """Load models on startup."""
    print("🚀 Starting M5 Sales Forecasting API...")
    load_status = model_manager.load_models()
    print(f"📦 Model load status: {load_status}")
    yield
    print("👋 Shutting down API...")


# Initialize FastAPI app
app = FastAPI(
    title="M5 Sales Forecasting API",
    description="REST API for predicting Walmart store sales using LightGBM and Sklearn GBM models",
    version=VERSION,
    lifespan=lifespan
)

# CORS middleware for React frontend
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],  # In production, specify exact origins
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)


# ============== Endpoints ==============

@app.get("/", tags=["Info"])
async def root():
    """Root endpoint with API information."""
    return {
        "name": "M5 Sales Forecasting API",
        "version": VERSION,
        "description": "Predict daily sales for Walmart products",
        "docs": "/docs",
        "health": "/health"
    }


@app.get("/health", response_model=HealthResponse, tags=["Info"])
async def health_check():
    """Health check endpoint."""
    # Check both single-file and store-specific models
    all_model_names = list(model_manager.single_model_paths.keys()) + list(model_manager.store_model_paths.keys())
    models_status = {
        name: model_manager.is_loaded(name) 
        for name in all_model_names
    }
    
    return HealthResponse(
        status="healthy" if any(models_status.values()) else "degraded",
        models_loaded=models_status,
        version=VERSION
    )


@app.get("/models", response_model=List[ModelInfo], tags=["Models"])
async def list_models():
    """List available models and their status."""
    return model_manager.get_model_info()


@app.post("/predict", response_model=PredictionResponse, tags=["Predictions"])
async def predict(request: PredictionRequest):
    """
    Make a single sales prediction.
    
    Provide input features including store info, time features, 
    price data, and historical lag features.
    """
    model_type = request.model_type.value
    
    if not model_manager.is_loaded(model_type):
        raise HTTPException(
            status_code=503,
            detail=f"Model '{model_type}' is not available"
        )
    
    try:
        # Convert Pydantic model to dict
        input_data = request.data.model_dump()
        
        # Make prediction
        prediction = model_manager.predict(model_type, input_data)
        
        return PredictionResponse(
            success=True,
            model_used=model_type,
            predicted_sales=round(prediction, 2),
            message="Prediction successful"
        )
    
    except Exception as e:
        raise HTTPException(
            status_code=500,
            detail=f"Prediction failed: {str(e)}"
        )


@app.post("/predict/batch", response_model=BatchPredictionResponse, tags=["Predictions"])
async def predict_batch(request: BatchPredictionRequest):
    """
    Make batch predictions for multiple items.
    
    Submit an array of input features to get predictions for all items at once.
    """
    model_type = request.model_type.value
    
    if not model_manager.is_loaded(model_type):
        raise HTTPException(
            status_code=503,
            detail=f"Model '{model_type}' is not available"
        )
    
    if len(request.data) == 0:
        raise HTTPException(
            status_code=400,
            detail="No data provided for batch prediction"
        )
    
    if len(request.data) > 1000:
        raise HTTPException(
            status_code=400,
            detail="Batch size exceeds maximum of 1000 items"
        )
    
    try:
        # Convert all inputs to dicts
        input_data = [item.model_dump() for item in request.data]
        
        # Make batch prediction
        predictions = model_manager.predict_batch(model_type, input_data)
        predictions = [round(p, 2) for p in predictions]
        
        return BatchPredictionResponse(
            success=True,
            model_used=model_type,
            predictions=predictions,
            count=len(predictions),
            message=f"Successfully predicted {len(predictions)} items"
        )
    
    except Exception as e:
        raise HTTPException(
            status_code=500,
            detail=f"Batch prediction failed: {str(e)}"
        )


# Run with: uvicorn app:app --reload --port 8000
if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=8000)
