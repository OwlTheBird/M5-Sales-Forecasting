"""
Pydantic schemas for API request/response models.
M5 Sales Forecasting API
"""
from pydantic import BaseModel, Field
from typing import List, Optional
from enum import Enum


class ModelType(str, Enum):
    """Available model types for prediction."""
    LIGHTGBM = "lightgbm"
    GBTREGRESSOR = "gbtregressor"
    HYBRID = "hybrid"


class PredictionInput(BaseModel):
    """Input features for a single prediction."""
    
    # Store/Item identifiers
    store_id: str = Field(..., example="CA_1", description="Store identifier")
    dept_id: str = Field(..., example="FOODS_1", description="Department ID")
    cat_id: str = Field(..., example="FOODS", description="Category ID")
    state_id: str = Field(..., example="CA", description="State ID")
    
    # Time features
    wday: int = Field(..., ge=1, le=7, example=1, description="Day of week (1-7)")
    month: int = Field(..., ge=1, le=12, example=1, description="Month (1-12)")
    year: int = Field(..., ge=2011, le=2030, example=2016, description="Year")
    
    # Price features
    sell_price: float = Field(..., gt=0, example=2.99, description="Selling price")
    
    # Event features (optional)
    event_name_1: Optional[str] = Field(None, example="SuperBowl", description="Event name")
    event_type_1: Optional[str] = Field(None, example="Sporting", description="Event type")
    
    # SNAP features
    snap_CA: int = Field(0, ge=0, le=1, example=0, description="SNAP eligible in CA")
    snap_TX: int = Field(0, ge=0, le=1, example=0, description="SNAP eligible in TX")
    snap_WI: int = Field(0, ge=0, le=1, example=0, description="SNAP eligible in WI")
    
    # Lag features (historical sales)
    lag_28: float = Field(0.0, example=5.0, description="Sales 28 days ago")
    lag_35: float = Field(0.0, example=4.0, description="Sales 35 days ago")
    lag_42: float = Field(0.0, example=6.0, description="Sales 42 days ago")
    lag_49: float = Field(0.0, example=3.0, description="Sales 49 days ago")
    
    # Rolling statistics
    roll_mean_28: float = Field(0.0, example=4.5, description="28-day rolling mean")
    roll_std_28: float = Field(0.0, example=1.2, description="28-day rolling std")
    
    # Price momentum features
    price_max: float = Field(3.99, gt=0, example=3.99, description="Maximum historical price")
    price_momentum: float = Field(1.0, ge=0, le=1, example=0.95, description="Price ratio to max")
    price_roll_std_7: float = Field(0.0, example=0.1, description="7-day price volatility")

    class Config:
        json_schema_extra = {
            "example": {
                "store_id": "CA_1",
                "dept_id": "FOODS_1",
                "cat_id": "FOODS",
                "state_id": "CA",
                "wday": 1,
                "month": 3,
                "year": 2016,
                "sell_price": 2.99,
                "event_name_1": None,
                "event_type_1": None,
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
        }


class PredictionRequest(BaseModel):
    """Request body for prediction endpoint."""
    model_type: ModelType = Field(ModelType.LIGHTGBM, description="Model to use")
    data: PredictionInput


class BatchPredictionRequest(BaseModel):
    """Request body for batch prediction endpoint."""
    model_type: ModelType = Field(ModelType.LIGHTGBM, description="Model to use")
    data: List[PredictionInput]


class PredictionResponse(BaseModel):
    """Response from prediction endpoint."""
    success: bool
    model_used: str
    predicted_sales: float
    message: Optional[str] = None


class BatchPredictionResponse(BaseModel):
    """Response from batch prediction endpoint."""
    success: bool
    model_used: str
    predictions: List[float]
    count: int
    message: Optional[str] = None


class HealthResponse(BaseModel):
    """Health check response."""
    status: str
    models_loaded: dict
    version: str


class ModelInfo(BaseModel):
    """Information about available models."""
    name: str
    type: str
    file_path: str
    loaded: bool
