"""
Model loading and inference utilities.
M5 Sales Forecasting API
"""
import os
import numpy as np
import onnxruntime as ort
from typing import Dict, List, Optional, Any, Union
from pathlib import Path


# Paths
BASE_DIR = Path(__file__).parent.parent
MODELS_DIR = BASE_DIR / "models"

# Store IDs for store-specific models
STORE_IDS = ['CA_1', 'CA_2', 'CA_3', 'CA_4', 'TX_1', 'TX_2', 'TX_3', 'WI_1', 'WI_2', 'WI_3']

# Feature configuration matching training
CATEGORICAL_FEATURES = ['store_id', 'dept_id', 'cat_id', 'state_id', 'event_name_1', 'event_type_1']

# Category mappings (from training data) - Used for LightGBM ONNX
CATEGORY_MAPPINGS = {
    'store_id': {'CA_1': 0, 'CA_2': 1, 'CA_3': 2, 'CA_4': 3, 'TX_1': 4, 'TX_2': 5, 'TX_3': 6, 'WI_1': 7, 'WI_2': 8, 'WI_3': 9},
    'dept_id': {'FOODS_1': 0, 'FOODS_2': 1, 'FOODS_3': 2, 'HOBBIES_1': 3, 'HOBBIES_2': 4, 'HOUSEHOLD_1': 5, 'HOUSEHOLD_2': 6},
    'cat_id': {'FOODS': 0, 'HOBBIES': 1, 'HOUSEHOLD': 2},
    'state_id': {'CA': 0, 'TX': 1, 'WI': 2},
    'event_name_1': {None: -1, '': -1, 'SuperBowl': 0, 'ValentinesDay': 1, 'PresidentsDay': 2, 'LentStart': 3, 
                     'LentWeek2': 4, 'StPatricksDay': 5, 'Purim End': 6, 'OrthodoxEaster': 7, 'Pesach End': 8,
                     'Cinco De Mayo': 9, 'Mother\'s day': 10, 'MemorialDay': 11, 'NBAFinalsStart': 12,
                     'NBAFinalsEnd': 13, 'Father\'s day': 14, 'IndependenceDay': 15, 'Ramadan starts': 16,
                     'Eid al-Fitr': 17, 'LaborDay': 18, 'ColumbusDay': 19, 'Halloween': 20, 'EidAlAdha': 21,
                     'VeteransDay': 22, 'Thanksgiving': 23, 'Christmas': 24, 'Chanukah End': 25, 'NewYear': 26,
                     'OrthodoxChristmas': 27, 'MartinLutherKingDay': 28, 'Easter': 29},
    'event_type_1': {None: -1, '': -1, 'Sporting': 0, 'Cultural': 1, 'National': 2, 'Religious': 3}
}

# Feature order for LightGBM model input (must match training)
# Total: 22 features
FEATURE_ORDER_LIGHTGBM = [
    'store_id', 'dept_id', 'cat_id', 'state_id',
    'wday', 'month', 'year',
    'sell_price',
    'event_name_1', 'event_type_1',
    'snap_CA', 'snap_TX', 'snap_WI',
    'lag_28', 'lag_35', 'lag_42', 'lag_49',
    'roll_mean_28', 'roll_std_28',
    'price_max', 'price_momentum', 'price_roll_std_7'
]

# Feature order for GBTRegressor model (12 features) - PySpark trained
# Based on similar feature engineering as LightGBM but fewer features
FEATURE_ORDER_GBT = [
    'sell_price', 'wday', 'month',
    'snap',  # Store-specific snap (will be mapped based on store_id)
    'lag_28', 'lag_35',
    'roll_mean_28', 'roll_std_28',
    'dept_id', 'cat_id', 'event_name_1', 'event_type_1'
]

# Feature order for Hybrid model (13 features) - SARIMAX + LightGBM
# From training script: store_trend, lag_28, lag_35, rolling_mean_28_7, rolling_std_28_7,
#                       sell_price, snap_{state}, dept_id, cat_id, event_name_1, wday, month, day
FEATURE_ORDER_HYBRID = [
    'store_trend',  # SARIMAX prediction - will use lag_28 as proxy for API
    'lag_28', 'lag_35',
    'rolling_mean_28_7', 'rolling_std_28_7',  # Using roll_mean_28/roll_std_28 as input
    'sell_price', 'snap',  # Store-specific snap
    'dept_id', 'cat_id', 'event_name_1',
    'wday', 'month', 'day'
]

# Sklearn ONNX model expects separate named inputs with specific types
# elem_type: 11 = float64, 6 = int32, 8 = string
SKLEARN_INPUT_TYPES = {
    'sell_price': 'float64',
    'wday': 'int32',
    'month': 'int32',
    'year': 'int32',
    'snap_CA': 'int32',
    'snap_TX': 'int32',
    'snap_WI': 'int32',
    'store_id': 'string',
    'dept_id': 'string',
    'cat_id': 'string',
    'state_id': 'string',
    'event_name_1': 'string',
    'event_type_1': 'string',
    'lag_28': 'float64',
    'lag_35': 'float64',
    'lag_42': 'float64',
    'lag_49': 'float64',
    'roll_mean_28': 'float64',
    'roll_std_28': 'float64',
    'price_max': 'float64',
    'price_momentum': 'float64',
    'price_roll_std_7': 'float64'
}


class ModelManager:
    """Manages loading and inference for ONNX models."""
    
    def __init__(self):
        self.models: Dict[str, ort.InferenceSession] = {}
        self.store_models: Dict[str, Dict[str, ort.InferenceSession]] = {}  # For store-specific models
        self.model_types: Dict[str, str] = {}  # Track model type
        
        # Single-file models
        self.single_model_paths = {
            'lightgbm': MODELS_DIR / 'LightGBM' / 'm5_lightgbm.onnx'
        }
        
        # Store-specific models (directory path + file pattern)
        self.store_model_paths = {
            'gbtregressor': (MODELS_DIR / 'GBTRegressor', 'gbt_{store_id}.onnx'),
            'hybrid': (MODELS_DIR / 'SARIMAX + LightGBM Hybird', 'model_{store_id}.onnx')
        }
    
    def load_models(self) -> Dict[str, bool]:
        """Load all available ONNX models."""
        status = {}
        
        # Load single-file models
        for name, path in self.single_model_paths.items():
            try:
                if path.exists():
                    sess_options = ort.SessionOptions()
                    sess_options.graph_optimization_level = ort.GraphOptimizationLevel.ORT_ENABLE_ALL
                    
                    self.models[name] = ort.InferenceSession(
                        str(path),
                        sess_options,
                        providers=['CPUExecutionProvider']
                    )
                    self.model_types[name] = 'single'
                    status[name] = True
                    print(f"✓ Loaded {name} model from {path}")
                else:
                    status[name] = False
                    print(f"✗ Model not found: {path}")
            except Exception as e:
                status[name] = False
                print(f"✗ Error loading {name}: {e}")
        
        # Load store-specific models
        for name, (dir_path, file_pattern) in self.store_model_paths.items():
            self.store_models[name] = {}
            loaded_count = 0
            
            for store_id in STORE_IDS:
                model_file = dir_path / file_pattern.format(store_id=store_id)
                try:
                    if model_file.exists():
                        sess_options = ort.SessionOptions()
                        sess_options.graph_optimization_level = ort.GraphOptimizationLevel.ORT_ENABLE_ALL
                        
                        self.store_models[name][store_id] = ort.InferenceSession(
                            str(model_file),
                            sess_options,
                            providers=['CPUExecutionProvider']
                        )
                        loaded_count += 1
                except Exception as e:
                    print(f"✗ Error loading {name}/{store_id}: {e}")
            
            if loaded_count > 0:
                self.model_types[name] = 'store_specific'
                status[name] = True
                print(f"✓ Loaded {name} models for {loaded_count}/{len(STORE_IDS)} stores")
            else:
                status[name] = False
                print(f"✗ No {name} models found in {dir_path}")
        
        return status
    
    def is_loaded(self, model_type: str) -> bool:
        """Check if a model is loaded."""
        return model_type in self.models or model_type in self.store_models
    
    def get_model_info(self) -> List[Dict[str, Any]]:
        """Get information about available models."""
        info = []
        # Single-file models
        for name, path in self.single_model_paths.items():
            info.append({
                'name': name,
                'type': 'ONNX',
                'file_path': str(path),
                'loaded': name in self.models
            })
        # Store-specific models
        for name, (dir_path, _) in self.store_model_paths.items():
            loaded_stores = len(self.store_models.get(name, {}))
            info.append({
                'name': name,
                'type': 'ONNX (store-specific)',
                'file_path': str(dir_path),
                'loaded': loaded_stores > 0,
                'stores_loaded': loaded_stores
            })
        return info
    
    def preprocess_input_lightgbm(self, data: Dict[str, Any]) -> np.ndarray:
        """Convert input data to LightGBM model-ready numpy array."""
        features = []
        
        for feat in FEATURE_ORDER_LIGHTGBM:
            value = data.get(feat)
            
            if feat in CATEGORICAL_FEATURES:
                # Encode categorical features
                mapping = CATEGORY_MAPPINGS.get(feat, {})
                encoded = mapping.get(value, -1)
                features.append(float(encoded))
            else:
                # Numeric features
                features.append(float(value) if value is not None else 0.0)
        
        return np.array([features], dtype=np.float32)
    
    def preprocess_input_gbt(self, data: Dict[str, Any]) -> np.ndarray:
        """Convert input data to GBTRegressor model-ready numpy array (12 features)."""
        store_id = data.get('store_id', 'CA_1')
        state = store_id[:2]  # Get state prefix (CA, TX, WI)
        
        features = []
        
        for feat in FEATURE_ORDER_GBT:
            if feat == 'snap':
                # Get store-specific snap value
                snap_key = f'snap_{state}'
                value = data.get(snap_key, 0)
                features.append(float(value))
            elif feat in CATEGORICAL_FEATURES:
                value = data.get(feat)
                mapping = CATEGORY_MAPPINGS.get(feat, {})
                encoded = mapping.get(value, -1)
                features.append(float(encoded))
            else:
                value = data.get(feat)
                features.append(float(value) if value is not None else 0.0)
        
        return np.array([features], dtype=np.float32)
    
    def preprocess_input_hybrid(self, data: Dict[str, Any]) -> np.ndarray:
        """Convert input data to Hybrid model-ready numpy array (13 features)."""
        store_id = data.get('store_id', 'CA_1')
        state = store_id[:2]  # Get state prefix (CA, TX, WI)
        
        features = []
        
        for feat in FEATURE_ORDER_HYBRID:
            if feat == 'store_trend':
                # Use lag_28 as a proxy for store_trend (SARIMAX output)
                # In production, this would come from a SARIMAX model
                value = data.get('lag_28', 0.0) * 1.1  # Simple adjustment
                features.append(float(value))
            elif feat == 'rolling_mean_28_7':
                # Map from input field name
                value = data.get('roll_mean_28', 0.0)
                features.append(float(value))
            elif feat == 'rolling_std_28_7':
                value = data.get('roll_std_28', 0.0)
                features.append(float(value))
            elif feat == 'snap':
                snap_key = f'snap_{state}'
                value = data.get(snap_key, 0)
                features.append(float(value))
            elif feat == 'day':
                # Extract day from date or use default
                value = data.get('day', 15)  # Default to mid-month
                features.append(float(value))
            elif feat in CATEGORICAL_FEATURES:
                value = data.get(feat)
                mapping = CATEGORY_MAPPINGS.get(feat, {})
                encoded = mapping.get(value, -1)
                features.append(float(encoded))
            else:
                value = data.get(feat)
                features.append(float(value) if value is not None else 0.0)
        
        return np.array([features], dtype=np.float32)
    
    def preprocess_input_sklearn(self, data: Dict[str, Any]) -> Dict[str, np.ndarray]:
        """Convert input data to Sklearn ONNX model format (separate named inputs)."""
        inputs = {}
        
        for feat, dtype in SKLEARN_INPUT_TYPES.items():
            value = data.get(feat)
            
            if dtype == 'string':
                # Handle None/empty strings for categorical features
                if value is None or value == '':
                    value = 'unknown'
                inputs[feat] = np.array([[value]], dtype=object)
            elif dtype == 'int32':
                val = int(value) if value is not None else 0
                inputs[feat] = np.array([[val]], dtype=np.int32)
            else:  # float64
                val = float(value) if value is not None else 0.0
                inputs[feat] = np.array([[val]], dtype=np.float64)
        
        return inputs

    
    def predict(self, model_type: str, data: Dict[str, Any]) -> float:
        """Make a single prediction."""
        # Check if model is loaded
        if not self.is_loaded(model_type):
            raise ValueError(f"Model '{model_type}' not loaded")
        
        # Handle store-specific models
        if model_type in self.store_models:
            store_id = data.get('store_id')
            if store_id not in self.store_models[model_type]:
                raise ValueError(f"Model '{model_type}' not available for store '{store_id}'")
            session = self.store_models[model_type][store_id]
            
            # Use model-specific preprocessing
            if model_type == 'gbtregressor':
                input_array = self.preprocess_input_gbt(data)
            elif model_type == 'hybrid':
                input_array = self.preprocess_input_hybrid(data)
            else:
                input_array = self.preprocess_input_lightgbm(data)
            
            input_name = session.get_inputs()[0].name
            outputs = session.run(None, {input_name: input_array})
        else:
            # Single-file models
            session = self.models[model_type]
            
            if model_type == 'sklearn':
                # Sklearn model expects separate named inputs
                input_dict = self.preprocess_input_sklearn(data)
                outputs = session.run(None, input_dict)
            else:
                # LightGBM model expects single tensor input
                input_array = self.preprocess_input_lightgbm(data)
                input_name = session.get_inputs()[0].name
                outputs = session.run(None, {input_name: input_array})
        
        prediction = outputs[0][0]
        
        # Handle array output
        if hasattr(prediction, '__iter__'):
            prediction = prediction[0]
        
        # Ensure non-negative sales
        return max(0.0, float(prediction))
    
    def predict_batch(self, model_type: str, data_list: List[Dict[str, Any]]) -> List[float]:
        """Make batch predictions."""
        if not self.is_loaded(model_type):
            raise ValueError(f"Model '{model_type}' not loaded")
        
        # For batch, run individual predictions (simpler and more robust)
        results = []
        for data in data_list:
            pred = self.predict(model_type, data)
            results.append(round(pred, 2))
        
        return results


# Global model manager instance
model_manager = ModelManager()
