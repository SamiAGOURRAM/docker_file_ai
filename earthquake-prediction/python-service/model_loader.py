import os
import json
import numpy as np
import joblib
import tensorflow as tf
from sklearn.preprocessing import StandardScaler
from tensorflow.keras.models import load_model
from tensorflow.keras import Model

def load_model_by_config(config_path='config.json', models_dir='models'):
    """
    Load a model based on the configuration file.
    
    Args:
        config_path (str): Path to the configuration file
        models_dir (str): Path to the directory containing model directories
        
    Returns:
        tuple: (model, scaler, model_info) - The loaded model, scaler, and model info
    """
    # Check if the config file exists
    if not os.path.exists(config_path):
        raise FileNotFoundError(f"Configuration file '{config_path}' not found")
    
    # Load configuration
    with open(config_path, 'r') as f:
        config = json.load(f)
    
    # Get active model configuration
    active_model = config.get('active_model', {})
    model_type = active_model.get('type')
    sampling = active_model.get('sampling', 'original')
    
    if not model_type:
        raise ValueError("Invalid configuration: missing model type")
    
    # Load the model
    model_info = {
        'type': model_type,
        'sampling': sampling
    }
    
    # Define model path based on model type and sampling technique
    if model_type in ['MLP', 'CNN', 'RNN', 'GRU']:
        model_path = os.path.join(models_dir, model_type, f"{sampling}.h5")
        if not os.path.exists(model_path):
            raise FileNotFoundError(f"Model file '{model_path}' not found")
        
        # Load Keras model
        model = load_model(model_path)
        model.name = model_type
    
    elif model_type == 'RandomForest':
        model_path = os.path.join(models_dir, model_type, f"{sampling}.joblib")
        if not os.path.exists(model_path):
            raise FileNotFoundError(f"Model file '{model_path}' not found")
        
        # Load RandomForest model
        model = joblib.load(model_path)
        model.name = model_type
    
    else:
        raise ValueError(f"Unsupported model type: {model_type}")
    
    # Load scaler
    scaler_path = 'scaler.joblib'
    if not os.path.exists(scaler_path):
        raise FileNotFoundError(f"Scaler file '{scaler_path}' not found")
    
    scaler = joblib.load(scaler_path)
    
    return model, scaler, model_info

def get_available_models(config_path='config.json'):
    """
    Get list of available models from the configuration file.
    
    Args:
        config_path (str): Path to the configuration file
        
    Returns:
        list: Available models with their metadata
    """
    if not os.path.exists(config_path):
        return []
    
    with open(config_path, 'r') as f:
        config = json.load(f)
    
    return config.get('available_models', [])

def update_active_model(model_type, sampling, config_path='config.json'):
    """
    Update the active model in the configuration file.
    
    Args:
        model_type (str): Model type (MLP, CNN, RNN, etc.)
        sampling (str): Sampling technique (original, smote, under, combined)
        config_path (str): Path to the configuration file
        
    Returns:
        bool: True if successful, False otherwise
    """
    if not os.path.exists(config_path):
        return False
    
    with open(config_path, 'r') as f:
        config = json.load(f)
    
    # Check if requested model exists in available models
    model_exists = False
    for model in config.get('available_models', []):
        if model['type'] == model_type and model['sampling'] == sampling:
            model_exists = True
            break
    
    if not model_exists:
        return False
    
    # Update active model
    config['active_model'] = {
        'type': model_type,
        'sampling': sampling
    }
    
    # Save updated config
    with open(config_path, 'w') as f:
        json.dump(config, f, indent=2)
    
    return True