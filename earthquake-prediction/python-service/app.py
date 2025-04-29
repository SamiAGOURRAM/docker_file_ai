import os
import numpy as np
import pandas as pd
import json
import joblib
from flask import Flask, request, jsonify
from flask_cors import CORS
import tensorflow as tf
from tensorflow.keras.models import load_model
from model_loader import load_model_by_config, get_available_models, update_active_model

app = Flask(__name__)
CORS(app)  # Enable CORS for all routes

# Global variables for models
MODEL = None
SCALER = None
MODEL_INFO = None

@app.route('/health', methods=['GET'])
def health_check():
    """Health check endpoint."""
    return jsonify({'status': 'healthy', 'service': 'earthquake-prediction-api'})

@app.route('/predict', methods=['POST'])
def predict():
    """
    API endpoint to make earthquake predictions.
    
    Expects JSON input with 'features' key containing time series data.
    Returns prediction result with class and probability.
    """
    # Get the input data
    data = request.get_json(force=True)
    
    if 'features' not in data:
        return jsonify({'error': 'No features provided'}), 400
    
    try:
        # Convert features to numpy array and scale
        features = np.array(data['features']).reshape(1, -1)
        
        # Scale the features
        features_scaled = SCALER.transform(features)
        
        # For neural network models (CNN/RNN/GRU), reshape the data
        if MODEL_INFO['type'] in ["CNN", "RNN", "GRU"]:
            features_scaled = features_scaled.reshape(features_scaled.shape[0], 
                                                     features_scaled.shape[1], 1)
            
            # Make prediction with Keras model
            prediction_proba = MODEL.predict(features_scaled, verbose=0)
            predicted_class = np.argmax(prediction_proba, axis=1)[0]
            probability = float(prediction_proba[0][predicted_class])
            
        elif MODEL_INFO['type'] in ["MLP"]:
            # Make prediction with Keras model (no reshaping needed)
            prediction_proba = MODEL.predict(features_scaled, verbose=0)
            predicted_class = np.argmax(prediction_proba, axis=1)[0]
            probability = float(prediction_proba[0][predicted_class])
            
        elif MODEL_INFO['type'] == "RandomForest":
            # Make prediction with sklearn model
            predicted_class = MODEL.predict(features_scaled)[0]
            probability = float(np.max(MODEL.predict_proba(features_scaled)[0]))
        
        # Determine if it's a major earthquake event (assuming class 1 is a major event)
        is_major_event = bool(predicted_class == 1)
        
        # Create and return the response
        response = {
            'prediction': 'Major Earthquake' if is_major_event else 'Minor Earthquake',
            'class': int(predicted_class),
            'probability': probability,
            'model': MODEL_INFO,
            'status': 'success'
        }
        
        return jsonify(response)
    
    except Exception as e:
        return jsonify({'error': str(e), 'status': 'error'}), 500

@app.route('/config', methods=['GET'])
def get_config():
    """Return the model configuration."""
    if MODEL is None:
        return jsonify({'error': 'Model not loaded', 'status': 'error'}), 500
    
    # Get available models
    available_models = get_available_models()
    
    config = {
        'active_model': MODEL_INFO,
        'available_models': available_models,
        'api_version': '1.0',
        'status': 'success'
    }
    
    # Add extra info for Keras models
    if hasattr(MODEL, 'input_shape'):
        config['input_shape'] = MODEL.input_shape
        
    if hasattr(MODEL, 'output_shape'):
        config['output_shape'] = MODEL.output_shape
    
    return jsonify(config)

@app.route('/models', methods=['GET'])
def get_models():
    """Get list of available models."""
    available_models = get_available_models()
    return jsonify({
        'models': available_models,
        'active_model': MODEL_INFO,
        'status': 'success'
    })

@app.route('/models/activate', methods=['POST'])
def activate_model():
    """Activate a different model with specified sampling technique."""
    data = request.get_json(force=True)
    
    if 'type' not in data or 'sampling' not in data:
        return jsonify({'error': 'Model type and sampling technique required', 'status': 'error'}), 400
    
    model_type = data['type']
    sampling = data['sampling']
    
    # Update config file
    if update_active_model(model_type, sampling):
        # Reload the model
        initialize()
        return jsonify({
            'status': 'success', 
            'message': f'Activated {model_type} model with {sampling} sampling'
        })
    else:
        return jsonify({
            'error': 'Failed to activate model', 
            'status': 'error'
        }), 400

def initialize():
    """Initialize the model and scaler."""
    global MODEL, SCALER, MODEL_INFO
    
    print("Loading model based on configuration...")
    try:
        MODEL, SCALER, MODEL_INFO = load_model_by_config()
        print(f"Loaded {MODEL_INFO['type']} model with {MODEL_INFO['sampling']} sampling successfully")
    except Exception as e:
        print(f"Error loading model: {str(e)}")
        # Try to load any available model as fallback
        available_models = get_available_models()
        if available_models:
            fallback = available_models[0]
            print(f"Trying fallback model: {fallback['type']} with {fallback['sampling']} sampling")
            update_active_model(fallback['type'], fallback['sampling'])
            MODEL, SCALER, MODEL_INFO = load_model_by_config()
            print(f"Loaded fallback model successfully")

if __name__ == '__main__':
    # Initialize the model on startup
    initialize()
    
    # Get port from environment or use default
    port = int(os.environ.get('PORT', 5000))
    
    # Start the Flask app
    app.run(host='0.0.0.0', port=port, debug=False)