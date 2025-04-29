# Earthquake Prediction System

A containerized machine learning application for predicting whether an earthquake event will be major or minor based on time series data with different sampling techniques.

## Project Structure

This project consists of two main services:

1. **Python ML Service**: A Flask API that serves the trained machine learning model for earthquake prediction.
2. **Spring Boot Web Application**: A Java web application that provides a user interface and acts as a gateway to the ML service.

## Requirements

- Docker and Docker Compose
- At least 4GB of RAM available for Docker
- Internet connection (for initial image pulling)

## Quick Start

### 1. Export Models from Colab

First, run your machine learning notebook in Google Colab. At the end of your notebook, add the model export code:

```python
# Add this code at the end of your notebook to export models with sampling variants
import os
import json
import shutil
import zipfile
import joblib
import numpy as np
from tensorflow.keras import models

# Create a temporary directory to store models
os.makedirs("export", exist_ok=True)
os.makedirs("export/models", exist_ok=True)

# Model types and sampling techniques
model_types = ['MLP', 'CNN', 'RNN', 'GRU', 'RandomForest']
sampling_techniques = ['original', 'smote', 'under', 'combined']

# List to store all model variants
all_model_variants = []
best_f1_score = -1
best_model_config = None

# Export models
for model_type in model_types:
    # Create directory for this model type
    os.makedirs(f"export/models/{model_type}", exist_ok=True)
    
    for technique in sampling_techniques:
        # Skip RandomForest for now if you have issues with joblib serialization
        if model_type == 'RandomForest' and technique != 'original':
            continue
            
        # Get metrics from your resampling_results
        if model_type in resampling_results and technique in resampling_results[model_type]:
            result = resampling_results[model_type][technique]
            
            # Save model based on your notebook structure
            if model_type in ['MLP', 'CNN', 'RNN', 'GRU']:
                # For neural network models, get the model from your current scope
                # You'll need to modify this to access your trained models
                if technique == 'original':
                    # Example access - replace with your actual code to get the model
                    # model = your_trained_models[model_type][technique]
                    pass
                
                # Save model to export directory
                output_path = f"export/models/{model_type}/{technique}.h5"
                # model.save(output_path)  # Uncomment when you have the model object
                
                # Use placeholder for now
                with open(output_path, 'w') as f:
                    f.write("Placeholder for model file")
            
            elif model_type == 'RandomForest':
                # For RandomForest model
                output_path = f"export/models/{model_type}/{technique}.joblib"
                # joblib.dump(model, output_path)  # Uncomment when you have the model
                
                # Use placeholder for now
                with open(output_path, 'w') as f:
                    f.write("Placeholder for model file")
            
            # Extract metrics
            model_info = {
                'type': model_type,
                'sampling': technique,
                'accuracy': float(result['accuracy']),
                'precision': float(result['precision']),
                'recall': float(result['recall']),
                'f1_score': float(result['f1_score']),
                'roc_auc': float(result['roc_auc']),
                'pr_auc': float(result['pr_auc'])
            }
            
            # Add to all models list
            all_model_variants.append(model_info)
            
            # Check if this is the best model by F1 score
            if model_info['f1_score'] > best_f1_score:
                best_f1_score = model_info['f1_score']
                best_model_config = {
                    'type': model_type,
                    'sampling': technique
                }

# Save the scaler
joblib.dump(scaler, "export/scaler.joblib")

# Create config file with best model as default
config = {
    'active_model': best_model_config,
    'available_models': all_model_variants
}

# Save config file
with open("export/config.json", 'w') as f:
    json.dump(config, f, indent=2)

# Create a zip file of all exported files
with zipfile.ZipFile('earthquake_models.zip', 'w') as zipf:
    for root, dirs, files in os.walk('export'):
        for file in files:
            file_path = os.path.join(root, file)
            arcname = file_path.replace('export/', '')
            zipf.write(file_path, arcname)

# Download the zip file
from google.colab import files
files.download('earthquake_models.zip')

print("Export complete! Downloading earthquake_models.zip")
print(f"Best model: {best_model_config['type']} with {best_model_config['sampling']} sampling (F1 Score: {best_f1_score:.4f})")
```

### 2. Set Up the Project

1. Clone this repository:
   ```bash
   git clone <repository-url>
   cd earthquake-prediction
   ```

2. Extract the downloaded `earthquake_models.zip` to the project directory:
   ```bash
   unzip earthquake_models.zip
   ```

   This should create:
   - `models` directory with subdirectories for each model type
   - `config.json` file with model configuration
   - `scaler.joblib` file for feature scaling

3. Move these files to the Python service directory:
   ```bash
   mv models python-service/
   mv config.json python-service/
   mv scaler.joblib python-service/
   ```

4. Make the startup script executable:
   ```bash
   chmod +x go.sh
   ```

5. Run the application:
   ```bash
   ./go.sh
   ```

6. Open your browser and navigate to:
   ```
   http://localhost:8080
   ```

## Routes and Functionality

The application provides three main routes:

1. **Home Page** (`/`): Landing page with basic information about the application.
2. **Configuration** (`/configuration`): Shows all available models, sampling techniques, and their performance metrics.
3. **Classification** (`/classification`): Upload data, select models with different sampling techniques, and get earthquake predictions.

## Testing the Application

Two test files are provided to demonstrate the functionality:

- `test/positive_test.json`: Features that should predict a major earthquake event.
- `test/negative_test.json`: Features that should predict a minor earthquake event.

You can use these files through the web interface or via API calls:

```bash
# Test positive case (major earthquake)
curl -X POST -H "Content-Type: application/json" -d @test/positive_test.json http://localhost:8080/api/predict

# Test negative case (minor earthquake)
curl -X POST -H "Content-Type: application/json" -d @test/negative_test.json http://localhost:8080/api/predict
```

## Model Selection with Sampling Techniques

The application allows you to select different model architectures with various sampling techniques:

1. **Model Types**:
   - MLP (Multi-Layer Perceptron)
   - CNN (Convolutional Neural Network)
   - RNN (Recurrent Neural Network)
   - GRU (Gated Recurrent Unit)
   - RandomForest

2. **Sampling Techniques**:
   - original: No resampling (original data distribution)
   - smote: SMOTE oversampling of minority class
   - under: Undersampling of majority class
   - combined: Combined SMOTE and undersampling

To select a model:
1. Go to the **Classification** page
2. Choose a model type from the dropdown
3. Choose a sampling technique from the dropdown
4. View the performance metrics for your selection
5. Click "Activate Selected Model" to switch the active model
6. Submit data to see predictions using the newly activated model

## Understanding Sampling Techniques

The project addresses class imbalance in the earthquake dataset using these techniques:

1. **Original Data**:
   - Uses the original imbalanced dataset
   - Applies class weights during training to handle imbalance

2. **SMOTE (Synthetic Minority Over-sampling Technique)**:
   - Creates synthetic examples of the minority class
   - Balances the dataset by increasing minority class samples

3. **Undersampling**:
   - Randomly removes examples from the majority class
   - Balances the dataset by reducing majority class samples

4. **Combined Approach**:
   - Applies both SMOTE and undersampling
   - First increases minority class with SMOTE, then decreases majority class

## API Endpoints

The Spring Boot service exposes the following REST endpoints:

- `POST /api/predict`: Submit features for prediction
- `GET /api/config`: Get model configuration information
- `GET /api/models`: Get list of available models with sampling variants
- `POST /api/models/activate`: Activate a different model with specific sampling technique

## Stopping the Application

To stop the application, run:

```bash
docker-compose down
```

## Troubleshooting

- **Services not starting**: Check the logs with `docker-compose logs`
- **Model loading errors**: Ensure model files and config are correctly placed in the Python service directory
- **Connection issues**: Make sure both containers are running with `docker ps`
- **Missing models**: Re-run the export script in your Colab notebook